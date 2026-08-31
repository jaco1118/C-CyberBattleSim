# Copyright (c) 2025.
# Licensed under the MIT License.
"""
event_graph_logger.py
Task-L side logger for CyberBattleCompressedEnv. APPEND-ONLY, self-contained, and completely
independent of the drift CSV (drift_logger.py) -- it never touches, reads, or reorders any drift
column. It records, per logged change-step, the data the four Task-N gates and the probe re-run
need but the drift log never captured:
  - the changed node's identity, its degree in evolving_visible_graph, discovered/owned flags, type
  - the evolving_visible_graph edge list (pre- and post-change) -- the REAL runtime graph
  - the full pre- and post-change observation vectors (graph_embeddings + discrete_features)
  - the candidate action-key set (pre- and post-change)

Everything here is READ-ONLY with respect to environment state and draws NO random numbers. It is
invoked only from the drift-logging path (already gated on drift_logging=True) and only when the
separate event_graph_logging flag is also set, so default runs execute none of it.

Storage (two files, both opened in append mode so no close() hook is required and a crash keeps all
flushed steps):
  <dir>/event_graph.jsonl : one JSON object per logged step; floats live in the side file by offset.
  <dir>/event_obs.f32     : raw little-endian float32 bytes for the observation vectors (EXACT, no
                            float16 -- exact zeros must survive, per the load-bearing null controls).
Join back to the drift CSV on (run_id, seed, episode, step).
"""
import os
import json
import numpy as np


class EventGraphLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.jsonl_path = os.path.join(out_dir, "event_graph.jsonl")
        self.obs_path = os.path.join(out_dir, "event_obs.f32")
        # byte offset into the .f32 file for the next vector written
        self._obs_offset = os.path.getsize(self.obs_path) if os.path.exists(self.obs_path) else 0
        self._n_steps = 0
        self._n_events = 0

    def _write_vec(self, vec):
        """Append a 1-D float32 vector to the raw store; return (offset_bytes, length_floats)."""
        arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32))
        assert arr.ndim == 1
        with open(self.obs_path, "ab") as f:
            f.write(arr.tobytes(order="C"))
        off, n = self._obs_offset, int(arr.shape[0])
        self._obs_offset += arr.nbytes
        return off, n

    @staticmethod
    def _edges(edge_list):
        # node ids -> str so the JSON is stable regardless of the id type
        return [[str(u), str(v)] for (u, v) in edge_list]

    @staticmethod
    def _key(k):
        # a single action key: tuple (source, target, vuln, outcome-ish) -> list of str for JSON
        return [str(x) for x in k]

    @classmethod
    def _action_delta(cls, pre_keys, post_keys):
        # Log the candidate-action-set CHANGE compactly (STEP 0.3: do NOT store the full ~477-key set
        # twice per step -- ~8 GB for the sweep). The full pre/post sets are rebuildable offline from
        # the logged visible graph; the gate needs which actions the change added/removed, which is a
        # handful of keys, plus the pre/post counts.
        pre_set = {tuple(k) for k in pre_keys}
        post_set = {tuple(k) for k in post_keys}
        added = [cls._key(k) for k in post_set - pre_set]
        removed = [cls._key(k) for k in pre_set - post_set]
        return len(pre_set), len(post_set), added, removed

    def log_step(self, *, run_id, seed, scenario_id, episode, step,
                 pre_obs_graph, pre_obs_discrete, post_obs_graph, post_obs_discrete,
                 pre_edges, post_edges, pre_degrees,
                 pre_discovered, pre_owned, pre_action_keys, post_action_keys, events,
                 pre_root=None, distinct_max_holders=None, distinct_min_holders=None):
        """One record per logged change-step. `events` is a list of dicts, one per fired dynamic
        event this step, each: {event_index, change_type, node_ids}. Degree/discovered/owned/root are
        resolved here from the PRE-change captures (a departed node is gone post-change).
        distinct_{max,min}_holders (Task CX B.2): per-step count of distinct nodes holding >=1
        coordinate-wise extreme in the pre-change per-node embeddings, for the effective-p test."""
        pre_g_off, pre_g_n = self._write_vec(pre_obs_graph)
        post_g_off, post_g_n = self._write_vec(post_obs_graph)
        pre_d_off, pre_d_n = self._write_vec(pre_obs_discrete)
        post_d_off, post_d_n = self._write_vec(post_obs_discrete)
        ev_out = []
        for ev in events:
            nid = ev["node_ids"][0] if ev["node_ids"] else None
            ev_out.append({
                "event_index": ev["event_index"],
                "change_type": ev["change_type"],
                "node_ids": [str(n) for n in ev["node_ids"]],
                "changed_node_id": None if nid is None else str(nid),
                # degree in the PRE-change visible graph; None if the node is not in it (e.g. a join,
                # whose node is not yet discovered -> not in evolving_visible_graph). Honest, not 0.
                "changed_node_degree": (int(pre_degrees[nid]) if nid in pre_degrees else None),
                "changed_node_discovered": (None if nid is None else int(nid in pre_discovered)),
                "changed_node_owned": (None if nid is None else int(nid in pre_owned)),
                # Task CX B.1: root-owned status of the changed node at fire, ALONGSIDE owned (not a
                # substitute). None if pre_root not supplied or node id is None. Matches the mech-CSV
                # `was_root` field used for the 69-76% root-departure decomposition.
                "changed_node_root": (None if (nid is None or pre_root is None) else int(nid in pre_root)),
            })
            self._n_events += 1
        rec = {
            "run_id": run_id, "seed": seed, "scenario_id": scenario_id,
            "episode": int(episode), "step": int(step),
            "obs": {  # byte offsets + float counts into event_obs.f32 (float32)
                "pre_graph": [pre_g_off, pre_g_n], "post_graph": [post_g_off, post_g_n],
                "pre_discrete": [pre_d_off, pre_d_n], "post_discrete": [post_d_off, post_d_n],
            },
            "pre_edges": self._edges(pre_edges), "post_edges": self._edges(post_edges),
            "events": ev_out,
        }
        # candidate action set: counts + delta (compact). The exact pre/post sets + 906-d embeddings are NOT
        # logged (they re-bake ~9,700 distinct vectors/episode -> ~700-1000 GB; and per-node embeddings cannot
        # reconstruct them because action embeddings are frozen at discovery via processed_pairs). The
        # observation-channel / total-effect behavioural terms are instead obtained by deterministic REPLAY
        # (Task L DECISION 1, verified byte-identical) rather than from disk.
        n_pre, n_post, added, removed = self._action_delta(pre_action_keys, post_action_keys)
        rec["action_keys_pre_count"] = n_pre
        rec["action_keys_post_count"] = n_post
        rec["action_keys_added"] = added
        rec["action_keys_removed"] = removed
        # Task CX B.2: distinct extremal-holder counts (per-step; one int each for max and min).
        rec["distinct_max_holders"] = (None if distinct_max_holders is None else int(distinct_max_holders))
        rec["distinct_min_holders"] = (None if distinct_min_holders is None else int(distinct_min_holders))
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._n_steps += 1

    def log_episode(self, *, run_id, seed, scenario_id, episode, record):
        """Task CX B (per-episode record). One JSON object per completed episode in event_episode.jsonl.
        `record` is a dict of already-computed episode-terminal quantities (score components, owned/root
        counts, alive counts, discovered counts, length, termination reason, per-type + root-departure
        counts). Raw components are logged so BOTH score forms (count-based and ownership ratio) are exact
        and derivable offline. Append-only, independent of the drift CSV and event_graph.jsonl."""
        path = os.path.join(self.out_dir, "event_episode.jsonl")
        out = {"run_id": run_id, "seed": seed, "scenario_id": scenario_id, "episode": int(episode)}
        out.update(record)
        with open(path, "a") as f:
            f.write(json.dumps(out) + "\n")

    @property
    def stats(self):
        return {"steps": self._n_steps, "events": self._n_events, "obs_bytes": self._obs_offset}
