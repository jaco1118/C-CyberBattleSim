"""Task Z / RQ2B Arm-3 wrapper: ExtremalMask.

Reconstructed from evidence_taskZ.md (0.3). Arm 3 of the three-arm pooling ablation keeps the full
256-d graph_embeddings observation but holds the 128 EXTREMAL pooled dims (max slice [64:128] + min
slice [128:192]) bit-exactly constant at 0.0, so the policy has the same input width as Arm 1 (full
info) but no information in the extremal channels -- the capacity-vs-information control.

CRITICAL (the Task-N failure mode this avoids): the substitution is applied AFTER VecNormalize, i.e.
this wrapper sits OUTSIDE the VecNormalize wrapper. The inner VecNormalize therefore updates its
running stats on the RAW (unmasked) observation exactly as Arm 1 does; only the observation actually
handed to the policy has the extremal dims zeroed. Applying the mask BEFORE VecNormalize (so its
stats see zeros) is the failure mode Task N flagged and is deliberately not done here.

Layout (graph_embeddings, 256-d): mean [0:64], max [64:128], min [128:192], next_escalation [192:256].
The mask zeros [64:192] (max+min). The 64 mean dims and the 64 next_escalation dims are untouched.

Usage (train and eval identically):
    venv = VecNormalize(...)
    venv = ExtremalMask(venv)     # outside VecNormalize
"""
import json
import os

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

EXTREMAL_SLICE = slice(64, 192)   # max [64:128] + min [128:192]
MASK_CONSTANT = 0.0


class ExtremalMask(VecEnvWrapper):
    """Zeroes graph_embeddings[:, 64:192] to 0.0 on every observation handed to the policy.
    Read-only w.r.t. the wrapped venv's dynamics/stats; only rewrites the returned obs dict."""

    def __init__(self, venv):
        super().__init__(venv)
        # Gated pre-flight accounting (Z_PREFLIGHT=1): records the bit-exact assertion the card
        # requires (128 substituted dims, 1 distinct value = 0.0, mean/tail dims unchanged) over the
        # real training obs stream, then writes it to Z_PREFLIGHT_OUT. Inert when Z_PREFLIGHT unset.
        self._pf = os.environ.get("Z_PREFLIGHT") == "1"
        if self._pf:
            self._pf_n = 0
            self._pf_distinct = set()
            self._pf_max_mean = 0.0
            self._pf_max_tail = 0.0

    def _apply(self, obs):
        # obs is a dict {"graph_embeddings": (n_envs, 256), "discrete_features": (n_envs, k)}.
        # Copy so we never mutate the inner venv's buffers in place.
        ge = np.array(obs["graph_embeddings"], copy=True)
        ge[:, EXTREMAL_SLICE] = MASK_CONSTANT
        out = dict(obs)
        out["graph_embeddings"] = ge
        if self._pf:
            src = np.asarray(obs["graph_embeddings"])
            self._pf_distinct.update(np.unique(ge[:, EXTREMAL_SLICE]).tolist())
            self._pf_max_mean = max(self._pf_max_mean, float(np.max(np.abs(src[:, 0:64] - ge[:, 0:64]))))
            self._pf_max_tail = max(self._pf_max_tail, float(np.max(np.abs(src[:, 192:256] - ge[:, 192:256]))))
            self._pf_n += ge.shape[0]
            if self._pf_n % 512 < ge.shape[0]:
                json.dump({"steps": self._pf_n, "distinct_in_masked": sorted(self._pf_distinct),
                           "n_distinct": len(self._pf_distinct), "max_abs_mean_diff": self._pf_max_mean,
                           "max_abs_tail_diff": self._pf_max_tail},
                          open(os.environ.get("Z_PREFLIGHT_OUT", "/tmp/z_preflight.json"), "w"))
        return out

    def reset(self):
        return self._apply(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._apply(obs), rewards, dones, infos
