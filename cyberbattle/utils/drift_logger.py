# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    drift_logger.py
    Diagnostic-only CSV logger for CyberBattleCompressedEnv's drift instrumentation
    (see CyberBattleCompressedEnv's drift_logging constructor kwarg). Buffers rows in
    memory and flushes periodically rather than writing per-step, to keep the overhead
    of running with drift_logging=True bounded.
"""

import csv
import os

# Fixed column order for the drift log. Kept as a module-level constant (not derived from
# whatever keys happen to be in the first logged row) so every flush -- even the first, even if
# the caller logs rows with a different key ordering across steps -- writes columns in the same
# stable order, and any accidentally-missing/extra key is caught explicitly rather than silently
# reordering the header.
DRIFT_LOG_COLUMNS = [
    "run_id", "seed", "scenario_id", "episode", "step",
    "n_scenario", "n_discovered", "n_discovered_h1", "n_discovered_h2", "n_discovered_h3",
    "change_type", "change_fired", "n_touched_nodes", "relevant",
    # Deferred change attribution (membership_join's touched node may not be in the discovered/
    # pooled set at fire time -- see CyberBattleCompressedEnv._log_drift_rows/_log_attribution_rows):
    #   event_phase: "immediate" (fired and visible same step) | "fired" (fired, >=1 touched node
    #       not yet visible -- a pending registry entry now exists for it) | "attributed" (a
    #       previously-pending node, OR an ordinary never-removed node, became visible this step)
    #       | "no_change" (the encoder determinism sanity-check row, no event at all)
    #   event_id: shared between a "fired" row and its later "attributed"/flush row, so they can
    #       be joined; None for "immediate"/"no_change" rows and for ordinary-discovery
    #       "attributed" rows (node_origin_is_join=False) which were never a tracked event
    #   step_fired / visibility_lag_steps: only meaningful for "fired" and "attributed" rows tied
    #       to an actual pending event (visibility_lag_steps = step_attributed - step_fired)
    #   touched_node_visible: whether ALL touched nodes of this event were already visible (in
    #       h2, i.e. before this step's dynamic mutation) at fire time; None for "no_change" rows
    #   attributed: None while a "fired" row's resolution is still unknown, True once resolved
    #       (immediate or attributed rows), False on the end-of-episode flush of a never-resolved event
    #   node_origin_is_join: distinguishes a joined donor node's first-visibility transition from
    #       an ordinary pre-existing node's first-visibility transition -- the free control group
    #       from Step 4 of the deferred-attribution spec (None where not applicable)
    "event_phase", "event_id", "step_fired", "visibility_lag_steps",
    "touched_node_visible", "attributed", "node_origin_is_join",
    "agent_drift_full", "change_drift_full",
    # per-slice columns are appended dynamically by DriftLogger.__init__ from the env's
    # graph_embeddings_aggregations list, immediately after the two "_full" columns above
]


class DriftLogger:
    """Buffers drift-instrumentation rows and flushes them to a CSV file periodically.

    If csv_path is None, rows are only kept in memory (self.rows) -- useful for unit tests
    that want to inspect logged rows directly without touching disk.
    """

    def __init__(self, csv_path, aggregation_names, flush_every=200):
        self.csv_path = csv_path
        self.flush_every = flush_every
        self.rows = []  # rows not yet flushed (also the full in-memory history if csv_path is None)
        self._all_rows_ever = [] if csv_path is None else None
        self.columns = list(DRIFT_LOG_COLUMNS)
        insert_at = self.columns.index("change_drift_full") + 1
        # per-slice columns, named from the aggregation list so this does not break if
        # pooling_mode/graph_embeddings_aggregations changes later (e.g. mean/min/max today,
        # something else tomorrow)
        per_slice_columns = []
        for agg in aggregation_names:
            per_slice_columns.append(f"agent_drift_{agg}")
        for agg in aggregation_names:
            per_slice_columns.append(f"change_drift_{agg}")
        for agg in aggregation_names:
            per_slice_columns.append(f"attenuation_ratio_{agg}")
        self.columns = self.columns[:insert_at] + per_slice_columns + self.columns[insert_at:]
        self.columns += [
            "norm_h1", "norm_h2", "norm_h3",
            "delta_h_v_norm", "attenuation_ratio_full",
            "action_referenced_removed_entity", "agent_action_succeeded",
            "pooling_mode",
            "connectivity_event_fired", "connectivity_n_touched_nodes",
        ]
        # Per-slice ABSOLUTE norms (Phase 1 follow-up): appended at the very end so every
        # existing column keeps its position. Without these, absolute drift is only
        # reconstructable for the "full" slice (norm_h2 * change_drift_full) -- mean/max/min have
        # no per-slice norm anywhere, so absolute per-slice drift is not recoverable at all without
        # them. Snapshot-major order (all of h1, then all of h2, then all of h3), named from the
        # aggregation list for the same forward-compat reason as the per-slice columns above.
        for snapshot_label in ("h1", "h2", "h3"):
            for agg in aggregation_names:
                self.columns.append(f"norm_{snapshot_label}_{agg}")
        if csv_path:
            directory = os.path.dirname(csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def log(self, row: dict):
        missing = set(self.columns) - set(row.keys())
        extra = set(row.keys()) - set(self.columns)
        if missing or extra:
            raise ValueError(f"drift log row schema mismatch: missing={missing}, extra={extra}")
        self.rows.append(row)
        if self._all_rows_ever is not None:
            self._all_rows_ever.append(row)
        if self.csv_path and len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.csv_path or not self.rows:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerows(self.rows)
        self.rows = []

    def close(self):
        self.flush()

    @property
    def all_rows(self):
        """All rows logged so far, regardless of flush state. Only meaningful when
        csv_path is None (in-memory mode) -- otherwise flushed rows are on disk, not here."""
        if self._all_rows_ever is not None:
            return self._all_rows_ever
        return list(self.rows)
