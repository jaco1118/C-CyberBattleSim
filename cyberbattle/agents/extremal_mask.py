"""Task Z / RQ2B Arm-3 wrapper (+ O8/Arm-4 extension): ExtremalMask.

Reconstructed from evidence_taskZ.md (0.3). Arm 3 of the three-arm pooling ablation keeps the full
256-d graph_embeddings observation but holds the 128 EXTREMAL pooled dims (max slice [64:128] + min
slice [128:192]) bit-exactly constant at 0.0, so the policy has the same input width as Arm 1 (full
info) but no information in the extremal channels -- the capacity-vs-information control.

O8/Arm 4 EXTENSION (2026-08-04): the same wrapper, parameterized to mask the FULL_SLICE [0:256]
instead of EXTREMAL_SLICE, zeroes every graph_embeddings dim -- the zero-graph-info floor. Same
class, same post-VecNormalize placement, same bit-exact-zero guarantee; only which dims are masked
changes. mask_slice defaults to EXTREMAL_SLICE (Arm 3) so existing callers are unaffected.

CRITICAL (the Task-N failure mode this avoids): the substitution is applied AFTER VecNormalize, i.e.
this wrapper sits OUTSIDE the VecNormalize wrapper. The inner VecNormalize therefore updates its
running stats on the RAW (unmasked) observation exactly as Arm 1 does; only the observation actually
handed to the policy has the masked dims zeroed. Applying the mask BEFORE VecNormalize (so its stats
see zeros) is the failure mode Task N flagged and is deliberately not done here.

Layout (graph_embeddings, 256-d): mean [0:64], max [64:128], min [128:192], next_escalation [192:256].
Arm 3 (EXTREMAL_SLICE) zeros [64:192] (max+min) -- mean and next_escalation dims untouched.
Arm 4 (FULL_SLICE) zeros [0:256] (everything) -- no dims untouched.

Usage (train and eval identically):
    venv = VecNormalize(...)
    venv = ExtremalMask(venv)                          # Arm 3 (default)
    venv = ExtremalMask(venv, mask_slice=FULL_SLICE)   # Arm 4
"""
import json
import os

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

EXTREMAL_SLICE = slice(64, 192)   # Arm 3: max [64:128] + min [128:192]
FULL_SLICE = slice(0, 256)        # Arm 4 (O8): all graph_embeddings dims -- zero-graph-info floor
MASK_CONSTANT = 0.0


class ExtremalMask(VecEnvWrapper):
    """Zeroes graph_embeddings[:, mask_slice] to 0.0 on every observation handed to the policy.
    Read-only w.r.t. the wrapped venv's dynamics/stats; only rewrites the returned obs dict."""

    def __init__(self, venv, mask_slice=EXTREMAL_SLICE):
        super().__init__(venv)
        self._slice = mask_slice
        # Gated pre-flight accounting (Z_PREFLIGHT=1): records the bit-exact assertion the card
        # requires (masked dims -> 1 distinct value = 0.0, unmasked dims unchanged) over the real
        # training obs stream, then writes it to Z_PREFLIGHT_OUT. Inert when Z_PREFLIGHT unset.
        self._pf = os.environ.get("Z_PREFLIGHT") == "1"
        if self._pf:
            self._pf_n = 0
            self._pf_distinct = set()
            self._pf_max_outside = 0.0

    def _apply(self, obs):
        # obs is a dict {"graph_embeddings": (n_envs, 256), "discrete_features": (n_envs, k)}.
        # Copy so we never mutate the inner venv's buffers in place.
        ge = np.array(obs["graph_embeddings"], copy=True)
        ge[:, self._slice] = MASK_CONSTANT
        out = dict(obs)
        out["graph_embeddings"] = ge
        if self._pf:
            src = np.asarray(obs["graph_embeddings"])
            self._pf_distinct.update(np.unique(ge[:, self._slice]).tolist())
            outside = np.ones(ge.shape[1], dtype=bool)
            outside[self._slice] = False
            if outside.any():
                self._pf_max_outside = max(self._pf_max_outside,
                                            float(np.max(np.abs(src[:, outside] - ge[:, outside]))))
            self._pf_n += ge.shape[0]
            if self._pf_n % 512 < ge.shape[0]:
                json.dump({"steps": self._pf_n, "masked_width": int((~outside).sum()),
                           "distinct_in_masked": sorted(self._pf_distinct),
                           "n_distinct": len(self._pf_distinct),
                           "max_abs_outside_diff": self._pf_max_outside},
                          open(os.environ.get("Z_PREFLIGHT_OUT", "/tmp/z_preflight.json"), "w"))
        return out

    def reset(self):
        return self._apply(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._apply(obs), rewards, dones, infos
