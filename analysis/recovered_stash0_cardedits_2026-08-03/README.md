# Recovered uncommitted content from stash@{0} (2026-08-03)

**What this is:** a faithful copy of the diff held in `git stash@{0}` (label:
"On taskY-probe-n90: pre-existing card edits (not RQ2B)"), preserved because an
uncommitted stash can be silently dropped/overwritten by a later `git stash`.

**Contents (671 insertions, 0 deletions, across 3 files):**
- `evidence_cards/evidence_taskCX.md` (+588 lines) — includes the RQ3(d) §3.9 loss-concentration
  text with the reported 0.70/1.23 root-departure and 18%/32%/34% top-decile figures, plus other
  CX PART 3/4 findings.
- `evidence_cards/evidence_taskAN.md` (+56 lines)
- `evidence_cards/evidence_taskF4.md` (+27 lines)

**IMPORTANT — what this is NOT:** this is evidence-card TEXT/findings only. It does **not** contain
the CX analysis *pipeline code* that computed 0.70/1.23 / 18-32-34 — that code was in ephemeral job
scratch and is not recovered here (still lost). The figures are documented; the script that produced
them is not.

**Status:** recovered as-is from stash@{0} on 2026-08-03. NOT reviewed, re-run, or validated. The
original stash was left in place. To reconstruct the edited cards: `git apply` this patch onto the
committed base of those three files.
