# MSc dissertation artifacts

These files accompany the dissertation *Evaluating Robustness to Network Change in Autonomous Penetration Testing: Representation, Exploration, and Scale in a GNN–RL Agent* by Chan Sau Lai, submitted for the UCL MSc Information Security in 2026.

The university portal archive is the primary submission and includes a complete source snapshot. This GitHub repository is a supplementary way to inspect the same source and analysis code. Large experimental outputs are available separately from OneDrive:

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucab347_ucl_ac_uk/IgAmnHotK-mlSZLrjn7cKj9yAWANGwV9ileVk1IjlyMB0AY?e=YP9Ix3>

`EXTERNAL_BULK_ARTIFACTS.md` describes the external files and their checksum manifest.

## Finding the dissertation materials

The analysis programs and compact outputs are in `analysis/`, with supporting records in `evidence_cards/`. The `artifact_provenance/` directory records the source version and maps the final analyses to their inputs and checkpoints. Six compact inputs used by the final RQ3d analysis are included in `artifact_data/rq3d_inputs/`.

To create the software environment, use either:

```bash
conda env create -f environment.yml
```

or:

```bash
pip install -r requirements.txt
```

API credentials are not included. They are unnecessary for reading the supplied results or using the preserved scenarios in the external archive; they are needed only for collecting new Shodan or NVD data.

## Source version

The dissertation source has the following identity:

- Canonical GitHub merge commit: `d0153c6e0b21f96b6ddf5eaccfdc6f92b0f6ab82`
- Audited local source commit: `8c2862ba2ba7424bb27e3ca67c5a9630a486bbde`
- Git tree: `358ed75c2ba43d0d16a615b619f905fc9cbf3e61`

The two commits have the same Git tree, so their checked-in file contents are identical. Two tracked macOS metadata files, `evidence_cards/._.DS_Store` and `evidence_cards/._evidence_taskF1.md`, remain in that historical tree but have no scientific role.

## Principal analysis locations

- RQ1: `analysis/rq1a_regression_recovered_2026-08-07/`, `analysis/rq1b_mech_split_scale_2026-08-08/`, `analysis/nodecount_ci_2026-08-09/`, `analysis/nodecount_ci_n60_2026-08-16/`, and `analysis/rq1c_mde_2026-08-10/`.
- RQ2: `analysis/graph_depth_2026-08-10/decomposition_wide/`, `analysis/three_lookups_2026-08-15/`, the committed RQ2c analysis, and `evidence_cards/taskZ_raw/`.
- RQ3: `analysis/rq3a_gate_recompute_2026-08-09/`, `analysis/rq3b_slice_recompute_2026-08-09/`, `analysis/rq3c_rebuild_2026-08-10/`, and `analysis/rq3d_ranking_overlap_2026-08-22/`.

Paths in older run records may refer to their original server locations. Replace those prefixes with the path to your checkout or downloaded archive.

## Reproducibility notes

The evidence cards and files in `artifact_provenance/` give the commands, filters, seeds, and qualifications for individual results. OneDrive holds the selected large results, exact topologies, and final checkpoints needed for closer inspection and reproduction.

Some results should be interpreted with care. The main policies did not meet the final convergence rule; RQ1c has limited statistical power; the shared join donor pool is a confounding factor; and RQ2b is inconclusive for networks with 80--100 nodes. Stored step-3 actions do not provide a bit-for-bit replay of every run, so the relevant evidence supports distributional rather than exact trajectory reproduction. The final RQ3d result is the same-episode reranking analysis in `analysis/rq3d_ranking_overlap_2026-08-22/`.

Further detail is available in the records under `evidence_cards/`. The exact RQ1c checkpoints and RQ3d inputs are listed in `artifact_provenance/RQ1C_FINAL_CHECKPOINT_ALLOWLIST.tsv` and `artifact_provenance/RQ3D_FINAL_INPUTS.tsv`.
