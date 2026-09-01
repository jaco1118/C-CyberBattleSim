# Bulk experimental artifacts

The large files used to support the dissertation results are available from OneDrive:

<https://liveuclac-my.sharepoint.com/:f:/g/personal/ucab347_ucl_ac_uk/IgAmnHotK-mlSZLrjn7cKj9yAWANGwV9ileVk1IjlyMB0AY?e=YP9Ix3>

They are stored separately because the result datasets, generated network scenarios, and trained models are too large for the university portal archive or an ordinary Git repository. The portal archive already includes the complete submitted source snapshot, analysis code, provenance records, and compact data; GitHub is supplementary.

The OneDrive directory provides:

- the final large experimental result datasets;
- the selected network topologies and scenarios used by the reported experiments;
- the final checkpoints and VecNormalize state files for the 15 main policies;
- the 19 checkpoint pairs used for the final RQ1c analysis;
- the six exact episode inputs used for RQ3d;
- file maps, provenance records, and checksum manifests.

After downloading the directory, verify it from its root with:

```bash
sha256sum -c BULK_SHA256SUMS
```

On macOS, where `sha256sum` may not be installed, use:

```bash
shasum -a 256 -c BULK_SHA256SUMS
```

`BULK_SHA256SUMS` covers every file in the downloaded directory except the checksum file itself. It passed a complete verification before upload. `BULK_FILE_MAP.tsv` records the source and purpose of each supplied file.
