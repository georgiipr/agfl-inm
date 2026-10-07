# Reader-friendly experiment report

Build the TeX report and PDF with:

```sh
make -C docs/published-covariance
```

The report summarizes the completed exploratory study. Its scientific and
reproducibility details are supported by the adjacent audit documents and the
accepted output at `results/published-covariance-v1-cuda-root/`.

For a fresh run by a collaborator, see
[`friend-reproduction.md`](friend-reproduction.md). The nine frozen upstream
checkpoints and their pinned source manifest are in `checkpoints/` and
`checkpoints-origin.json`.
