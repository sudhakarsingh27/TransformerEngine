# MegaUP operation timings

[Open the interactive charts](https://sudhakarsingh27.github.io/TransformerEngine/).

This standalone page compares separate Transformer Engine GEMM + NCCL all-to-all operations with MegaUP fusion across four projection/communication operations around attention. These are operation timings, not full attention or model benchmarks. Timing summaries and configuration geometry are embedded in the HTML.

Open `docs/index.html` locally to preview; no build or server is required.

GitHub Pages publishes `docs/` from the `sudhakars/megaup-pages` branch. Update `docs/index.html` and push to that branch to redeploy. This branch contains only the visualization and is independent of the TE source branches.
