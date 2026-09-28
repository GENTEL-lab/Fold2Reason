# Contributing

Keep core implementations in the corresponding package rather than adding one-off root scripts. Add command defaults to `configs/` and document new data contracts under `docs/`.

Run `python -m unittest discover -s tests -v` and `python tools/check_release.py` before submitting a change. Tests should use synthetic inputs and avoid network access, credentials, model downloads, and GPU allocation by default.

Changes to prompt rendering, token alignment, geometry scoring, checkpoint loading, or dataset selection require an explicit compatibility note and a regression test. Keep benchmark sample IDs and model comparisons paired; report the scope of any validation you ran.

The project license and public contribution channel are pending maintainer confirmation; finalize them before accepting external contributions.

