# Archived workflows

These workflow files are retained for reference but are intentionally **not**
active: GitHub Actions only runs workflows located directly in
`.github/workflows/`, so nothing here is triggered.

- `run-tests.yml` — original single-job test workflow (Python 3.11, installed
  the now-removed `requirements.txt`). Superseded by `ci.yml` (matrix over
  ubuntu/macos/windows × Python 3.12/3.13, plus lint, rust, and min-deps gates).
- `release.yml` — original single dispatch-driven release (bump + tag + build +
  publish in one run). Superseded by the split `bump.yml` (version bump + tag)
  and `release.yml` (tag-triggered build/verify/publish).
