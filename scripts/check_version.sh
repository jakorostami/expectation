#!/usr/bin/env bash
#
# Verify that the project version is identical across every manifest and that
# the README references it. Exits non-zero with a plain error message on any
# mismatch.
#
# Single source of truth: pyproject.toml [project].version. All other files
# must agree with it. Invoked by .github/workflows/ci.yml and by
# scripts/bump_version.sh after a bump.
#
# Usage: scripts/check_version.sh
set -euo pipefail

# Run from the repository root regardless of the caller's working directory.
cd "$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"

# First `version = "X.Y.Z"` line (the [project]/[package] key sits at the top).
toml_version() { grep -m1 -E '^version = "' "$1" | sed -E 's/^version = "([^"]+)".*/\1/'; }

pyproject=$(toml_version pyproject.toml)
cargo=$(toml_version Cargo.toml)
citation=$(grep -m1 -E '^version:' CITATION.cff | sed -E 's/^version:[[:space:]]*"?([^"[:space:]]+)"?.*/\1/')
# The [[package]] block named "expectation" in the committed lockfile.
lock=$(awk '/^name = "expectation"$/{f=1; next} f && /^version = /{gsub(/[",]/, "", $3); print $3; exit}' Cargo.lock)

fail() {
  echo "ERROR: $*" >&2
  {
    echo "Versions found:"
    printf '  %-16s %s\n' \
      "pyproject.toml" "${pyproject:-<none>}" \
      "Cargo.toml"     "${cargo:-<none>}" \
      "CITATION.cff"   "${citation:-<none>}" \
      "Cargo.lock"     "${lock:-<none>}"
  } >&2
  exit 1
}

for pair in "pyproject.toml=$pyproject" "Cargo.toml=$cargo" "CITATION.cff=$citation" "Cargo.lock=$lock"; do
  [[ -n "${pair#*=}" ]] || fail "could not read a version from ${pair%%=*}"
done

if [[ "$cargo" != "$pyproject" || "$citation" != "$pyproject" || "$lock" != "$pyproject" ]]; then
  fail "version mismatch across manifests"
fi

# The README carries the version in prose and citation blocks; keep it in sync.
for token in "v$pyproject" "Version $pyproject" "version = {$pyproject}"; do
  grep -qF "$token" README.md || fail "README.md is missing the expected version token '$token'"
done

echo "Version consistent across manifests and README: $pyproject"
