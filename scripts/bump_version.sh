#!/usr/bin/env bash
#
# Bump the project version everywhere it is recorded, refresh the lockfile
# entry, and verify consistency. Prints the new version as the final stdout
# line so a workflow can capture it with command substitution.
#
# Usage:
#   scripts/bump_version.sh <patch|minor|major|X.Y.Z>
#
# Examples:
#   scripts/bump_version.sh patch     # 0.6.0 -> 0.6.1
#   scripts/bump_version.sh minor     # 0.6.0 -> 0.7.0
#   scripts/bump_version.sh 1.2.3      # explicit
set -euo pipefail

cd "$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

input="${1:-}"
[[ -n "$input" ]] || die "usage: $0 <patch|minor|major|X.Y.Z>"

current=$(grep -m1 -E '^version = "' pyproject.toml | sed -E 's/^version = "([^"]+)".*/\1/')
[[ -n "$current" ]] || die "could not read current version from pyproject.toml"

case "$input" in
  patch|minor|major)
    IFS=. read -r major minor patch <<<"$current"
    case "$input" in
      patch) patch=$((patch + 1)) ;;
      minor) minor=$((minor + 1)); patch=0 ;;
      major) major=$((major + 1)); minor=0; patch=0 ;;
    esac
    new="$major.$minor.$patch"
    ;;
  *)
    new="$input"
    ;;
esac

[[ "$new" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$new' is not a valid semantic version (X.Y.Z)"
[[ "$new" != "$current" ]] || die "new version $new is the same as the current version"

echo "Bumping version: $current -> $new" >&2

current_re=${current//./\\.}

edit() {
  local file=$1; shift
  local tmp; tmp=$(mktemp)
  sed "$@" "$file" >"$tmp" && mv "$tmp" "$file"
}

edit pyproject.toml "s/^version = \"$current_re\"/version = \"$new\"/"
edit Cargo.toml     "s/^version = \"$current_re\"/version = \"$new\"/"
edit CITATION.cff   "s/^version:[[:space:]]*\"\{0,1\}$current_re\"\{0,1\}/version: $new/"
# README: prose token (vX.Y.Z), BibTeX (version = {X.Y.Z}), APA (Version X.Y.Z).
edit README.md      "s/v$current_re/v$new/g; s/version = {$current_re}/version = {$new}/g; s/Version $current_re/Version $new/g"
# Cargo.lock: the version line directly under the "expectation" package entry.
edit Cargo.lock     "/^name = \"expectation\"$/{n;s/^version = \"$current_re\"/version = \"$new\"/;}"

scripts/check_version.sh >&2

echo "$new"
