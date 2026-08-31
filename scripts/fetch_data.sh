#!/usr/bin/env bash
#
# Download the IO-VNBD dataset into data/raw/.
#
# The upstream repo stores every CSV in Git LFS. Without `git lfs install`
# having been run first, a clone yields ~130-byte pointer files instead of
# data, and every downstream load silently reads garbage. This script refuses
# to finish in that state.
#
# Full download is roughly 1.8 GB.

set -euo pipefail

REPO_URL="https://github.com/onyekpeu/IO-VNBD.git"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# data/raw/ sits inside the git-ignored data/ directory, so the nested .git the
# clone creates is invisible to the outer repo. No accidental submodule, and no
# need for a gitlink entry.
DEST="$REPO_ROOT/data/raw"

# Pinned verification targets.
#
# Sizes are the `size` field of each file's Git LFS pointer at upstream HEAD,
# read August 2026. They are exact, not thresholds, so a truncated or
# interrupted download is caught as well as a pointer-only checkout.
#
# The three cover both top-level trees (Synchronised and Unsynchronised) and one
# file from each split -- S-M is train (Ford Fiesta / Huawei P20 Pro), S-T1 is
# validate (Renault Megane / Moto G7 Power), S-A1 is test (Volvo XC70 /
# Blackberry Priv). A partial clone that drops a whole tree or a whole split
# therefore fails here, at fetch time, instead of much later during dataset
# building.
#
# If upstream is ever updated these numbers go stale and the script will report
# a spurious SIZE MISMATCH. The repo has had four commits in two years, so this
# is unlikely -- but if it happens, re-read the pointer sizes at the new HEAD
# and update the constants below. Note the "abd" in the first path: that typo is
# upstream's, not ours.
EXPECT_1_BYTES=19798721
EXPECT_1_PATH="Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
EXPECT_2_BYTES=3801675
EXPECT_2_PATH="Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-T1.csv"
EXPECT_3_BYTES=1072488
EXPECT_3_PATH="Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-A1.csv"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- 1. git-lfs must be present and initialised -----------------------------

command -v git >/dev/null 2>&1 || die "git is not installed."

if ! git lfs version >/dev/null 2>&1; then
    cat >&2 <<'MSG'
error: git-lfs is not installed.

The IO-VNBD repo stores every CSV in Git LFS. Cloning without it gives you
~130-byte pointer files, not data. Install it, then re-run this script:

  macOS:          brew install git-lfs
  Debian/Ubuntu:  sudo apt-get install git-lfs
  Fedora:         sudo dnf install git-lfs
  other:          https://git-lfs.com

MSG
    exit 1
fi

# Registers the LFS smudge/clean filters for this user. Must happen BEFORE the
# clone, or the checkout writes pointers. Safe to run repeatedly.
git lfs install --skip-repo

# --- 2. clone ---------------------------------------------------------------

if [ -e "$DEST" ]; then
    die "$DEST already exists. Remove it to re-download, or leave it as is."
fi

mkdir -p "$(dirname "$DEST")"
printf 'Cloning IO-VNBD into %s (~1.8 GB, this will take a while)...\n' "$DEST"
git clone --depth 1 "$REPO_URL" "$DEST"

# Belt and braces: if the clone ran with LFS filters somehow disabled, this
# pulls the real objects in after the fact.
git -C "$DEST" lfs pull

# --- 3. verify we got data, not pointers ------------------------------------

# Three distinguishable failures: absent (partial clone / wrong path), under
# 1 KB (LFS pointer, `git lfs install` was skipped), wrong size (truncated).
verify() {
    local expected="$1" path="$DEST/$2"
    if [[ ! -f "$path" ]]; then
        echo "MISSING: $2" >&2
        echo "         partial clone, or the upstream path has moved." >&2
        return 1
    fi
    local actual
    actual=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")
    if [[ "$actual" -eq "$expected" ]]; then
        printf 'OK   %10s bytes  %s\n' "$actual" "$2"
    elif [[ "$actual" -lt 1024 ]]; then
        echo "LFS POINTER ($actual B) -- run 'git lfs install', then re-clone: $2" >&2
        return 1
    else
        echo "SIZE MISMATCH: expected $expected, got $actual -- likely truncated: $2" >&2
        return 1
    fi
}

printf '\nVerifying LFS contents...\n'

failed=0
verify "$EXPECT_1_BYTES" "$EXPECT_1_PATH" || failed=1
verify "$EXPECT_2_BYTES" "$EXPECT_2_PATH" || failed=1
verify "$EXPECT_3_BYTES" "$EXPECT_3_PATH" || failed=1

if [[ "$failed" -ne 0 ]]; then
    cat >&2 <<MSG

error: dataset verification failed. See the messages above.

A pointer-only checkout is fixed by re-cloning with LFS enabled:

  git lfs install
  rm -rf "$DEST"
  $0

MSG
    exit 1
fi

printf '\nOK -- %s fetched to %s\n' "$(du -sh "$DEST" | cut -f1)" "${DEST#"$REPO_ROOT"/}"
