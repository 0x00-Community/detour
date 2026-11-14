#!/usr/bin/env dash
# Regenerate crates/intel/dialect/tests/eu.expected: assemble the gentool text in the comment above each op of
# crates/intel/dialect/tests/eu.mlir with Mesa's gentool for Panther Lake, one instruction at a time, as a line of hex.
# GENTOOL: how to run gentool (`gentool asm -p ptl`, built from a Mesa tree with -Dtools=intel).
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
gentool=${GENTOOL:?set GENTOOL to the gentool binary}
cd "$root"
out=crates/intel/dialect/tests/eu.expected
: > "$out"
sed -n 's|^// ||p' crates/intel/dialect/tests/eu.mlir | while IFS= read -r line; do
  hex=$(printf '%s\n' "$line" | $gentool asm -p ptl -i - -o - | od -An -tx1 -v | tr -s ' \n' ' ' | sed 's/^ //; s/ $//')
  [ -n "$hex" ] || { echo "gentool rejected: $line" >&2; exit 1; }
  echo "$hex" >> "$out"
done
wc -l "$out"
