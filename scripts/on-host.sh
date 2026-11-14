#!/usr/bin/env dash
# Run a command in a copy of this tree on the test machine (an xe GPU): rsync, then `cd ~/detour && <command>`.
# DETOUR_HOST=user@host (required), DETOUR_DIR (default detour, relative to the remote home).
set -eu
: "${DETOUR_HOST:?set DETOUR_HOST=user@host}"
dir=${DETOUR_DIR:-detour}
root=$(cd "$(dirname "$0")/.." && pwd)
rsync -a --delete --exclude /target --exclude .git --exclude .DS_Store "$root/" "$DETOUR_HOST:$dir/"
exec ssh "$DETOUR_HOST" "cd $dir && { [ ! -f ~/.cargo/env ] || . ~/.cargo/env; } && $*"
