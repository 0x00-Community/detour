#!/bin/sh
# What Metal's driver asks macOS's GPU kernel driver (AGXG16G, IOKit's IOGPU) for one run of a program: every
# IOServiceOpen / IOConnectCall* / IOConnectTrap* / IOConnectMapMemory64 into $OUT.<pid>, and at each submission
# (IOConnectTrap3 selector 0) the submit struct, the memory it points at, and every allocation's non-zero bytes
# (sel=10 GPU memory, sel=15 shared pages) into $OUT.submit.<pid>.<n>. The program must not have the hardened runtime
# (DYLD_INSERT_LIBRARIES): one built here (cargo, swiftc) is ad hoc signed.
#   scripts/apple-iogpu-trace.sh OUT PROGRAM [ARGS...]
set -eu
cd "$(dirname "$0")/.."
out=$1; shift
mkdir -p target
clang -dynamiclib -O1 -framework IOKit scripts/apple-iogpu-trace.c -o target/apple-iogpu-trace.dylib
IOTRACE=$out DYLD_INSERT_LIBRARIES=$PWD/target/apple-iogpu-trace.dylib "$@" </dev/null
