#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build results
find src -name '*.java' | sort > build/sources.txt
javac --release 17 -encoding UTF-8 -d build @build/sources.txt
exec java -Xms128m -Xmx2g -cp build dtw.kpz.KpzEquivalenceStudy "$@"
