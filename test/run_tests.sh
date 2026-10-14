#!/usr/bin/env bash
# =============================================================================
# Geodesic thorn testsuite launcher (official Einstein Toolkit mechanism).
#
# Usage:
#   ./run_tests.sh         default set: the 8 regression tests that cover
#                          all four metric-sampling back-ends and both
#                          metric families (stationary Kerr grid,
#                          non-stationary TwoPunctures merger):
#
#                            Geodesic/conv100
#                            Geodesic/conv100_l27
#                            Geodesic/conv100_local27
#                            Geodesic/conv100_aei
#                            Geodesic/conv100_carpet
#                            Geodesic/circ_orbit
#                            Geodesic/tp_local27_025
#                            Geodesic/tp_far_local27_025_t80
#
#   ./run_tests.sh -all    every test listed in test/test.ccl (18 tests,
#                          including the AMR / MPI variants and the aei /
#                          carpet TwoPunctures probes).
#
# The script drives the standard Cactus testsuite infrastructure
# (lib/sbin/RunTest.pl via "gmake sim-testsuite"), non-interactively
# (PROMPT=no).  It requires a Cactus installation whose
# arrangements/EinsteinAnalysis/Geodesic points at this thorn and whose
# "sim" configuration is built (gmake sim -> $CCTK/exe/cactus_sim).
#
# Machine profile (Einstein Toolkit convention: the user declares the
# local machine and every test instance is launched to fit inside it;
# see Cactus Users' Guide B2.6, the CCTK_TESTSUITE_* variables below).
# On a single node the machine is auto-detected with "nproc" (cluster
# users: run one node's worth of cores at a time, or wrap the script in
# their job scheduler).  The instance size keeps ~9% headroom:
#
#   88-core node: 4 ranks x 20 threads (4p*20t) = 80 of 88 cores.
#
# Tests run one at a time (RunTest.pl's default;
# CCTK_TESTSUITE_PARALLEL_TESTS is not set).  OMP_NUM_THREADS must be
# pinned: leaving it unset makes OpenMP take ALL cores per rank, a 4x
# oversubscription that slows the TwoPunctures t = 0 puncture solve by
# an order of magnitude (and the reference data were generated with
# 20 threads, so 20 is also the reproducibility value on this node).
#
# Environment overrides:
#   CACTUS_HOME          Cactus root (default: four levels above this
#                        script, i.e. the Cactus installation itself).
#   NPROC_TOTAL          total cores of the node (default: nproc).
#   NPROCS_PER_TEST      MPI ranks per test instance (default 4; tests
#                        with an explicit NPROCS in test.ccl keep it).
#   TEST_THREADS         OpenMP threads per rank (default: auto,
#                        (NPROC_TOTAL/NPROCS_PER_TEST) * 91/100).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACTUS_HOME="${CACTUS_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
if [ ! -r "$CACTUS_HOME/lib/sbin/RunTest.pl" ]; then
    echo "error: $CACTUS_HOME/lib/sbin/RunTest.pl not found" >&2
    echo "       (set CACTUS_HOME to your Cactus installation root)" >&2
    exit 1
fi
if [ ! -x "$CACTUS_HOME/exe/cactus_sim" ]; then
    echo "error: $CACTUS_HOME/exe/cactus_sim not found; build the" >&2
    echo "       'sim' configuration first:  gmake -C $CACTUS_HOME sim" >&2
    exit 1
fi

CONFIG="sim"

DEFAULT_TESTS="Geodesic/conv100 \
  Geodesic/conv100_l27 \
  Geodesic/conv100_local27 \
  Geodesic/conv100_aei \
  Geodesic/conv100_carpet \
  Geodesic/circ_orbit \
  Geodesic/tp_local27_025 \
  Geodesic/tp_far_local27_025_t80"

case "${1:-}" in
    ""|-default|--default)
        export CCTK_TESTSUITE_RUN_TESTS="$DEFAULT_TESTS"
        ;;
    -all|--all)
        # Unset => RunTest.pl runs every test found in test/test.ccl.
        unset CCTK_TESTSUITE_RUN_TESTS
        ;;
    *)
        echo "usage: $0 [-all]" >&2
        exit 2
        ;;
esac

export PROMPT=no
# --- machine profile: auto-detected, one test instance at a time ---
NPROC_TOTAL="${NPROC_TOTAL:-$(nproc 2>/dev/null || echo 8)}"
NPROCS_PER_TEST="${NPROCS_PER_TEST:-4}"
if [ "$NPROCS_PER_TEST" -gt "$NPROC_TOTAL" ]; then
    NPROCS_PER_TEST="$NPROC_TOTAL"
fi
# ~9% headroom: 88 cores -> 4 ranks x 20 threads (80 cores busy).
THREADS_PER_RANK=$(( (NPROC_TOTAL / NPROCS_PER_TEST) * 91 / 100 ))
if [ "$THREADS_PER_RANK" -lt 1 ]; then
    THREADS_PER_RANK=1
fi
export OMP_NUM_THREADS="${TEST_THREADS:-$THREADS_PER_RANK}"
export CCTK_TESTSUITE_RUN_PROCESSORS="$NPROCS_PER_TEST"
echo "machine profile: ${NPROC_TOTAL} cores; each test instance = ${NPROCS_PER_TEST} ranks x ${OMP_NUM_THREADS} threads, one instance at a time"

cd "$CACTUS_HOME"
# RunTest.pl always exits 0, so the script itself must check the summary.
LOG="$(mktemp)"
gmake -s "${CONFIG}-testsuite" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
nf="$(grep -E 'Number failed' "$LOG" | tail -1 | grep -oE '[0-9]+' | tail -1 || true)"
rm -f "$LOG"
if [ -n "${nf:-}" ] && [ "$nf" -gt 0 ]; then
    echo "testsuite: ${nf} test(s) failed" >&2
    exit 1
fi
[ "$status" -eq 0 ] || exit "$status"
