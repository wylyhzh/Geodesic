#!/usr/bin/env bash
# =============================================================================
# Geodesic thorn testsuite launcher (official Einstein Toolkit mechanism).
#
# Usage:
#   ./run_tests.sh         the default set: the eleven tests registered in
#                          test/test.ccl (the same set that "-all" runs for
#                          this thorn; they are listed explicitly so that the
#                          script fails loudly if one of them disappears):
#
#                            Geodesic/circ_orbit
#                            Geodesic/circ_orbit_conserved
#                            Geodesic/circ_orbit_mpi
#                            Geodesic/circ_orbit_grid
#                            Geodesic/circ_orbit_grid_l27
#                            Geodesic/circ_orbit_grid_aei
#                            Geodesic/circ_orbit_grid_carpet
#                            Geodesic/circ_orbit_grid_2p
#                            Geodesic/circ_orbit_amr_small
#                            Geodesic/minkowski_line_carpetx
#                            Geodesic/bl_orbit_carpetx
#
#   ./run_tests.sh -all    every test found in any test.ccl of the
#                          arrangement, i.e. CCTK_TESTSUITE_RUN_TESTS is
#                          left unset.  For this thorn that is the same
#                          eleven tests, plus whatever other thorns in the
#                          arrangement register tests.
#
# The heavier runs (100-particle rings, 3-level AMR, the TwoPunctures BBH
# probes, the conv100 sampler comparisons) live in test/extra/ and are NOT
# part of the testsuite: RunTest.pl globs test/*.par only, and the review
# criteria are under 10 s per test / under 2 min for the whole component.
# test/extra/README explains how to run them as a local regression set.
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
# their job scheduler).  Tests run one at a time (RunTest.pl's default;
# CCTK_TESTSUITE_PARALLEL_TESTS is not set).  OMP_NUM_THREADS defaults to
# one per rank: it is the fastest tested profile on the development node,
# and the Brill-Lindquist CarpetX test has been verified at that setting.
# Leaving OMP_NUM_THREADS unset can instead make OpenMP take all cores per
# rank and oversubscribe the node.
#
# Environment overrides:
#   CACTUS_HOME          Cactus root (default: four levels above this
#                        script, i.e. the Cactus installation itself).
#   NPROC_TOTAL          total cores of the node (default: nproc).
#   NPROCS_PER_TEST      MPI ranks per test instance (default 4; tests
#                        with an explicit NPROCS in test.ccl keep it).
#   TEST_THREADS         OpenMP threads per rank (default: 1).
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

DEFAULT_TESTS="Geodesic/circ_orbit \
  Geodesic/circ_orbit_conserved \
  Geodesic/circ_orbit_mpi \
  Geodesic/circ_orbit_grid \
  Geodesic/circ_orbit_grid_l27 \
  Geodesic/circ_orbit_grid_aei \
  Geodesic/circ_orbit_grid_carpet \
  Geodesic/circ_orbit_grid_2p \
  Geodesic/circ_orbit_amr_small \
  Geodesic/minkowski_line_carpetx \
  Geodesic/bl_orbit_carpetx"

default_run=0
case "${1:-}" in
    ""|-default|--default)
        export CCTK_TESTSUITE_RUN_TESTS="$DEFAULT_TESTS"
        default_run=1
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
NPROC_TOTAL="${NPROC_TOTAL:-$(nproc 2>/dev/null || echo 8)}"
NPROCS_PER_TEST="${NPROCS_PER_TEST:-4}"
if [ "$NPROCS_PER_TEST" -gt "$NPROC_TOTAL" ]; then
    NPROCS_PER_TEST="$NPROC_TOTAL"
fi
export OMP_NUM_THREADS="${TEST_THREADS:-1}"
export CCTK_TESTSUITE_RUN_PROCESSORS="$NPROCS_PER_TEST"
echo "machine profile: ${NPROC_TOTAL} cores; each test instance = at most ${NPROCS_PER_TEST} ranks x ${OMP_NUM_THREADS} threads, one instance at a time"

cd "$CACTUS_HOME"
# RunTest.pl always exits 0, so the script itself must check the full summary.
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
set +e
gmake -s "${CONFIG}-testsuite" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
    echo "testsuite: gmake exited with status $status" >&2
    exit "$status"
fi

summary_value() {
    grep -E "$1" "$LOG" | tail -1 | grep -oE '[0-9]+' | tail -1 || true
}
available="$(summary_value 'Total available tests')"
unrunnable="$(summary_value 'Unrunnable tests')"
runnable="$(summary_value 'Runnable tests')"
passed="$(summary_value 'Number of tests passed')"
tolerance_only="$(summary_value 'set tolerance')"
failed="$(summary_value 'Number failed')"

for value in "$available" "$unrunnable" "$runnable" "$passed" "$tolerance_only" "$failed"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "testsuite: complete summary not found; treating the run as failed" >&2
        exit 1
    fi
done

if [ "$default_run" -eq 1 ]; then
    expected="$(wc -w <<< "$DEFAULT_TESTS")"
    if [ "$available" -ne "$expected" ] || [ "$runnable" -ne "$expected" ] || [ "$passed" -ne "$expected" ]; then
        echo "testsuite: expected $expected available/runnable/passed tests, got $available/$runnable/$passed" >&2
        exit 1
    fi
fi
if [ "$unrunnable" -ne 0 ] || [ "$failed" -ne 0 ] || [ "$tolerance_only" -ne 0 ] || [ "$passed" -ne "$runnable" ]; then
    echo "testsuite: runnable=$runnable passed=$passed failed=$failed unrunnable=$unrunnable tolerance-only=$tolerance_only" >&2
    exit 1
fi

echo "testsuite: $passed passed, $failed failed, $tolerance_only tolerance-only"
