#!/usr/bin/env bash
# =============================================================================
# Geodesic: rank-count agreement check (regression for the rank-face
# stencil bug reported by E. Schnetter, fixed by the halo exchange, the
# edge/corner halo extension and the FACE_EPS chunk-boundary fix).
#
# The SAME par file is run at 1, 2 and 4 MPI ranks (sequentially, one
# OpenMP thread per rank, so the comparison is not a thread-count
# artefact), and the particle dumps must agree to within TOL (default
# 1e-6, override with GEODESIC_MPI_IDENTITY_TOL).  Before those fixes the
# 4-rank local27 runs disagreed with the 1-rank run at the O(1) level
# (x off by 2.35 after 100 steps): edge/corner cells of the decomposition
# were never halo-filled, which silently switched the Christoffel
# assembly to the analytic branch, and a particle sitting exactly on a
# rank face skipped the chunk-boundary constraint and flew across the
# neighbouring subdomain before being dropped.  The assertion on |z| and
# |u_z| below catches the symmetry-breaking symptom directly.
#
# The criterion is a tolerance, not bit-identity: the FACE_EPS fix
# changes how the integration is cut into chunks, so a particle is handed
# to the neighbouring rank at a different point along its path and the
# floating-point summation order changes.  What remains is roundoff
# (measured max over all columns: 0 at 2 ranks, 1.1e-8 at 4 ranks).
#
# Usage:  ./check_mpi_identity.sh [parfile] [itlast]
#         (default: extra/conv100_local27.par, the 100^3 dx=0.5 Kerr-Schild
#          circular-orbit test whose z = 0 plane is a rank face for
#          the 2- and 4-rank decompositions)
#         itlast (default 100) truncates the run: rank-face crossings are
#          already exercised by then, so long integrations add runtime
#          without adding evidence.
# Exit 0 = agreement within TOL, 1 = mismatch.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACTUS_HOME="${CACTUS_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
EXE="$CACTUS_HOME/exe/cactus_sim"
PAR="${1:-$SCRIPT_DIR/extra/conv100_local27.par}"
ITLAST="${2:-100}"
BASE="$(mktemp -d /tmp/geodesic_mpi_identity.XXXXXX)"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found (build the 'sim' configuration)" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
rc=0
for n in 1 2 4; do
    d="$BASE/r$n"
    mkdir -p "$d"
    # Work from a copy of the par with a shortened cctk_itlast so the check
    # stays quick; the termination file is created only on real completion
    # (create_termination_file = "no" in the pars), so nothing has to be
    # staged here.
    sed "s/^Cactus::cctk_itlast[[:space:]]*=.*/Cactus::cctk_itlast      =  ${ITLAST}/" \
        "$PAR" > "$d/$(basename "$PAR")"
    ( cd "$d" && mpirun -np "$n" "$EXE" "$d/$(basename "$PAR")" > log 2>&1 )
    if [ $? -ne 0 ]; then
        echo "FAIL: run at $n rank(s) exited non-zero (see $d/log)" >&2
        rc=1
    fi
done
[ "$rc" -ne 0 ] && { rm -rf "$BASE"; exit 1; }

# locate the dump inside each run dir (IO::out_dir = $parfile)
declare -a dumps
for n in 1 2 4; do
    f="$(find "$BASE/r$n" -name geodesic_particles.txt | head -1)"
    if [ -z "$f" ]; then
        echo "FAIL: no geodesic_particles.txt in the $n-rank run" >&2
        rm -rf "$BASE"; exit 1
    fi
    dumps[$n]="$f"
done

# Largest absolute deviation per column between two dumps (header lines
# starting with '#' and the particle index are skipped), so a mismatch is
# reported as a magnitude rather than as a truncated textual diff.
maxcoldiff()
{
    awk '
        /^#/ { next }
        FNR == NR { row[++nrow] = $0; next }
        {
            q++
            if (q > nrow) { more = 1; exit }
            split(row[q], f, " ")
            if (NF != length(f)) { nfm = 1; next }
            for (i = 1; i <= NF; i++) {
                d = $i - f[i]; if (d < 0) d = -d
                if (d > m[i]) m[i] = d
                if (i > ncol) ncol = i
            }
        }
        END {
            if (more) print "  (second file has more data rows than the first)"
            if (nfm)  print "  (some rows have differing field counts)"
            for (i = 1; i <= ncol; i++) printf "  col %d: max|delta| = %.3e\n", i, m[i] + 0
        }
    ' "$1" "$2"
}

TOL="${GEODESIC_MPI_IDENTITY_TOL:-1e-6}"
rc=0
for n in 2 4; do
    worst="$(maxcoldiff "${dumps[1]}" "${dumps[$n]}" \
             | awk '/max\|delta\|/ { if ($NF + 0 > w) w = $NF + 0 }
                    END { printf "%.3e", w + 0 }')"
    printf 'max|delta| over all columns, 1 vs %d rank(s): %s\n' "$n" "$worst"
    if awk -v a="$worst" -v b="$TOL" 'BEGIN { exit !(a > b) }'; then
        rc=1
    fi
done
if [ "$rc" -ne 0 ]; then
    echo "FAIL: dumps differ beyond tolerance $TOL (dumps kept in $BASE)" >&2
    for n in 2 4; do
        echo "--- 1 vs $n rank(s) ---" >&2
        diff "${dumps[1]}" "${dumps[$n]}" | head -4 >&2
        echo "    columns: t x y z ut ux uy uz tau [cctk_time]" >&2
        maxcoldiff "${dumps[1]}" "${dumps[$n]}" >&2
    done
    exit 1
fi
echo "OK: 1-, 2- and 4-rank dumps agree within $TOL"

# equatorial-symmetry assertion: the conv100_local27 particle starts on
# the z = 0 equatorial orbit; an exact symmetry of the data keeps |z| and
# |u_z| at the roundoff level in EVERY rank configuration.  (Only run for
# the default par; other pars need not be equatorial.)
if [ "$(basename "$PAR")" = "conv100_local27.par" ]; then
    bad=0
    for n in 1 2 4; do
        # data columns: t x y z ut ux uy uz tau (dump rows every cctk_itlast
        # step); the '#'-prefixed header lines are skipped.
        vals="$(awk '!/^#/ && NF>=9 { z=($4<0?-$4:$4); uz=($8<0?-$8:$8);
                   if (z>1e-6 || uz>1e-6) print NR": z="$4" uz="$8 }' \
                   "${dumps[$n]}" | head -3)"
        if [ -n "$vals" ]; then
            echo "FAIL: $n-rank run left the equatorial plane:" >&2
            echo "$vals" >&2
            bad=1
        fi
    done
    [ "$bad" -ne 0 ] && { rm -rf "$BASE"; exit 1; }
    echo "OK: |z| and |u_z| stayed at the roundoff level at 1, 2 and 4 ranks"
fi

rm -rf "$BASE"
exit 0
