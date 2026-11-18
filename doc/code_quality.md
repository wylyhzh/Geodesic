# Geodesic — static analysis (clanker / clang-tidy)

Run of 2026-11-13, against `src/` at the revision tagged `2026.11.09-1`
plus the fixes listed in §4 below.  This is the record for the "passes a
clanker review" item of the Einstein Toolkit thorn-submission checklist.

## 1. What was run

The official clanker driver (`Cactus/contrib/clanker/`) is not present in
this checkout and is not fetchable anonymously, so the check set below is
our own: clang-tidy with the Cactus-relevant bug families.  The config that
encodes it is [.clang-tidy](../.clang-tidy) in the thorn root; every check
it switches off is justified there and in §5 here.

    clang-tidy 18.1.3            (apt: clang-tidy clang-format bear clang libomp-dev)
    bear 3.x                     (captures the compile database)

## 2. How to reproduce

    sudo apt-get install -y clang-tidy clang-format bear clang libomp-dev
    cd /path/to/Cactus
    touch repos/einsteinanalysis/Geodesic/src/*.c
    bear --output compile_commands.json -- make -j24 sim
    cp repos/einsteinanalysis/Geodesic/.clang-tidy .        # see gotcha (c)
    clang-tidy -p . --quiet \
        configs/sim/build/Geodesic/Geodesic_integrate.c \
        configs/sim/build/Geodesic/Geodesic_init.c \
        configs/sim/build/Geodesic/ParamCheck.c

Three things that are not obvious:

(a) **Analyse the build copy, not `src/`.**  Cactus copies each thorn's
    sources into `configs/<cfg>/build/<Thorn>/` and compiles there, adding
    the generated bindings (`cctk_Functions.h`, `<Thorn>_Prototypes.h`) and
    the `-I` flags.  Running clang-tidy on `src/Geodesic_integrate.c`
    directly fails on missing headers; running it on
    `configs/sim/build/Geodesic/Geodesic_integrate.c` with the compile
    database from `bear` works.

(b) **`libomp-dev` is required.**  `Geodesic_integrate.c` includes `<omp.h>`;
    without that package clang's own `<omp.h>` is absent and the run dies
    with a `clang-diagnostic-error` before any check runs.

(c) **`.clang-tidy` is looked up next to the analysed file**, which is the
    build copy, so the thorn's own copy is not found.  Copy or symlink it to
    the Cactus root (clang-tidy walks up the directory tree).  Note that
    `clang-tidy` has no `-config=file:...` form — passing the path that way
    is parsed as inline YAML and fails with "not a mapping".

`bear` only records the files `make` actually recompiles, so after a
`touch` of the Geodesic sources the database holds exactly the three files
above — which is all this analysis needs.

## 3. Results

| run | check set | findings | of which reported as `error:` |
|---|---|---|---|
| before fixes | all of bugprone/cert/misc/performance/clang-analyzer on | 106 | 94 |
| after fixes | same | **83** | 71 |
| after fixes | shipped `.clang-tidy` | **50** | **0** (exit code 0) |

The third column counts diagnostics of every kind; the fourth counts those
promoted to `error:` by `WarningsAsErrors`, not compile failures — none of the
runs had a `clang-diagnostic-error` once `libomp-dev` was installed (§2b).

Categories fixed by the changes in §4 (before → after):

| check | before | after |
|---|---|---|
| `clang-analyzer-core.NullDereference` | 9 | 0 |
| `bugprone-implicit-widening-of-multiplication-result` | 4 | 0 |
| `bugprone-multi-level-implicit-pointer-conversion` | 6 | 0 |
| `bugprone-narrowing-conversions` | 2 | 0 |
| `bugprone-switch-missing-default-case` | 2 | 0 |

Categories that still fire with every check switched on, all reviewed and left
alone (§5).  The shipped `.clang-tidy` switches three of them off outright —
`bugprone-casting-through-void`, `misc-misplaced-const`,
`misc-unused-parameters` — which is why the shipped run reports 50 rather
than 83:

| check | count | why not acted on |
|---|---|---|
| `cert-err33-c` | 46 | unchecked stdio return values: `fprintf` 17, `fclose` 12, `snprintf` 7, `fflush` 5, `fseek` 2, the rest inside `if` conditions |
| `bugprone-casting-through-void` | 21 | MPI predefined handles |
| `misc-misplaced-const` | 10 | Cactus `CCTK_POINTER` typedefs |
| `cert-err34-c` | 4 | `sscanf`/`fscanf`/`atof` on our own files |
| `misc-unused-parameters` | 2 | parallel helper signatures |

No finding in any run was a real defect.  The three logs behind this table are
working notes and are not shipped with the thorn; all three runs are
reproducible with the commands in §2.

## 4. Code changes made as a result

All are behaviour-preserving; `make -j24 sim` is warning-free and the
testsuite is 11 passed / 0 failed after them (see §6).

1. `src/Geodesic.h`: added `#include <stdlib.h>` and a `GEODESIC_FATAL(msg)`
   macro = `CCTK_WARN (CCTK_WARN_ABORT, msg); abort ();`.  `CCTK_WARN` is a
   macro (`cctk_core.h:433`), so the analyser cannot see that
   `CCTK_WARN_ABORT` never returns and keeps exploring the "execution
   continues past the abort" path.  The trailing `abort ()` makes that
   explicit to both the compiler and the analyser.
2. `src/Geodesic_integrate.c`: the ten out-of-memory guards ("... allocation
   failed") now use `GEODESIC_FATAL`.  This is what cleared the nine
   `NullDereference` findings: they were all "if the malloc failed we
   continued and dereferenced NULL", i.e. the analyser walking past an abort.
3. Explicit casts where an integer index was multiplied before being used as
   a `size_t` offset or converted back to `CCTK_INT`:
   `MPI_Allgather(bb + (size_t)me * 6, ...)` (401), `c + (size_t)a*25 +
   (size_t)b*5` (1065), `outbuf[(size_t)f*4]` (2424, 2574),
   `(CCTK_INT) lrint (floor (...))` (1491, 5963).
4. `Geodesics_integrate()`: `Geodesic_gstate.multi_amr` is read ~14 times
   across function calls, so the analyser must assume it can change between
   the owner-map allocation and the chunk loop that uses `bd`, and reports
   `bd` as a possible null dereference.  It is now cached once in
   `const int multi_amr` after being set.
5. MultiPatch call sites (757, 758, 774, 775): the coordinate and Jacobian
   arrays are now passed as `(CCTK_POINTER_TO_CONST)localp` /
   `(CCTK_POINTER)xgp` etc.  `MultiPatch_*` takes `CCTK_POINTER` (i.e.
   `void *`) arguments, so the conversion was already intended — it is now
   written down, which also keeps
   `bugprone-multi-level-implicit-pointer-conversion` enabled.
6. `fourd_tw()` / `fourd_tw_a()`: added `default: return 0.0;` instead of a
   fall-through after the `switch`.

## 5. Triage of the findings that were left alone

- **`cert-err33-c` (46)** — return values of `fprintf`/`fclose`/`fflush`/
  `snprintf`/`fseek` are ignored.  This is the prevailing convention in
  Cactus thorns and in the surrounding EinsteinAnalysis code.  The check
  stays enabled as a warning but is not in `WarningsAsErrors`.
- **`bugprone-casting-through-void` (21)** — every one is an OpenMPI
  predefined handle: `MPI_COMM_WORLD` expands to
  `OMPI_PREDEFINED_GLOBAL(MPI_Comm, ompi_mpi_comm_world)`, which is a cast
  through `void *` inside OpenMPI's own header (`mpi.h:406`).  Nothing in
  this thorn can be changed to avoid it.
- **`misc-misplaced-const` (10)** — all in the hand-written `MultiPatch_*`
  prototypes at `Geodesic_integrate.c:84-104`.  They use
  `const CCTK_POINTER_TO_CONST`, which expands to `void *const` rather than
  the "obviously intended" `const void *`.  That is what the generated
  binding declares (`configs/sim/bindings/include/Geodesic_Prototypes.h`),
  and our declarations must match it exactly.
- **`cert-err34-c` (4)** — `sscanf` (4167), `fscanf` (4389, 4517), `atof`
  (5109) read the particle dump / sidecar files and one environment
  variable, all written or supplied by this thorn's own users.
- **`misc-unused-parameters` (2)** — `bg` is unused in `metric_at_point()`
  (2869) and `christoffel_at_point()` (2974).  The two are parallel
  per-particle evaluators called with the same argument list; keeping the
  signatures identical is worth more than the unused double.

## 6. Verification after the changes

    cd /path/to/Cactus && make -j24 sim          # "All done !", no Geodesic warnings
    cd repos/einsteinanalysis/Geodesic/test && ./run_tests.sh
        -> testsuite: 11 passed, 0 failed, 0 tolerance-only

## 7. Open question: formatting

The submission checklist asks for "correct formatting", which is ambiguous
here.  Running Cactus' own `.clang-format` (from `repos/flesh/.clang-format`)
over this thorn rewrites essentially the whole file:

| file | lines changed by `clang-format --dry-run` |
|---|---|
| `Geodesic_integrate.c` | 6968 of 6953 |
| `Geodesic_init.c` | 180 of 248 |
| `Geodesic.h` | 256 of 559 |
| `ParamCheck.c` | 13 of 68 |

We have **not** reformatted.  A whole-file reformat would destroy the blame
history of a 7000-line kernel and would not match the style of the other
EinsteinAnalysis thorns.  To be settled with the release chair / arrangement
maintainers before submission: does "correct formatting" mean Cactus'
conventions (which the code follows), or a clang-format-clean tree?
