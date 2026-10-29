# Geodesic

A parallel test-particle geodesic tracker thorn for the
[Einstein Toolkit](https://einsteintoolkit.org) (Cactus).

- **Paper:** Yuhua Li, *Geodesic: a parallel test-particle geodesic tracker
  for the Einstein Toolkit* (2026, preprint draft — PDF included in
  [doc/geodesic_paper.pdf](doc/geodesic_paper.pdf), Chinese translation in
  [doc/geodesic_paper_zh.pdf](doc/geodesic_paper_zh.pdf); comments welcome.
  To be submitted to arXiv, gr-qc cross-listed with astro-ph.HE).
- **Licence:** GPLv2+ (see [COPYING](COPYING))
- **Maintainer:** Yuhua Li <liyhts@yeah.net>

## Purpose

Particles (test masses) follow the geodesic equation

    du^mu/dtau = -Gamma^mu_{ab} u^a u^b ,   dy^mu/dtau = u^mu

with no coupling to any fluid.  The 4-velocity is kept normalized
(u_mu u^mu = -1) throughout.  The thorn is intended for tracking large
numbers of tracer particles through numerical general-relativistic
evolutions (e.g. GRHydro/Carpet) or through the exact Kerr spacetime.

Two metric sources are supported:

- `Exact = no` (default): the metric and Christoffel symbols are
  sampled from the ADMBase grid (`gxx..gzz`, `alp`, `betax..betaz`).
  Four samplers are available (`Geodesic::metric_interp`): `l27` fits
  the simplest 27-point (3x3x3) Lagrange polynomial of the metric and
  of the nodal Christoffel symbols from the **current time level
  only**; `local27` (default) fits a self-contained 125-node (5x5x5)
  quartic tensor-product polynomial of the metric and of the nodal
  Christoffel symbols and blends the three stored time levels with a
  3-node Lagrange in the particle's coordinate time; `aei` evaluates
  the order-4 pointwise Lagrange interpolant of the `AEILocalInterp`
  thorn on the ten 3+1 fields and their first spatial derivatives at
  the particle position (the two stored time levels are linearly
  blended); `carpet` uses the same AEI order-4 spatial molecule as
  `aei` and takes the carpet-style 2nd-order 3-point time difference
  of the full 4-metric.  `l27` and `local27`
  are fully self-contained (no interpolator from another thorn);
  `aei` and `carpet` need the `AEILocalInterp` thorn, whose
  interpolator is resolved at startup.  No analytic form of the
  metric is assumed.  The connection includes the `dt(g)` terms of
  `Gamma^mu_{00}` and `Gamma^mu_{0 nu}`.  Every sampler works for
  time-dependent metrics, not only stationary ones.  A stationary
  metric is detected automatically (the oldest two levels agree to
  relative 1e-12 on a subsample), in which case `dt(g) = 0` exactly
  and the time difference is skipped.  Time-dependent runs need the
  ADMBase metric fields at `timelevels >= 3` (with
  `Carpet::init_fill_timelevels = "yes"`), and the particle step is
  then capped at the grid time step (the connection is frozen at the
  grid's current time within one particle step).
- `Exact = yes`: the metric and Christoffel symbols are the exact
  analytic Kerr--Schild solution of mass `M` and spin `a`
  (`Geodesic::M`, `Geodesic::a`; the formulas are built into the thorn),
  independent of the grid.  Useful as a reference/verification mode;
  no grid data are needed for the metric itself.

## Interpolation modes

There are two independent metric regimes, selected by `Geodesic::Exact`.
They use different inputs and ignore different parameters, so they are not
two settings of one interpolator.

### `Exact = "yes"` -- analytic Kerr--Schild (grid-independent)

- Metric and Christoffel symbols are the exact analytic Kerr--Schild
  solution of a **single** black hole (mass `M`, spin `a`; formulas built
  into the thorn).  No grid is sampled.
- `metric_interp` and all AMR/stencil machinery are **ignored**.  Only
  `Exact`, `M`, `a` matter for the metric.
- Time is handled by feeding the particle's coordinate time straight into the
  analytic formulas: there are no stored time levels, no blend, and no
  "missing past level" case.
- The metric is exact, so the only error is the **ODE integrator** itself.
  Grid-convergence questions (dx, order) do not apply.
- This is a single Kerr hole in Kerr--Schild coordinates -- not a numeric
  evolution and not a binary.  It is the reference/verification mode used by
  `circ_orbit` and `ellipse_orbit_amr`.

### `Exact = "no"` (default) -- grid: four samplers

Metric and Christoffels are sampled from the ADMBase grid by one of four
samplers (`Geodesic::metric_interp`):

| # | `metric_interp` | what it does |
|---|-----------------|--------------|
| 1 | l27 | The simplest sampler: 27-point (3x3x3) Lagrange fit of the metric and of the nodal 6th-order-FD Christoffel symbols, **current level only**.  Node-exact on the 27 stencil nodes.  A degenerate / non-finite staged fit drops the particle via the loss policy (no fallback interpolant). |
| 2 | local27 (default) | Self-contained 125-node (5x5x5) quartic tensor-product fit of the metric and of the nodal 6th-order-FD Christoffel symbols (precomputed separable inverse Vandermonde, the 125x125 `MtxA` analogue); the three stored time levels are blended with a 3-node Lagrange in the particle's coordinate time.  Node-exact on all 125 grid nodes.  Uses **no** interpolator from another thorn. |
| 3 | aei | Pointwise order-4 Lagrange interpolation via `AEILocalInterp` (10 ADMBase fields + first spatial derivatives at the particle position); the 4-metric and Christoffels are composed from the sampled fields.  The two stored time levels are linearly blended in the particle's coordinate time. |
| 4 | carpet | Pointwise sampler with the same `AEILocalInterp` order-4 spatial molecule as `aei`; the time derivative is the carpet-style 2nd-order 3-point difference `(3 L0 - 4 L1 + L2) / (2 dt)` of the full 4-metric. |

**Semantics to keep straight**

- **`l27` and `local27` are fully self-contained**: they need no
  interpolator from another thorn.  `aei` and `carpet` resolve the
  AEI interpolator handle at startup and abort if it is missing.
- **The stencil cache is allocated only for the sampler in use.**  Each
  grid-mode particle carries a per-particle cache whose size depends on
  `metric_interp`: `l27` allocates the 27-node cache only (~35 kB per
  particle), `local27` allocates the 125-node cache only (~241 kB per
  particle), and `aei` / `carpet` allocate no node cache at all (they
  interpolate pointwise).  For 10000 particles that is ~0.35 GB
  (`l27`), ~2.4 GB (`local27`) or ~0 GB (`aei` / `carpet`).
- **Choosing a sampler.**  For a stationary (or slowly evolving) metric
  `l27` is the cheapest option.  For a dynamical metric — a binary
  merger such as GW150914 — use `local27` (the default, self-contained)
  or `aei` / `carpet`; `l27` reads the current time level only, so its
  connection is frozen during a particle step.  If memory is the
  binding constraint, switch to `aei` or `carpet` (which require the
  `AEILocalInterp` thorn).
- **Time handling is fixed per sampler, not a user parameter.**  `l27`
  uses the current level only; `local27` blends the three stored levels
  (3-node Lagrange); `aei` blends the two stored levels (linear);
  `carpet` keeps the current-level metric and takes a 2nd-order 3-point
  time difference.
- **There is no robust / adaptive stabilization layer.**  A failed or
  degenerate fit (`l27`, `local27`) drops the particle via the loss
  policy; an out-of-domain point in `aei` / `carpet` fails the
  interpolant and is dropped the same way.
- **AMR is level/stencil handling, not a separate "no-time" algorithm.**
  A particle's stencil is drawn from the finest level that contains it
  in its safety region.  When the *past* level is not yet filled at an
  AMR cell (first integration, or a freshly-grown refinement region),
  the `local27` blend pins to the available levels (2-node blend of
  [t, t-dt], or level 0 only).  That fallback is driven by AMR cell
  state, not by a separate parameter.

## Features

- OpenMP-parallel integration over particles (GSL `rkf45`, adaptive).
  The per-particle computations are fully independent, so the thorn
  scales to `particle_n_total` = 10^4 particles (validated on a
  44-core node; see the paper).
- AMR support (up to 16 Carpet refinement levels, parameter
  `geodesic_amr_levels`, default 3): each particle's 27-point stencil
  is drawn from the finest level that contains the particle in its
  safety region, so particles inside a refined region are
  automatically sampled at the finer grid spacing.
- The first integration step checks the parfile 4-velocity `u^mu`
  against `g(u,u) = -1`: normalized input is used unchanged (the check
  is idempotent); otherwise `u^0` is re-solved from the metric for a
  future-directed timelike vector and the correction is logged.
- Particles leaving the safe region (or with a bad norm) are handled by
  a run-time selectable loss policy (`particle_loss_policy`):
  `drop` (default; state set to NaN, particle no longer tracked),
  `flag` (frozen at its last state), or `respawn` (legacy: random safe
  re-placement with zero spatial velocity).

## Required thorns

- `ADMBase` — required by the grid samplers `l27`, `local27`, `aei` and
  `carpet` (they fetch its fields by name at run time); for
  `Exact = yes` only the grid geometry (dimensions, spacing, origin,
  ghost zones) is used, the metric values are ignored.  Not needed for
  `metric_interp = carpetx`, which samples `ADMBaseX` instead (ADMBase
  and CarpetX cannot be combined: CarpetX asserts that every non-grid
  group is REAL and CONSTANT, while ADMBase declares INT SCALAR groups)
- `CarpetX` — only for `metric_interp = carpetx`
- `GSL` (Runge--Kutta integrator)
- `Carpet` (required in current Einstein Toolkit builds: it is the
  only thorn implementing Cactus group storage, which `particle_arrays`
  and `ADMBase` need)

The `KerrSchild` thorn is **not** required: the analytic
Kerr--Schild formulas used by `Exact = yes` are built in.  Activate
`KerrSchild` only if you want a Kerr--Schild *initial grid* for
`Exact = no` runs.

## Limitations

- AMR runs normally use one (centred) patch per refinement level.
  Off-centre patches (e.g. the GW150914 double-centre TwoPunctures
  layout) are handled by the patch-origin reconstruction and the
  alignment diagnostics (`amr_align_warn`); multi-rank AMR is handled
  by the per-rank ownership logic (the `multi_amr` mode below).
- The particle arrays are replicated on every MPI rank
  (`DISTRIB=CONSTANT`).  In multi-process grid runs (`multi_amr` /
  `multi_exact` mode) the domain is partitioned among the ranks and
  each rank integrates the particles it owns; the per-particle state
  is exchanged between the ranks once per Cactus step, and in
  `Exact = no` runs the grid fields read by the interpolation stencils
  are exchanged as a five-cell halo around each rank's box, so a
  particle sees the same stencil data before and after a rank
  hand-over.  Multi-rank runs are therefore bit-for-bit reproducible
  (verified against single-rank baselines in the conv test series, and
  by `test/check_mpi_identity.sh`, which compares 1-, 2- and 4-rank
  dumps byte for byte).  The testsuite's 2-rank test
  (`test/circ_orbit_mpi`) and the 4-rank `conv100` / TwoPunctures
  tests are part of the default test set (see Tests below).
  Ownership is decided with a claim margin of 4 cells around each rank's box;
  the environment variable `GEODESIC_OWNER_MARGIN` overrides it without a
  rebuild (keep it well below the ~7-8 cell gap between the safe region and
  the patch edge, otherwise a claimed particle's stencil reads fall outside
  the patch).  `GEODESIC_HALO_DEBUG`, set to anything, prints the per-rank
  halo geometry (`lsh`, bounds, ghost zones, box extents) to stderr before
  each exchange.
- With `metric_interp = carpetx` (CarpetX driver) the multi-process mode is
  different: CarpetX's interpolator is an MPI collective (it redistributes
  the interpolation points over the octree and exchanges the results with
  `MPI_Alltoall`/`MPI_Alltoallv`), so the interpolation calls of all ranks
  have to match one-for-one and an adaptive ODE step cannot be split
  between ranks — owner-computes is not available there.  Instead every
  rank integrates every particle from the identical replicated state
  (`multi_cx` mode); the interpolated value is computed once, on the rank
  owning the leaf, so all ranks stay bit-identical with no state exchange
  at all, and rank 0 writes the dump.  The price is that the integration
  and its communication are duplicated once per rank, and the particle
  loop runs serially in this mode (MPI collectives must not be issued
  concurrently from several OpenMP threads).
- The `Exact = no` mode reads the metric and its time derivative from
  the ADMBase grid: time-varying metrics need the ADMBase
  metric/lapse/shift fields at `timelevels >= 3`, and within one
  particle step the connection is frozen at the grid's current time,
  so the particle step is capped at the grid time step (see the
  `Exact = no` entry above).
- A par that uses ADMBase but leaves `Carpet::prolongation_order_time`
  at its default of `1` runs normally yet **cannot be restarted from a
  checkpoint**.  The ADMBase `metric`, `curv`, `lapse`, `shift`,
  `dtlapse` and `dtshift` groups declare `timelevels = 3`, while Carpet
  sizes its time-level bookkeeping as
  `maxtimelevels = prolongation_order_time + 1 = 2` whenever
  `Carpet::max_timelevels` is left at the automatic `-1`.  The mismatch
  is harmless through the whole evolution and only surfaces when
  `CarpetIOHDF5` re-synchronises every declared time level during
  recovery, where it aborts in `CarpetLib::th::get_time` with
  `Assertion 'tl >= 0 and tl < timelevels' failed`.  The fix is one line
  in the par: `Carpet::prolongation_order_time = 2` (equivalently
  `Carpet::max_timelevels = 3`).  This is neither Geodesic- nor
  AMR-specific -- multi-level Geodesic runs checkpoint and restart
  normally once that parameter covers the declared time levels -- but
  it does mean the testsuite parfiles, none of which set it, cannot be
  used to exercise the checkpoint/restart path.
- The `l27` back-end (legacy 27-point stencil, `metric_interp = l27`)
  does not support multi-process runs in which particles sit on or
  within a few cells of a rank face: the stencil centre is clipped
  away from the particle, the resulting extrapolated 27-point metric
  fit fails the validity check and the particle is dropped at the
  first step (the other back-ends are unaffected).  With the default
  2x2 (y,z) decomposition of a 4-rank unigrid run this affects every
  equatorial (z = 0) particle, which is why the testsuite's
  `conv100_l27` test runs single process.  Use `local27` (default),
  `aei` or `carpet` for multi-process `Exact = no` runs.

## Building

Add the thorn to your thorn list, then build with SimFactory.
For a local checkout, add a line such as

    Geodesic=/path/to/Geodesic

to the thorn list file; for this repository on GitHub,

    Geodesic=https://github.com/wylyhzh/Geodesic@main

Then build as usual, e.g.

    simfactory/bin/sim build --mdbkey make 'make -j8' --thornlist einsteintoolkit.th

## Usage

A complete, tested parameter file (Kerr--Schild metric `M = 2`,
`a = 0.5` on a 50^3 Cartesian grid, one particle on a prograde
equatorial circular orbit, `Exact = yes`) is provided in
[par/example.par](par/example.par).

The parameter files that produced every number in the paper are also
included, so all reported results can be reproduced directly:

| File | Paper section |
|---|---|
| [par/conserv.par](par/conserv.par) | conservation of E, Lz and the norm (Sec. 5.2) |
| [par/conv50.par](par/conv50.par), [par/conv100.par](par/conv100.par), [par/conv200.par](par/conv200.par), [par/convexact.par](par/convexact.par) | grid-convergence study (Sec. 5.3) |
| [par/scale.par](par/scale.par) | strong scaling with 10^4 particles (Sec. 6) |
| [par/interp_amr_carpet.par](par/interp_amr_carpet.par), [par/interp_amr_aei.par](par/interp_amr_aei.par), [par/interp_amr_local27.par](par/interp_amr_local27.par) | the same orbit as [par/conv100.par](par/conv100.par) under 3-level AMR, with the three `metric_interp` samplers, for a line-by-line comparison against the unigrid conv100 dump |

The Geodesic-specific part of the example file is:

    Geodesic::Exact                  = "yes"
    Geodesic::M                      = 2
    Geodesic::a                      = 0.5
    Geodesic::particle_n             = 1
    Geodesic::particle_n_total       = 1
    Geodesic::particle_x[0]          = 11.989
    Geodesic::particle_y[0]          = 0.0
    Geodesic::particle_z[0]          = 0.0
    Geodesic::particle_tv[0]         = 1.392812583392655
    Geodesic::particle_xv[0]         = 0.0
    Geodesic::particle_yv[0]         = 0.5600648870761632
    Geodesic::particle_zv[0]         = 0.0

`particle_tv/xv/yv/zv` are the **contravariant 4-velocity components**
`u^mu = (u^t, u^x, u^y, u^z)` in Kerr--Schild coordinates, *not* a
3-velocity; the thorn verifies `g(u,u) = -1` and uses normalized input
unchanged.  For `Exact = no`, run the thorn inside any GR evolution
that provides the ADMBase grid fields.

## Main parameters

| Parameter | Meaning |
|---|---|
| `particle_n` | number of fixed particles (initial data from the parfile, max 1000); 0 disables |
| `particle_n_total` | total number of particles (max 10000); the remainder beyond `particle_n` are randomly placed with zero velocity |
| `particle_x/y/z[0..999]` | initial positions of the first `particle_n` particles |
| `particle_tv/xv/yv/zv[0..999]` | initial 4-velocity `u^mu` (see above) |
| `step` | coordinate-time advance per Cactus iteration; 0 uses `cctk_delta_time` |
| `max_iter_cap` | max ODE tau-steps per particle per Cactus iteration (default 2000) |
| `Exact` | metric source: `no` = grid, `yes` = analytic Kerr--Schild |
| `metric_interp` | grid-mode sampler: `l27` (simplest 27-point Lagrange fit, current level only, particle dropped on fit failure), `local27` (default; 125-node 5x5x5 quartic tensor-product fit, self-contained, no AEI), `aei` (AEILocalInterp pointwise order-4, 2-level linear time blend), `carpet` (AEI order-4 spatial molecule + carpet-style time difference) |
| `geodesic_amr_levels` | number of Carpet refinement levels used for the stencil (1-16, default 3); must equal `Carpet::refinement_levels` for an AMR run |
| `M` / `a` | Kerr mass and spin used by `Exact = yes` (built-in analytic formulas) |
| `initial_radius` / `excised_radius` | `initial_radius`: safe shell for the `respawn` policy (re-placement happens outside it).  `excised_radius` is a live boundary during integration: a particle whose Euclidean radius drops below it is marked out of bounds and handled by `particle_loss_policy` (dropped under `drop`); it is also the inner rejection radius for random (re)placement.  Set it to 0 for a binary, where the sphere has no meaning and can drop particles that merely pass close to a puncture. |
| `particle_loss_policy` | what to do with lost particles: `drop` (default), `flag`, `respawn` |
| `particle_rand_seed`, `particle_rand_*min/max`, `particle_middle_sp` | random (re)placement controls (respawn policy) |
| `particle_dump_every` | dump full particle state to `particle_dump_file` every k iterations (0 = off; rank 0 only) |
| `particle_dump_file` | dump file name, relative to the working directory (default `geodesic_particles.txt`; parent directories are created as needed) |
| `particle_dump_mode` | how the dump file is opened at the first write: `auto` (default) appends only when this process really recovered a checkpoint *and* a checkpointing IO thorn (`CarpetIOHDF5`/`CarpetIOF5`) is active, otherwise it starts the file clean; `truncate` / `append` force either way |
| `gverbose` / `rverbose` | diagnostics |
| `probe_file` / `probe_stride` / `probe_max_rows` | interpolation-error probe (see "Interpolation-error probe"): sidecar file name (empty = off, the default), evaluate once every N RHS evaluations, row cap (0 = unlimited) |

## Output

The evolved particle state (`tau`, `t`, `x, y, z`, `u^t..u^z`) is kept
in the `particle_*` arrays declared in `interface.ccl`; save them with
the Einstein Toolkit's standard output thorns (HDF5, Checkpoint, ...)
like any other grid function.  With `particle_dump_every = k > 0` the
thorn additionally writes a plain-text dump (`t, x, y, z, u^t..u^z,
tau, cctk_time` per particle) to the file given by `particle_dump_file`
(default `geodesic_particles.txt`, relative to the working directory;
parent directories are created as needed), starting with the initial
state and then one block of rows every k iterations.  Files written by
revision `2026.10.21-1` and earlier have nine columns (no `cctk_time`);
readers should accept both.

Restart: when a checkpointing IO thorn is active, the thorn records the dump
file offset at every checkpoint in `<particle_dump_file>.ckpt` and, on
recovery, cuts the dump back to the frame the recovered state belongs to, so
frames the crashed run wrote after that checkpoint are discarded instead of
staying as a stale tail, and the block the restarted run re-emits for the
recovered state is skipped instead of duplicated.  If the sidecar has no entry
for the checkpoint being restarted from, the file's own `t` column decides the
cut; when the two disagree the thorn says so and uses the sidecar.

`t` is the particle's own coordinate time, integrated as the ODE state
`y[0]` (`dy[0]/dtau = u^t`), not the grid time.  It is anchored to
`cctk_time` once, at the release step, and from then on advanced by the
finest-level `dt`, so it only *tracks* the grid: a particle that runs out
of tau-steps before the step end falls short, and because the ensemble
target is the smallest live particle time, that one particle drags every
other particle's clock with it.  The `cctk_time` column is the grid time
of the dump, so the accumulated lag is readable as `cctk_time - t`
instead of having to assume the two agree.  Comparing two runs at equal
dump *index* therefore compares them at slightly different coordinate
times; the resulting along-track offset is `|v| * delta_t`, which for the
r=8 ring runs reaches 1e-2 -- larger than the sampler-to-sampler
trajectory difference itself.  Interpolate each worldline onto a common
`t` before differencing.

### Revision stamp

The thorn carries its own version in `GEODESIC_REVISION` (`Geodesic.h`,
format `YYYY.MM.DD-<counter>`); bump it whenever behaviour changes, since
the Cactus build id does not say which Geodesic sources produced a run.
It is reported by `Geodesic_ParamCheck` and by `Geodesics_init` (together
with the parameter values that actually took effect) and is written as a
`#` comment at the top of every particle dump, so a dump file can always
be traced back to the code and parameter set that produced it.

## Tests

The thorn ships a Cactus testsuite in `test/` with 19 tests, driven by
the official `RunTest.pl` infrastructure.  The convenient launcher is
`test/run_tests.sh`:

- `./run_tests.sh` (no argument) runs the 9-test default set, which
  covers all four metric-sampling back-ends and both metric families
  (stationary Kerr grid and the two-puncture frozen-initial-data
  background; see the paper for the ML_BSSN-evolved merger runs):
  `conv100`, `conv100_l27`, `conv100_local27`, `conv100_local27_1p`,
  `conv100_aei`, `conv100_carpet`, `circ_orbit`, `tp_local27_025`,
  `tp_far_local27_025_t80`.
- `./run_tests.sh -all` runs all 19 tests in `test/test.ccl`.

Following the Einstein Toolkit convention that the user declares the
local machine, `run_tests.sh` auto-detects the core count (`nproc`)
and launches each test instance one at a time as 4 ranks with
~91% of the per-rank core share as OpenMP threads (e.g. 4p×20t on an
88-core node, ~9% headroom).  It pins `OMP_NUM_THREADS` to the value
the reference data were generated with (20 on this node): leaving it
unset lets OpenMP take all cores per rank, a 4x oversubscription that
slows the TwoPunctures t = 0 puncture solve by an order of magnitude.
Override with `NPROC_TOTAL`, `NPROCS_PER_TEST`, `TEST_THREADS`.

Equivalently, from the Cactus root:
`PROMPT=no CCTK_TESTSUITE_RUN_TESTS="Geodesic/..." gmake sim-testsuite`
(an empty `CCTK_TESTSUITE_RUN_TESTS` runs everything).  Reference data
for every test live in `test/<testname>/geodesic_particles.txt`, the
only file compared; run directories of instrumented builds may also
contain `formaline-*` / timer artifacts, which the comparator ignores.
Seventeen of the nineteen references predate the `cctk_time` column and
still have nine columns (`tp_local27_025` and `tp_far_local27_025_t80`
have ten); the comparator (`RunTestUtils.pl`) compares only the leading
columns the two files share, so both forms pass -- but with a nine-column
reference the `cctk_time` column is not checked at all.  `./run_tests.sh
-all` regenerates the whole set in one go (~30-45 min on an 88-core node)
and would make them uniform.

- `circ_orbit` integrates a prograde equatorial circular orbit
  (r = 11.989, M = 2, a = 0.5, `Exact = yes`) and compares the full
  particle state dump against reference data row by row.
- `circ_orbit_conserved` runs the same dump through
  `util/check_conserved`, which prints the Kerr constants of motion
  (E, L_z) for every row, and compares them against the reference.
  On a correct run both are constant to about nine digits over the
  whole orbit; a broken integrator shows up as a drift of E / L_z.
- `circ_orbit_mpi` is the `circ_orbit` run with 2 MPI ranks (requires
  `mpirun`); the integration runs on a single rank, so the dump must be
  bit-identical to the single-rank reference.  The default set also
  includes 4-rank runs (`conv100`, `conv100_local27`, `conv100_aei`,
  `conv100_carpet`, `tp_local27_025`, `tp_far_local27_025_t80`),
  which exercise the same owner-computes multi-rank path.
- `circ_orbit_grid` is the same orbit through the `Exact = no`
  interpolation path: the metric is the Kerr--Schild initial data
  stored on the ADMBase grid, the past time levels are filled with the
  current (static) values so the thorn's stationary-metric detection
  engages and `dt(g) = 0` exactly.  The dump is compared against
  reference data row by row, as for `circ_orbit`; E and L_z stay
  stable to about four digits over the orbit, which is the expected
  grid-metric interpolation error of this mode.
- `circ_orbit_amr` is a 100-particle ring on the same circular
  orbits, `Exact = no`, with AMR: a 50^3 base grid (dx = 1) plus two
  static Carpet refinement boxes centred on the origin (half-widths
  16 and 8, factor 2).  The orbits stay at r in [11.7, 12.1], so
  every particle lives on the level-1 grid (dx = 0.5) for the whole
  run and the level-2 box (dx = 0.25) is never sampled; each
  particle's 27-point stencil is drawn from the finest level that
  contains it.  This validates the per-level metric acquisition, the
  stationary-metric detection on refined levels, and the level
  selection; the dump is compared against reference data row by
  row.  The test's par file is configured for a single process.
- `circ_orbit_100` is the control for `circ_orbit_amr`: the same
  100 particles on the same 50^3 grid with no refinement, so the
  stencil is always sampled on the base grid.  The E / L_z scatter
  of `circ_orbit_amr` (via `util/check_conserved`) is about 2-3x
  smaller than this run's, consistent with the 2nd-order 27-point
  metric interpolation.
- `ellipse_orbit_amr` is a 200-particle sample of one bound
  equatorial *elliptic* geodesic (Kerr M = 2, a = 0.5, Kerr-Schild
  coordinates): pericentre r = 7.5, apocentre (64, 0, 0),
  E = 0.9742480384407799, L_z = 7.1053558433047188, particle i at
  azimuth 2 pi i / 200 of the first winding of the strongly
  precessing orbit (~4.02 pi per radial period).  It runs on a
  144^3 base grid (domain +-72, dx = 1) with three static spherical
  Carpet refinement boxes (half-widths 36/18/9, factor 2), so the
  particles occupy all four levels at t = 0 and cross level
  boundaries in both directions.  With `Geodesic::step = 100` and
  256 Cactus steps (no subcycling) the particle coordinate time
  advances to 25600, about 19 radial periods.  It uses
  `Exact = yes` deliberately: on the 2nd-order grid metric
  (`Exact = no`) the init-time u^t re-solve plus coherent
  integration error push the near-pericentre particles across the
  plunging boundary into the excised sphere, where they are
  dropped; with the built-in analytic metric no particle is dropped
  and `util/check_conserved` gives max drifts of 2.4e-6 (E) and
  7.1e-5 (L_z) relative to the initial values over the whole run.
  The dump is compared against reference data row by row.
- `conv100` is the paper's grid-convergence orbit (the P0 circular
  orbit, r = 10, M = 2, a = 0.5) on a 100^3 grid (dx = 0.5),
  `Exact = no`, 4 MPI ranks -- shortened for the testsuite to 200 steps
  (t = 20.0); the paper run itself, [par/conv100.par](par/conv100.par),
  is 1500 steps (t = 150.1, ~1 orbital period).  The single-particle
  dump is compared against reference data row by row.
- `conv100_l27`, `conv100_local27`, `conv100_aei`, `conv100_carpet`
  are the same orbit on the same grid, one per metric-sampling
  back-end (`metric_interp = l27` / `local27` / `aei` / `carpet`; `aei`
  and `carpet` sample through the `AEILocalInterp` / `LocalInterp`
  interpolation interface), at 200 steps (t = 20.0) except `l27`, which
  runs 100 steps (t = 10.0).  Each dump must reproduce its reference to
  RELTOL 1e-12.
  `conv100_l27` runs single process: the equatorial ring sits on rank
  faces in the default 2x2 (y,z) multi-rank decomposition, where the
  `l27` stencil is clipped out of the particle (see Limitations).
- `conv100_local27_1p` is the single-process twin of `conv100_local27`
  compared against the four-process reference dump: a PASS is a
  1-vs-4-rank bit-identity check of the `local27` back-end (see
  `test/check_mpi_identity.sh` for the 1/2/4-rank version at three
  rank counts in one run).

On that 200-step orbit, single process, the maximum position error
against the `Exact = yes` trajectory of the same orbit is:

| `metric_interp` | t = 2 | t = 10 | t = 20 |
|---|---|---|---|
| `l27`     | 3.74e-6 | 4.42e-5 | 9.97e-5 |
| `aei`     | 3.68e-7 | 6.29e-6 | 2.97e-5 |
| `local27` | 1.60e-7 | 2.56e-7 | 1.97e-7 |

(100^3 grid, dx = 0.5, dt = 0.1, particle started at t = 0.)
- `tp_local27_025`, `tp_aei_025`, `tp_carpet_025` are NEAR-field
  probes on the two-puncture `GW150914` background (TwoPunctures
  initial data, 144^3 grid, dx = 0.25, no AMR, excised radius 2):
  a 100-particle ring at r = 8 with strong inward velocity, 20 steps.
  The strong-field plunge produces deterministic NaN drops
  (`loss_policy = drop`); the NaN-aware comparator skips rows that are
  NaN in both files but fails if the drop pattern (which particles,
  at which steps) changes.
- `tp_far_local27_025_t80`, `tp_far_aei_025_t80`,
  `tp_far_carpet_025_t80` are FAR-field probes on the same background:
  a 100-particle ring at r = 10 with modest inward velocity, 80
  steps.  All 100 particles survive with zero NaN on all three
  back-ends; these are well-posedness regression tests on a strongly
  curved two-puncture background.

`util/check_conserved` is a small wrapper script: on first use it
compiles `util/check_conserved.c` (plain C, libc + libm) into
`util/check_conserved.bin` and reuses the binary afterwards, so no
precompiled binary is stored in the repository.

## Plotting

`util/plot_geodesic.py` plots particle trajectories from a plain-text
dump (`geodesic_particles.txt`).  It needs only numpy and matplotlib.
The number of particles per dump block is inferred from the data
(override with `--npar`); the block time is the per-block median of the
`t` column, which stays correct in the presence of frozen or dropped
particles (their rows carry an earlier drop time).

    python3 util/plot_geodesic.py geodesic_particles.txt fig_orbits --2d
    python3 util/plot_geodesic.py mydump.txt figs/foo --both --plane xz --pdf

Options: `--2d` / `--3d` / `--both` (default `--2d`), `--plane xy|xz|yz`
(default `xy`), `--npar N`, `--dpi N` (default 150), `--pdf` (also write a
PDF of each figure), `--no-arrows`.  Dropped particles are marked with an
X at their last alive position.  If no `DATAFILE` is given, the first
`geodesic_particles*.txt` in the current directory is used.

## Notes

- `derivative.h` is the Kranc-generated header for the 27-point
  Lagrange interpolation; `make.code.defn` is its Kranc definition,
  kept for reproducibility.
- This thorn is a pure geodesic integrator: it does not evolve the
  metric and does not require a hydrodynamics thorn.

## Choosing dx and dt on a time-dependent background

Scope: these numbers are for a particle distribution that spans a large
fraction of the grid -- the r = 8 ring inside a +-10.5 domain.  If the
particles are instead confined to a small sub-volume, AMR can be the cheaper
option; see item 4 and the (1 + 16 f) table.

Measured on the standard r = 8 equatorial ring (100 particles released at
t = 1.60, dumped every 0.05 in coordinate time, 4 MPI ranks x 21 threads,
TwoPunctures b = 5 background, integrated to t = 14.05 unless the
background died first).  "lag" is `cctk_time - t` -- the grid time minus
each particle's own integrated coordinate time (column 10 minus column 1
of the dump) -- so it is exactly the clock skew that corrupts frame-by-frame
comparisons between runs.  For the pre-2026.10.28 dumps without
`cctk_time`, lag is measured against the nominal frame time
(t_first + 0.05 * frame index), which is exact because `particle_dump_every`
is fixed.

| grid | dx | dt (particle) | point-steps | wall (s) | point-steps/s | median lag at 14.05 | alive at 14.05 | background |
|---|---|---|---|---|---|---|---|---|
| AMR 1/8 base + 1/16 box r=8, local27 | 0.125 / 0.0625 | 0.025 / 0.0125 | 2.18e10 | 51,818 | 4.2e5 | +0.013 | 72/100 | stable |
| AMR 1/8 base + 1/16 box r=8, aei | 0.125 / 0.0625 | 0.025 / 0.0125 | 2.18e10 | 54,782 | 4.0e5 | +0.033 | 76/100 | stable |
| uniform 1.8/16 (187^3), local27 | 0.1123 | 0.00625 | 1.47e10 | 28,545 | 5.2e5 | 0.000 | 74/100 | stable |
| uniform 1.8/16 (187^3), aei | 0.1123 | 0.00625 | 1.47e10 | 25,014 | 5.9e5 | 0.000 | 81/100 | stable |
| uniform 1.5/16 (225^3), local27 | 0.09375 | 0.00625 | 2.56e10 | 47,340 | 5.4e5 | 0.000 | 74/100 | stable |
| uniform 1.5/16 (225^3), aei | 0.09375 | 0.00625 | 2.56e10 | 44,673 | 5.7e5 | 0.000 | 78/100 | stable |

Footnote: the AMR and 1.5/16 runs still use `excised_radius = 2.0` (the
uniform 1.8/16 run uses 0.0, the correct value for a binary).  That sphere is a
live drop boundary -- a particle inside it is marked out of bounds -- but it never
fired here: the smallest radius from the origin reached by any particle in those
four runs is 2.33-2.38, and lost particles never get closer than 4.24 because they
fall into the punctures (at x = +4.46 and -5.54 for this b = 5 setup, coordinate
separation 10), not toward the origin.  So the survival counts are comparable.

The dt ladder at fixed dx (uniform grids; lag taken at the last frame with
>= 50% of the particles alive):

| dx | dt | point-steps | median lag | outcome |
|---|---|---|---|---|
| 0.25 (74^3) | 0.05 | 9.4e7 | +0.155 at t=9.65 | ring lost before 11.60 |
| 0.125 (148^3) | 0.025 | 1.50e9 | +0.049 at t=10.85 | ring lost before 11.60 |
| 0.125 (168^3) | 0.0125 | 5.3e9 | 0.000 | background blew up at t=10.85 (shift ~1e246) |
| 0.125 (168^3) | 0.00625 | 8.1e9 | 0.000 | background blew up at t=10.65 (shift ~1e18) |
| 0.1123 (187^3) | 0.00625 | 1.47e10 | 0.000 | stable to 14.05 |
| 0.09375 (225^3) | 0.00625 | 2.56e10 | 0.000 | stable to 14.05 |

What this says:

1. **The clock lag is a dt effect, not a sampler effect.** At fixed dx the
   lag falls roughly in proportion to dt (0.155 at dt=0.05, 0.049 at 0.025,
   0 at 0.0125 and below on these runs).  The sampler matters only
   secondarily: at 1.5/16 the local27-vs-aei position difference at t=14.05 is
   3.2e-4 (median over particles), 8.6e-4 at the 90th percentile and 8.2e-3 at
   the worst particle, while 1.8/16-vs-1.5/16 is 1.8e-3, 1.8e-2 and 5.1e-1 --
   5.7x larger at the median, 21x at p90, i.e. dx dominates.  (A coarse grid
   also feeds back into dt indirectly: larger interpolation error pushes RKF45
   to smaller internal steps, and the `max_iter_cap` tau loop then truncates
   the step, which is what drags the whole particle clock.)
2. **dt is the cheap knob.** Halving dt doubles the step count -- cost is
   linear.  Halving dx multiplies the point-steps by 8 for the same volume.
   So once the background is stable, refine dt, not dx.
3. **dt cannot rescue a grid whose background is gauge-unstable.** Uniform
   1/8 (dx = 0.125) develops a local gauge instability at the punctures at
   t ~ 10.6-10.9 with shift jumping from ~0.1 to 1e18-1e246 in one step,
   for *both* dt = 0.0125 and dt = 0.00625, with no NaN anywhere.  dx has to
   be fine enough for the *background*; here that threshold sits between
   dx = 0.125 (fails) and dx = 0.1123 (works).

   "Jumping in one step" is meant literally.  PunctureTracker prints the
   interpolated shift at each puncture every step, and the sequence is smooth
   right up to the failure: uniform 1/8, dt = 0.0125 gives
   (-0.0908, -0.0319, -5.2e-4) at t = 10.8625 and (-4.7e245, 1.1e246,
   1.6e246) one step (0.0125) later at t = 10.875.  No ramp, no precursor.
   The value is finite -- so NaNChecker never fires -- but it is already 246
   orders of magnitude too large, and the failure time moves by only ~0.2M
   when dt is halved (uniform 1/8, dt = 0.00625 dies at t = 10.65 with shift
   ~1e18, after which PunctureTracker aborts Carpet).  The signature -- one
   step, no precursor, puncture-local, dt-independent, dx-thresholded -- is a
   grid-scale mode of the *spatial* discretization growing at the puncture,
   not a physical divergence and not a time-step effect.
4. **An AMR box around the particles pays off only if the particles occupy
   a small part of the domain.**  One extra level costs 16x per unit volume
   (8x the points, 2x the subcycled steps), so "uniform 1/8 base + 1/16 box"
   costs (1 + 16 f) times plain uniform 1/8, where f is the box volume over
   the domain volume (domain here is the cube +-10.5):

   | box half-width | f | cost vs uniform 1/8 | saving vs refining everywhere |
   |---|---|---|---|
   | 4 | 0.055 | 1.9x | 8.9x |
   | 6 | 0.187 | 4.0x | 4.3x |
   | 8 | 0.442 | 8.1x | 2.1x |
   | 9 | 0.630 | 11.1x | 1.5x |
   | 10 | 0.864 | 14.8x | 1.1x |
   | refine everywhere | 1 | 17x | 1x |

   The r = 8 ring forces the box to cover f = 0.44, so the box already costs
   8.1x -- less than half of refining everywhere and therefore worth almost
   nothing once the fixed AMR overheads are counted: AMR ran 22-31% worse per
   point-step (4.2e5 / 4.0e5 vs 5.4e5 / 5.7e5), spent 12-23% of wall in
   Carpet syncs, and left a residual lag (0.013-0.033) that the uniform runs
   do not.  Those overheads are roughly independent of f, so they are fatal at
   f = 0.44 and negligible for a population confined to r <= 4 (f = 0.055,
   1.9x), where AMR is the right choice.  One more caveat specific to this
   test: the resolution requirement that actually bit us is a *background*
   property (the local gauge behaviour at the punctures), so a box placed only
   around the particles would not remove it -- it would have to cover the
   punctures as well.

### Comparing two runs: error metrics, not survival counts

`conv/ring_table.py` (in the doc repo) turns the dumps into the two tables below,
and enforces two rules.

* **Survival is not a quality metric.**  A particle is lost here by running into
  one of the two punctures: on the uniform 1.8/16 local27 run, at its last frame
  with coordinates every one of the 26 lost particles is 0.15-0.36 from a
  puncture (median 0.28; the other runs give 0.08-0.50), while survivors get as
  close as 0.24-0.26.  Whether a scattered particle hits one is a knife-edge
  outcome -- the capture *time* differs by 0.45
  (median over the 22 particles captured in both) between local27 and aei on the
  *same* 1.5/16 grid, which is as large a change as a whole resolution tier
  produces (0.25 for 1.8/16 against 1.5/16, both local27).
* **Coordinates and u_mu are not comparable once a particle is inside the hole.**
  Each particle is compared only over times where it is outside the capture region
  in both runs (interpolated r > 3.0 from the origin and > 1.0 from a puncture;
  puncture positions come from one run's PunctureTracker log, interpolated and held
  fixed across the whole table) and its clock is not behind.  Capture itself is
  compared as a scalar, Delta t_drop.

Reference: uniform 1.5/16 local27 (`tp_ring_u1516_100p_x250`).  The first table is
the `all` set, i.e. all 100 particles of the IC block; the second repeats it with
the two outlier particles removed (`--exclude 53,54`, 98 particles).

| run (all set) | median dx at t=6 / 10 / 14 | rms_med | rms_max | dtau_med | median dt_drop | n_cap | alive |
|---|---|---|---|---|---|---|---|
| AMR 1/8 base + 1/16 box, local27 | 0.0041 / 0.0041 / 0.0050 | 0.0041 | 0.077 | 0.0052 | 0.30 | 26 | 72 |
| AMR 1/8 base + 1/16 box, aei | 0.0040 / 0.0037 / 0.0043 | 0.0039 | 0.075 | 0.0047 | 0.20 | 24 | 76 |
| uniform 1.8/16, local27 | 0.0000 / 0.0005 / 0.0018 | 0.0009 | 0.139 | 0.0008 | 0.25 | 25 | 74 |
| uniform 1.8/16, aei | 0.0001 / 0.0008 / 0.0018 | 0.0009 | 0.139 | 0.0008 | 1.25 | 19 | 81 |
| uniform 1.5/16, aei | 0.0000 / 0.0002 / 0.0003 | 0.0001 | 0.0029 | 0.0002 | 0.45 | 22 | 78 |
| uniform 1.5/16, local27 (reference) | 0 / 0 / 0 | 0 | 0 | 0 | 0 | 0 | 74 |

What the columns mean.  Every quantity is a difference between two runs at the same
*coordinate time*, in geometric units.  A particle's own coordinate time (dump
column 0) only tracks the grid clock, so the two worldlines of a particle are
linearly interpolated onto a common 0.05 coordinate-time grid (the dump interval)
before they are differenced; the trusted test is applied to the interpolated
positions.  A particle contributes only over *trusted* times: finite interpolated
coordinates in both runs, r > 3.0 from the origin, > 1.0 from either puncture.
Frames whose particle clock is more than 0.5 behind the grid are cut from the
worldline.  `--by-frame` falls back to the older comparison at matched dump index.

* **median dx at t=6/10/14** -- median over particles of the two runs' position
  separation at that exact coordinate time (both worldlines interpolated to it).  A
  snapshot; it hides the tail.
* **rms_med / rms_max** -- per particle, the rms of that separation over its
  trusted times (a time average, not an endpoint value); then the median, and the
  worst particle, over the set.  rms_med is the robust tier indicator, rms_max a
  tail indicator that one or two particles can dominate (see below).
* **dtau_med** -- median |Delta tau| at each particle's last trusted time.  Proper
  time accumulates along the trajectory and is gauge-independent, so this is an
  independent check on the position columns.
* **median dt_drop** -- median |Delta t_drop| over the particles captured in *both*
  runs, where t_drop is the frame time at which a particle first goes NaN (runs into
  a puncture).  Quantised to the 0.05 dump interval, so differences below 0.05 are
  not resolved.
* **n_cap** -- how many particles that subset contains.  Quote it next to dt_drop:
  it changes with the reference (19-26 here).
* **alive** -- diagnostic only, never a ranking criterion (first rule above).  It
  counts the run's own 100-particle block, so it does not change when particles
  are excluded from the statistics (the second table has n = 98 but the same
  alive counts).

The reference row is zero by construction (a run differs from itself by nothing);
it is listed so the six runs are all visible in one place.  Its own diagnostics are
74/100 alive, zero clock lag, 47,340 s wall.

Same six runs with the two worst particles removed (`--exclude 53,54`; #53 and #54
are the top two by rms in every pair, see the findings below):

| run (no #53, no #54) | median dx at t=6 / 10 / 14 | rms_med | rms_max | dtau_med | median dt_drop | n_cap | alive |
|---|---|---|---|---|---|---|---|
| AMR 1/8 base + 1/16 box, local27 | 0.0041 / 0.0041 / 0.0050 | 0.0040 | 0.0109 | 0.0052 | 0.30 | 26 | 72 |
| AMR 1/8 base + 1/16 box, aei | 0.0040 / 0.0037 / 0.0043 | 0.0039 | 0.0099 | 0.0046 | 0.20 | 24 | 76 |
| uniform 1.8/16, local27 | 0.0000 / 0.0005 / 0.0018 | 0.0009 | 0.0207 | 0.0008 | 0.25 | 25 | 74 |
| uniform 1.8/16, aei | 0.0001 / 0.0007 / 0.0018 | 0.0009 | 0.0207 | 0.0008 | 1.25 | 19 | 81 |
| uniform 1.5/16, aei | 0.0000 / 0.0002 / 0.0003 | 0.0001 | 0.0010 | 0.0002 | 0.45 | 22 | 78 |
| uniform 1.5/16, local27 (reference) | 0 / 0 / 0 | 0 | 0 | 0 | 0 | 0 | 74 |

Only rms_max moves by a large factor -- 6.7x at 1.8/16, 7.1x and 7.5x at AMR, 2.9x
at 1.5/16 -- while every median is unchanged to the fourth decimal (the three
exceptions, AMR local27 rms_med 0.0041 -> 0.0040, AMR aei dtau_med 0.0047 -> 0.0046
and uniform 1.8/16 aei dx at 10.05 0.0008 -> 0.0007, are of that size).  That is the
point of the exercise: the tier ordering is carried by rms_med, and rms_max is two
particles out of 100.

**What the interpolation changes.**  Comparing at matched dump index instead adds an
along-track offset `|v| * lag`, which is not small for the AMR runs: median |v| =
0.59 in every run, median lag 0.0134 (AMR local27) and 0.0329 (AMR aei), so the
spurious offset is 0.008 and 0.020.  With `--by-frame` the AMR aei row reads dx at
14.05 = 0.0159 and rms_med 0.0063 -- that row was mostly clock lag, not trajectory
error; interpolated, they are 0.0043 and 0.0039.  For AMR local27 the frame-index
numbers were *smaller* (dx at 10.05 0.0024 vs 0.0041, rms_med 0.0035 vs 0.0041,
dtau_med 0.0016 vs 0.0052): there the lag vector partly cancelled the real
separation, which is why that agreement was not to be trusted.  The uniform runs
have lag 0 and are unchanged to the last digit by the switch.

**Reference sensitivity.**  Taking uniform 1.5/16 aei instead of local27 as the
reference moves the position and proper-time columns only in the third digit
(AMR local27: rms_med 0.0041 either way, rms_max 0.077 vs 0.080, dx at 14.05
0.0050 vs 0.0053, dtau_med 0.0052 vs 0.0059) -- as a symmetric difference should;
the reference only decides which times are trusted.  `dt_drop` is the exception,
because it is a median over the particles captured in *both* runs and that subset
changes with the reference (n_cap 19-26): AMR local27 reads 0.30 against the
local27 reference and 0.73 against the aei one, while the per-particle differences
are of course unchanged.  The magnitude conclusion is the same under either
reference: capture times differ by 0.2-1.3 (median) and up to 3.2 (worst particle)
between runs whose positions agree to ~2e-3.

What that says:

1. **Excluding the particles that fall in first changes nothing.**  Dropping the
   two earliest captures leaves the position medians and rms_med identical; only
   dtau_med and dt_drop move at all, and only in the third decimal (AMR local27:
   0.0052 -> 0.0053 and 0.30 -> 0.325).  Those two particles have rms
   0.0000-0.0037 depending on the pair -- below the all-particle rms_max in every
   one.  The error is not carried by the captured particles.
2. **The tail is a handful of strong-field scatterers, not the captured
   particles.**  Ranked by rms the same names appear in every pair: #53 (0.139 at
   1.8/16, 0.077 at AMR, 0.0029 at 1.5/16 against the local27 reference of the tables
   above; 0.140 / 0.080 / 0.0029 against the aei one), then #54 (0.018-0.029), #55,
   #56, #34, #35.  None of them is ever captured.  These are the particles an
   error budget has to watch, not the ones that disappear.
3. **Two different kinds of outlier -- separate them before quoting rms_max.**
   Excluding #53 (`--exclude 53`) cuts rms_max 5.0x / 4.1x / 2.2x (1.8/16 0.139 ->
   0.028, AMR 0.077 -> 0.019, 1.5/16 0.0029 -> 0.0013; under the aei reference
   0.140 -> 0.027, 0.080 -> 0.020, 0.0029 -> 0.0013); excluding #53 and #54 cuts it
   6.7x / 7.1x / 2.9x under the local27 reference of the tables above (0.139 ->
   0.021, 0.077 -> 0.011, 0.0029 -> 0.0010) and 6.7x / 6.8x / 2.9x under the aei
   reference.  In every case the medians move in the fourth decimal at most, so the
   error is concentrated in a few trajectories rather than spread over the ring.
   #53 is a mask artefact: it is never closer than 1.02 to a puncture, so the
   R_PUN = 1.0 mask cuts none of its samples, while 143 of its 250 dump frames are
   grazing (dp < 1.5); raising `--rpun` to 1.5 reproduces the no-#53 rms_max
   (0.028 / 0.019 / 0.0013) without deleting the particle and also improves the
   medians (uniform 1.8/16 local27: rms_med 9e-4 -> 5e-4, dtau_med 8e-4 -> 5e-4, dx
   at 14.05 1.8e-3 -> 1.5e-3).  The two runs even disagree on how close #53 passes
   (1.02 vs 1.20), which is what a near-miss does to trajectories that already
   diverge.  #54 is *not* such an artefact: it passes 2.0 from a puncture and
   reaches r = 2.4 from the origin, so its 0.021 is genuine strong-field
   disagreement (the r > 3 mask already removes its periastron stretch).  Use the
   mask for #53-type particles; report rms_max both with and without the outliers.
4. **Sampler vs grid.**  Sampler-only at 1.5/16: rms_med 1e-4 (median dx at 14.05
   is 3.2e-4, p90 8.6e-4, max 8.2e-3).  Grid 1.8/16 vs 1.5/16: rms_med 9e-4
   (median dx at 14.05 1.8e-3, p90 1.8e-2).  AMR vs 1.5/16: rms_med 4.1e-3
   (local27) and 3.9e-3 (aei) -- the two AMR samplers agree once the comparison is
   aligned in time; the frame-index comparison made aei look 1.8x worse (6.3e-3 vs
   3.5e-3), and that gap was clock lag, not sampler error.
5. **aei is also the cheaper sampler here**: 25,014 s vs 28,545 s at 1.8/16
   (-12%) and 44,673 s vs 47,340 s at 1.5/16 (-6%), which is the opposite of the
   usual expectation that the cached local fit is the fast one.

Working rule -- **for a particle distribution spread over a large fraction of
the grid**, as the r = 8 ring is: pick the **coarsest uniform grid whose
background survives to the required time**, then set dt so that the measured
lag is below the tolerance (lag = 0 at dt = 0.00625 for dx >= 0.112 on this
background).  On this b = 5 setup that makes uniform 1.8/16 the cheapest
usable tier (1.47e10 point-steps, 7.9 h, 0.60x the cost of uniform 1.5/16),
with uniform 1.5/16 as the verified reference.  If instead the particles sit
in a small sub-volume, redo the (1 + 16 f) estimate before giving up on AMR:
the box wins by 17/(1 + 16 f), which is 9x at f = 0.055 but only 2x at
f = 0.44.

## Fix log: the local27 sampler and the particle clock

Bugs found while running the TwoPunctures ring tests.  Each entry is
symptom -> cause -> fix -> what now guards it.

1. **MPI runs disagreed with single-rank runs (spurious vertical velocity).**
   The `l27` / `local27` stencil is a 5x5x5 window, +-4 cells around the
   particle.  Near a rank face the window stuck out of the local patch: the
   patch's own 3-cell ghost zone is too thin, and clipping the stencil centre
   to the patch destroys the centring.  Equatorial particles sitting on a
   z = 0 rank face acquired a spurious vertical velocity, so 2- and 4-rank
   runs diverged from the scalar run (reported by E. Schnetter).
   *Fix*: keep a halo-padded copy of the ten ADMBase fields for all three time
   levels per level and exchange the outer `HALO_W`-cell band with the
   face-adjacent ranks at every acquire.  The stencil stays exactly centred in
   every decomposition and the values across a face are bit-identical copies
   of the neighbour's array, so the rank count stops being an accuracy
   parameter.  *Guarded by*: `test/check_mpi_identity.sh` (1-, 2- and 4-rank
   dumps compared byte for byte) plus the 2-rank `circ_orbit_mpi` and the
   4-rank `conv100` / TwoPunctures tests in the default set.

2. **Interpolation caches and quality flags did not follow the particle.**
   The `local27` fit caches and the flags `local27_bad`, `local27_lvl2_bad`
   and `local27_cf2_bad` described the history of whichever rank last owned
   the particle.  After an ownership hand-over the new owner took a different
   Christoffel path (fitted polynomial vs analytic assembly) and the
   trajectory branched.  *Fix*: the caches and flags travel with the
   per-particle state in the MPI exchange.

3. **One early refit degraded the whole run (the ps0 accuracy bug).**
   `local27_cf2_bad` was latched for the rest of the run.  A single early
   refit taken while the past time levels were still un-filled switched that
   particle to the analytic Christoffel assembly, which is only first order in
   dt, and it never came back: about 1e-4 in position over t = 1.
   *Fix*: the flag is non-sticky and describes the current refit only.

4. **Secular norm drift of the integrated 4-velocity.**
   An interpolated metric does not preserve `g(u,u) = -1`, and rescaling only
   `u^t` leaves a drift.  *Fix*: in `local27` mode all four components are
   rescaled by `1/sqrt(-g(u,u))` after each step, so the norm holds exactly in
   the interpolated metric frame while the direction produced by the ODE is
   untouched.  A non-finite or wrong-sign norm sets `norm_bad` and goes
   through `particle_loss_policy`; both counters are aggregated and reported.

5. **The particle clock could lag silently (revision 2026.10.28-1).**
   The old guard was `if (max_iter > 100000) { outbd = 1; ... }`, but the tau
   loop is capped by the parameter `max_iter_cap` (default 2000), so that test
   could never fire.  A particle that ran out of tau-steps before the step end
   simply fell short -- and because the ensemble target is the *minimum* live
   particle time, one short particle drags every other particle's clock and
   the whole dump drifts off `cctk_time`.
   *Fix*: falling short now sets `freeze_warn` and prints
   `WARNING (Geodesic): N particle(s) ran out of tau-steps (max_iter_cap=...)
   before the step end and are behind cctk_time ...`.  The particle is
   deliberately kept alive -- dropping it there would change the survival
   statistics of runs that previously only had a silent lag.  The dump also
   gained the `cctk_time` column, so the lag is readable as `cctk_time - t`.
   Measured lag at t = 14.05 on the r = 8 ring: 0 for uniform 1.5/16 and
   1.8/16 at dt = 0.00625; 0.013-0.033 for the AMR run (particle dt = 0.0125);
   0.049 for uniform 1/8 at dt = 0.025; 0.155 for uniform 1/4 at dt = 0.05.

The per-particle diagnostics (`dead`, `renorm_warn`, `norm_bad`,
`freeze_warn`, `rl_last`, `outbd`, `local27_cf2_bad`, `old_level_bad`) are part
of the MPI state exchange, so a particle keeps its history when ownership
moves; they are not written to any file.

### Interpolation-error probe (`probe_file`)

`Geodesic::probe_file` (empty by default) turns on an opt-in diagnostic that
answers "which interpolated quantity carries the largest error?".  Every
`probe_stride`-th geodesic right-hand-side evaluation appends 60 rows to a
sidecar file: the 10 independent `g_{mu nu}`, the 10 independent `g^{mu nu}`
(the inverse of the interpolated `g_{mu nu}`) and the 40 independent
`Gamma^mu_{alpha beta}` (`alpha <= beta`), each as `interp exact err`, plus the
sub-cell offset of the sample point from its nearest grid node (`s_x s_y s_z`,
in units of h), the radius, the position, the particle time and -- for
`local27` -- which Christoffel route produced the values (`fit` = blend of the
nodal 125-node Christoffel fits, `analytic` = assembled from the metric fits).

It is inert when `probe_file` is empty: no extra evaluations, no output, and
the integration path is bit-identical to a build without the probe.  The probe
never refits and never sets `outbd` or the `*_bad` flags, so it cannot perturb
a trajectory; `probe_max_rows` caps the output at block granularity (whole
blocks only, so the count can overshoot by up to 60 rows).  With more than one
MPI rank each rank writes its own `<probe_file>.proc<N>`.  It is skipped for
`Exact = yes` (nothing is interpolated) and for `metric_interp = carpetx`
(whose interpolator is an MPI collective and cannot be re-entered).

The reference column is the **analytic Kerr--Schild** metric of `M`/`a` at the
same event, so the numbers are only meaningful on a Kerr--Schild grid
(`Exact = no` with KerrSchild initial data); on a two-puncture background the
comparison measures the difference from Kerr, which is not an interpolation
error.
