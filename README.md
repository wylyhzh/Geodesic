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

- `ADMBase` — always required (the schedule reads its fields); for
  `Exact = yes` only the grid geometry (dimensions, spacing, origin,
  ghost zones) is used, the metric values are ignored
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
  is exchanged between the ranks once per Cactus step, so multi-rank
  runs are bit-for-bit reproducible (verified against single-rank
  baselines in the conv test series).  The testsuite's 2-rank test
  (`test/circ_orbit_mpi`) and the 4-rank `conv100` / TwoPunctures
  tests are part of the default test set (see Tests below).
- The `Exact = no` mode reads the metric and its time derivative from
  the ADMBase grid: time-varying metrics need the ADMBase
  metric/lapse/shift fields at `timelevels >= 3`, and within one
  particle step the connection is frozen at the grid's current time,
  so the particle step is capped at the grid time step (see the
  `Exact = no` entry above).
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
| `particle_n` | number of fixed particles (initial data from the parfile); 0 disables |
| `particle_n_total` | total number of particles (max 10000); the remainder beyond `particle_n` are randomly placed with zero velocity |
| `particle_x/y/z[0..99]` | initial positions of the first `particle_n` particles |
| `particle_tv/xv/yv/zv[0..99]` | initial 4-velocity `u^mu` (see above) |
| `step` | coordinate-time advance per Cactus iteration; 0 uses `cctk_delta_time` |
| `max_iter_cap` | max ODE tau-steps per particle per Cactus iteration (default 2000) |
| `Exact` | metric source: `no` = grid, `yes` = analytic Kerr--Schild |
| `metric_interp` | grid-mode sampler: `l27` (simplest 27-point Lagrange fit, current level only, particle dropped on fit failure), `local27` (default; 125-node 5x5x5 quartic tensor-product fit, self-contained, no AEI), `aei` (AEILocalInterp pointwise order-4, 2-level linear time blend), `carpet` (AEI order-4 spatial molecule + carpet-style time difference) |
| `geodesic_amr_levels` | number of Carpet refinement levels used for the stencil (1-16, default 3); must equal `Carpet::refinement_levels` for an AMR run |
| `M` / `a` | Kerr mass and spin used by `Exact = yes` (built-in analytic formulas) |
| `initial_radius` / `excised_radius` | safe shell for the `respawn` policy: outside `initial_radius`, inside `excised_radius` is rejected |
| `particle_loss_policy` | what to do with lost particles: `drop` (default), `flag`, `respawn` |
| `particle_rand_seed`, `particle_rand_*min/max`, `particle_middle_sp` | random (re)placement controls (respawn policy) |
| `particle_dump_every` | dump full particle state to `particle_dump_file` every k iterations (0 = off; rank 0 only) |
| `particle_dump_file` | dump file name, relative to the working directory (default `geodesic_particles.txt`; parent directories are created as needed) |
| `gverbose` / `rverbose` | diagnostics |

## Output

The evolved particle state (`tau`, `t`, `x, y, z`, `u^t..u^z`) is kept
in the `particle_*` arrays declared in `interface.ccl`; save them with
the Einstein Toolkit's standard output thorns (HDF5, Checkpoint, ...)
like any other grid function.  With `particle_dump_every = k > 0` the
thorn additionally writes a plain-text dump (`t, x, y, z, u^t..u^z,
tau` per particle) to the file given by `particle_dump_file` (default
`geodesic_particles.txt`, relative to the working directory; parent
directories are created as needed), starting with the initial state and
then one block of rows every k iterations.

## Tests

The thorn ships a Cactus testsuite in `test/` with 18 tests, driven by
the official `RunTest.pl` infrastructure.  The convenient launcher is
`test/run_tests.sh`:

- `./run_tests.sh` (no argument) runs the 8-test default set, which
  covers all four metric-sampling back-ends and both metric families
  (stationary Kerr grid and the two-puncture merger background):
  `conv100`, `conv100_l27`, `conv100_local27`, `conv100_aei`,
  `conv100_carpet`, `circ_orbit`, `tp_local27_025`,
  `tp_far_local27_025_t80`.
- `./run_tests.sh -all` runs all 18 tests in `test/test.ccl`.

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
- `conv100` is the paper's full grid-convergence run: the P0 circular
  orbit (r = 10, M = 2, a = 0.5) on a 100^3 grid (dx = 0.5),
  `Exact = no`, 1500 steps (t = 150.1, ~1 orbital period), 4 MPI
  ranks.  The single-particle dump is compared against reference data
  row by row.
- `conv100_l27`, `conv100_local27`, `conv100_aei`, `conv100_carpet`
  are the same orbit shortened to 200 steps (t = 20.0), one per
  metric-sampling back-end (`metric_interp = l27` / `local27` / `aei`
  / `carpet`; `aei` and `carpet` sample through the
  `AEILocalInterp` / `LocalInterp` interpolation interface).  Each
  dump must reproduce its reference to RELTOL 1e-12.
  `conv100_l27` runs single process: the equatorial ring sits on rank
  faces in the default 2x2 (y,z) multi-rank decomposition, where the
  `l27` stencil is clipped out of the particle (see Limitations).
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
