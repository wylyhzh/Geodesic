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
  Three spatial samplers are available
  (`Geodesic::metric_interp`): `local27` (default) fits a 27-term
  (3x3x3) Lagrange polynomial of the metric and of the Christoffel
  symbols around the particle; `aei` evaluates a fourth-order pointwise
  Lagrange interpolant (the `AEILocalInterp` thorn) of the ten 3+1
  fields and their first spatial derivatives at the particle position,
  and `carpet` (named after the original `CarpetInterp`-based
  implementation) evaluates a self-contained quadratic pointwise
  Lagrange interpolant on a 3x3x3 stencil around the particle; both
  build the Christoffel symbols pointwise from the sampled fields and
  derivatives.  No analytic form of the metric is assumed.  The connection includes the `dt(g)` terms of
  `Gamma^mu_{00}` and `Gamma^mu_{0 nu}`.  With the default
  `Geodesic::time_interp = off` the metric value is taken from time
  level 0 and the time derivative is the 2nd-order backward
  difference `(3 L0 - 4 L1 + L2) / (2 dt)` over the three most recent
  ADMBase time levels; `time_interp = linear` instead linearly
  blends the two stored levels with the particle's fractional
  coordinate time and uses the 1st-order difference `(g0 - g1) / dt`
  (not supported for `metric_interp = carpet`: rejected at startup).  Either way the mode works
  for time-dependent metrics, not only stationary ones.  A
  stationary metric is detected automatically (the oldest two levels
  agree to relative 1e-12 on a subsample), in which case `dt(g) = 0`
  exactly and the time difference is skipped.  Time-dependent runs
  need the ADMBase metric fields at `timelevels >= 3` (with
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
- `metric_interp`, `time_interp`, `robust_metric_interp`,
  `adaptive_interp`, and all AMR/stencil machinery are **ignored**.  Only
  `Exact`, `M`, `a` matter for the metric.
- Time is handled by feeding the particle's coordinate time straight into the
  analytic formulas: there are no stored time levels, no blend, and no
  "missing past level" case.
- The metric is exact, so the only error is the **ODE integrator** itself.
  Grid-convergence questions (dx, `time_interp`, order) do not apply.
- This is a single Kerr hole in Kerr--Schild coordinates -- not a numeric
  evolution and not a binary.  It is the reference/verification mode used by
  `circ_orbit` and `ellipse_orbit_amr`.

### `Exact = "no"` (default) -- grid: five valid combinations

Metric and Christoffels are sampled from the ADMBase grid.  Two parameters
combine:

- `metric_interp` (spatial sampler): `local27` (default) / `aei` / `carpet`
- `time_interp` (time handling): `off` (default) / `linear`

That is 3 x 2 = 6 pairs, but **`carpet` is `off`-only** (`linear` is
rejected at startup and marks every particle `outbd`), so **five combinations
are valid**:

| # | `metric_interp` | `time_interp` | what it does |
|---|-----------------|---------------|--------------|
| 1 | local27 | off | 27-point fit on the **current** level; 2nd-order backward time derivative.  The old level is sampled only to feed the robust path's `(g0-g1)/dt`. |
| 2 | local27 | linear | 27-point fit on **both** levels; metric value = 2-level linear blend; 1st-order `(g0-g1)/dt`. |
| 3 | aei | off | AEI 5-node order-4 pointwise, current level, plus its own time derivative. |
| 4 | aei | linear | AEI pointwise as above, plus a 2-level linear blend of the metric. |
| 5 | carpet | off | Carpet pointwise, current level, plus carpet's 2nd-order 3-point time difference. |
| -- | carpet | linear | **unsupported** (startup abort + `outbd`). |

**Semantics to keep straight**

- **`off` is not "no time".**  It means "no 2-level *blend*."  Every sampler
  still computes a time derivative in its own way.  `linear` only adds a 2-level
  linear *blend* of the metric value and switches the time derivative to
  1st-order -- it is **not** a new sampler.
- **`linear` applies to `local27` and `aei` only**, never `carpet`.
- **"Stabilization" is a separate layer, not `time_interp`.**  The
  strong-curvature robust path (`robust_metric_interp` = `aei`/`fourd`/
  `fourdh`/`poly` with `adaptive_interp = on`) is what prevents NaN /
  superluminal drops.  It is independent of `time_interp` but **needs the old
  time level** for its `(g0-g1)/dt` -- which is why `local27` samples the old
  level even in `off` mode (historically, not sampling it caused the `l27`
  particle-drop bug).
- **AMR is level/stencil handling, not a separate "no-time" algorithm.**  A
  particle's stencil is drawn from the finest level that contains it in its
  safety region.  When the *past* level is not yet filled at an AMR cell
  (first integration, or a freshly-grown refinement region), the code flags it
  and falls back to the current level only.  That fallback is the closest
  thing to "no time interpolation" -- but it is a fallback *inside* the same
  samplers, driven by AMR cell state, not by a separate parameter.

## Features

- OpenMP-parallel integration over particles (GSL `rkf45`, adaptive).
  The per-particle computations are fully independent, so the thorn
  scales to `particle_n_total` = 10^4 particles (validated on a
  44-core node; see the paper).
- AMR support (up to four Carpet refinement levels, parameter
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

- AMR runs assume one centered patch per refinement level and a
  single process; multi-patch refined regions and multi-process AMR
  runs are not supported.
- The particle arrays are replicated on every MPI rank
  (`DISTRIB=CONSTANT`) and the integration runs on a single rank, so
  multi-process runs do not distribute the particles across ranks.
  Multi-process AMR (`multi_amr` mode: rank 0 integrates, the other
  ranks sync the particle state) is implemented, but the multi-rank
  path has not been run yet; the testsuite contains a 2-rank test
  (`test/circ_orbit_mpi`) for it that is not part of the currently
  executed test set.
- Concretely: a multi-process grid run (`Exact = no`) silently
  produces wrong trajectories.  The integration runs on rank 0, but a
  particle whose position falls inside another rank's grid patch gets
  no valid field samples there: in a 4-process Kerr test the particle
  barely moved (t = 100: x ~= 11.89, tau ~= 1.46 instead of the
  single-process values) and was dropped shortly after.  Grid-mode
  convergence/production runs must therefore be single process
  (OpenMP threads are fine and scale; MPI ranks are not).
- The `Exact = no` mode reads the metric and its time derivative from
  the ADMBase grid: time-varying metrics need the ADMBase
  metric/lapse/shift fields at `timelevels >= 3`, and within one
  particle step the connection is frozen at the grid's current time,
  so the particle step is capped at the grid time step (see the
  `Exact = no` entry above).

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
| [par/interp_amr_carpet.par](par/interp_amr_carpet.par), [par/interp_amr_aei.par](par/interp_amr_aei.par), [par/interp_amr_l27.par](par/interp_amr_l27.par) | the same orbit as [par/conv100.par](par/conv100.par) under 3-level AMR, with the three `metric_interp` samplers, for a line-by-line comparison against the unigrid conv100 dump |
| [par/aei_2048.par](par/aei_2048.par) | GW150914 non-stationary robustness run (Sec. 5.6): TwoPunctures BBH, moving-box AMR, 2048 Cactus steps, 100 particles |

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
| `metric_interp` | grid-mode sampler: `local27` (default, 27-point Lagrange fit), `aei` (fourth-order pointwise, AEILocalInterp), `carpet` (quadratic pointwise, self-contained) |
| `time_interp` | grid-mode time handling: `off` (default, level-0 value + 2nd-order dt(g)) or `linear` (blend of the two stored levels + 1st-order dt(g)) |
| `robust_metric_interp` | robust-path (local27, `adaptive_interp = on`) interpolation engine: `aei` (default, AEI order-4 pointwise), `fourd` (4D tensor product of the ADM fields: 5-node order-4 space x 3-node time), `fourdh` (same applied to the nodal 4-metric deviation h = g - eta, g composed exactly at each node; slower, strong-curvature path only), `poly` (node-exact 4D tensor product: local27 27-point quadratic space fit x 3-node order-2 time Lagrange over levels [t, t-dt, t-2dt]; metric value is exactly zero-error on all 81 spacetime grid nodes, fixing the spatial node-drift of fourd/fourdh) |
| `geodesic_amr_levels` | number of Carpet refinement levels used for the stencil (1-4, default 3); must equal `Carpet::refinement_levels` for an AMR run |
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

The thorn ships a Cactus testsuite in `test/` (run with
`gmake sim-testsuite` from the Cactus root, or
`CCTK_TESTSUITE_RUN_TESTS="Geodesic/circ_orbit Geodesic/circ_orbit_conserved Geodesic/circ_orbit_mpi Geodesic/circ_orbit_grid Geodesic/circ_orbit_amr Geodesic/circ_orbit_100 Geodesic/ellipse_orbit_amr"
gmake sim-testsuite` to run only these seven tests):

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
  bit-identical to the single-rank reference.  (This test is defined
  but has not been executed yet; it is the validation target for the
  multi-rank path, see Limitations.)
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
  row.  Runs single process, because the refinement boxes must stay
  centred on the origin and only one rank integrates (see
  Limitations).
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
  boundaries in both directions as the orbit completes ~2.42 radial
  periods (t to 3300).  It uses `Exact = yes` deliberately: on the
  2nd-order grid metric (`Exact = no`) the init-time u^t re-solve
  plus coherent integration error push the near-pericentre
  particles across the plunging boundary into the excised sphere,
  where they are dropped; with the built-in analytic metric the
  constants of motion stay within ~2e-6 relative over the whole run
  and no particle is dropped.  The dump is compared against
  reference data row by row.

`util/check_conserved` is a small wrapper script: on first use it
compiles `util/check_conserved.c` (plain C, libc + libm) into
`util/check_conserved.bin` and reuses the binary afterwards, so no
precompiled binary is stored in the repository.

## Notes

- `derivative.h` is the Kranc-generated header for the 27-point
  Lagrange interpolation; `make.code.defn` is its Kranc definition,
  kept for reproducibility.
- This thorn is a pure geodesic integrator: it does not evolve the
  metric and does not require a hydrodynamics thorn.
