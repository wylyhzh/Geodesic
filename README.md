# Geodesic

A parallel test-particle geodesic tracker thorn for the
[Einstein Toolkit](https://einsteintoolkit.org) (Cactus).

- **Paper:** Yuhua Li, *Geodesic: a parallel test-particle geodesic tracker
  for the Einstein Toolkit* (2026, preprint draft — PDF included in
  [doc/geodesic_paper.pdf](doc/geodesic_paper.pdf); comments welcome.
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

- `Exact = no` (default): the metric and Christoffel symbols are sampled
  from the ADMBase grid (`gxx..gzz`, `alp`, `betax..betaz`) at the 27
  points of a local stencil and interpolated with a 27-term Lagrange
  polynomial (6th-order finite differences for metric derivatives).
  No analytic form of the metric is assumed.  The connection is built
  from a single time slice with the time derivatives of the metric set
  to zero, so the mode is exact for stationary metrics and an
  approximation for time-dependent spacetimes (the omitted `dt(g)`
  terms of `Gamma^mu_{00}` and `Gamma^mu_{0 nu}` scale with the rate of
  change of the metric).
- `Exact = yes`: the metric and Christoffel symbols are the exact
  analytic Kerr--Schild solution of mass `M` and spin `a`
  (`Geodesic::M`, `Geodesic::a`; the formulas are built into the thorn),
  independent of the grid.  Useful as a reference/verification mode;
  no grid data are needed for the metric itself.

## Features

- OpenMP-parallel integration over particles (GSL `rkf45`, adaptive).
  The per-particle computations are fully independent, so the thorn
  scales to `particle_n_total` = 10^4 particles (validated on a
  44-core node; see the paper).
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
  and `ADMBase` need; the thorn has been validated on a single
  refinement level)

The `KerrSchild` thorn is **not** required: the analytic
Kerr--Schild formulas used by `Exact = yes` are built in.  Activate
`KerrSchild` only if you want a Kerr--Schild *initial grid* for
`Exact = no` runs.

## Limitations

- Single refinement level only (Carpet level 0); AMR support was
  intentionally not developed.
- The particle arrays are replicated on every MPI rank
  (`DISTRIB=CONSTANT`) and the integration runs on a single rank;
  multi-process runs are supported and validated
  (`test/circ_orbit_mpi`, 2 ranks, bit-identical to the single-rank
  result), but they do not distribute the particles across ranks.
- The interpolation mode assumes a stationary metric
  (`dt(g) = 0`); see the `Exact = no` entry above.

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
| `Exact` | metric source: `no` = grid (27-point interpolation), `yes` = analytic Kerr--Schild |
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
`CCTK_TESTSUITE_RUN_TESTS="Geodesic/circ_orbit Geodesic/circ_orbit_conserved Geodesic/circ_orbit_mpi"
gmake sim-testsuite` to run only these three tests):

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
  bit-identical to the single-rank reference.

`util/check_conserved` is plain C (libc + libm) and can be rebuilt
with `cc -O2 -o check_conserved check_conserved.c -lm` on any
machine.

## Notes

- `derivative.h` is the Kranc-generated header for the 27-point
  Lagrange interpolation; `make.code.defn` is its Kranc definition,
  kept for reproducibility.
- This thorn is a pure geodesic integrator: it does not evolve the
  metric and does not require a hydrodynamics thorn.
