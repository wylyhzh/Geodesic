/*
  Geodesic -- pure geodesic particle tracker (Cactus/Einstein Toolkit thorn)
  Copyright (C) 2026  Yuhua Li

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#ifndef GEODESIC_H
#define GEODESIC_H

/*
  Geodesic.h   (pure geodesic integrator, AMR-capable)

  Data structures used by Geodesic_integrate.c to compute:
    - geodesic motion (via Christoffel symbols)
    - 4-velocity normalization (u_mu u^mu = -1)

  Grid model (single process, multi-thread, <= GEODESIC_MAX_LEVELS
  Carpet refinement levels; a single-level run is the default case):

    * CCTK_POSTSTEP is traversed by Carpet in level mode, i.e. the
      scheduled routine is called once per refinement level (per
      component).  With a single process and one centered patch per
      level, geodesics_integrate is called once per level per
      iteration.  On each call it only stores the level's grid-function
      data POINTERS and grid geometry into gstate.LD[level] (no data copy).
      Carpet does not move grid data during one schedule traversal, and
      the particle integration happens inside the finest-level call of
      the same traversal, so the cached pointers are always valid when
      used.

    * When the call pattern of the iteration is complete (level 0 seen
      in this iteration + the current call is on the finest level + not
      yet integrated in this iteration) the particles are integrated
      exactly once, each particle on the finest level whose SAFE region
      (7-cell margin) contains it.  Without AMR (one level) this is
      exactly the plain unigrid behavior: one call, one integration.

    * Inside the integration (OpenMP), every thread keeps a
      thread-local "current level view" (gview) pointing at the level
      of the particle it is currently integrating (switch_to_level()):
      the level's data pointers for all three time levels plus its
      grid geometry.  All grid access, including g4dn() in
      derivative.h, resolves through this view (CCTK_GFINDEX3D
      against the per-level ghost cGH stored in it).

    * Multi-process AMR (gstate.multi_amr): the level-mode call pattern
      is unchanged (every rank is called once per level it owns) and each
      rank caches the pointers of its OWN local patch per level.  Particle
      ownership is by interior boxes: at the trigger, all ranks Allgather
      the per-level interior boxes of their local patches and every rank
      computes the identical owner map (finest level first, then rank
      order; half-open boxes).  Each rank integrates the particles it owns
      and an MPI_Allgather of the full particle state afterwards keeps all
      ranks' views bit-identical.  At most 4 ranks (a 2x2 y,z
      decomposition with one patch per rank per level).  The level-0
      patch-symmetry check applies to single-process runs only.
*/

#include "cctk.h"
#include <stdbool.h>
#include <stdio.h>

/* -------------------- Polynomial interpolation order (fixed) -------------------- */
#ifndef LAGRANGE_POINTS_3
#define LAGRANGE_POINTS_3 3
#endif

#ifndef POLY27
#define POLY27 27
#endif

/* -------------------- AMR: maximum number of refinement levels -------------------- */
#ifndef GEODESIC_MAX_LEVELS
#define GEODESIC_MAX_LEVELS 4
#endif

/* -------------------- Multi-process AMR: per-particle state payload --------------------
   Number of doubles exchanged per particle by the MPI_Allgather of the
   shared particle state (position, velocity, tau, loss flags):
   tt tx ty tz ttv txv tyv tzv ttw dead renorm_warn norm_bad freeze_warn
   rl_last outbd. outbd (particle crossed a local box edge this step) is
   needed by the chunked integrator on every rank to pick a common chunk
   boundary. */
#define GEODESIC_MPI_NPAY 15

/* -------------------- Base data containers at grid points -------------------- */
typedef struct CF2_DATA {
  CCTK_REAL cf2[4][4][4];
} cf2_data;

typedef struct GD_DATA {
  CCTK_REAL gd[4][4];
} gd_data;

/* -------------------- Per-particle cached interpolation data -------------------- */
typedef struct PARTICLEGEODESIC
{
  CCTK_INT  flag; /* cached data valid */
  CCTK_INT  outbd; /* out of bounds */
  CCTK_INT  norm_bad; /* Normalization check bad */
  CCTK_INT  level; /* AMR: refinement level of the current stencil */
  CCTK_INT  point[4]; /* base point grid (i,j,k) */
  CCTK_REAL below[4]; /* polynomial interpolation base point (physical coords) */
  CCTK_REAL dxp[3]; /* AMR: grid spacing of the particle's current level */

  /* ---- values on 3x3x3 stencil (grid-aligned) ---- */
  cf2_data    cf2g[3][3][3]; /* Christoffel symbols (2nd kind) at grid points */
  gd_data     gd[3][3][3]; /* 4-metric g_{mu nu} at grid points */

  /* ---- polynomial coefficients (27-term) for fast evaluation in func_ode ---- */
  CCTK_REAL pcf2[4][4][4][POLY27]; /* Christoffel polynomial interpolation coeffs */
  CCTK_INT  pcf2c[4][4][4]; /* Christoffel is constant? */

  CCTK_REAL pgd[4][4][POLY27]; /* 4-metric polynomial interpolation coeffs */
  CCTK_INT  pgdc[4][4]; /* 4-metric is constant? */

  /* time_interp = linear (local27 only): the metric polynomial fitted on
     the OLDER time level (t - dt), used together with pgd (t) to blend
     the metric to the particle's fractional coordinate time and to form
     the 1st-order time derivative (g0 - g1)/dt.  Unused unless
     time_interp = linear and metric_interp = local27. */
  CCTK_REAL pgd1[4][4][POLY27];
  CCTK_INT  pgd1c[4][4];

  /* time_interp = linear (local27 only): the Christoffel polynomial
     fitted on the OLDER time level (t - dt), 27-term fits of the
     pointwise Christoffel symbols computed on the stencil at that
     level (6th-order FD spatial derivatives, time derivative
     (g0 - g1)/dt).  Blended with pcf2 (t) in christoffel_at_point to
     the particle's fractional coordinate time; for a static metric
     the two fits are bit-identical and the blend is a no-op, so the
     path reduces exactly to time_interp = off.  Unused unless
     time_interp = linear and metric_interp = local27. */
  CCTK_REAL pcf2_1[4][4][4][POLY27];
  CCTK_INT  pcf2_1c[4][4][4];

  /* ---- per-step diagnostics: written inside the OpenMP particle loop
     (each particle by exactly one thread), aggregated single-threaded
     after the loop in geodesics_integrate ---- */
  CCTK_INT  renorm_warn;   /* u.u could not be renormalized this step */
  CCTK_INT  freeze_warn;   /* tau-loop iteration cap hit */
  CCTK_REAL rl_last;       /* last computed u_mu u^mu (diagnostic) */

  /* persistent across steps (NOT reset by the per-step flag reset):
     set by the loss policy when the particle is dropped or flagged;
     a dead particle is skipped by the integration loop forever */
  CCTK_INT  dead;

} ParticleGeodesic;

/* ============================================================
 * AMR per-level data descriptor.
 *
 * Stores POINTERS into the per-level grid function storage
 * (no data copy), plus the level's grid geometry, plus a minimal
 * "ghost cGH" whose geometry fields point at our own arrays so
 * that legacy index macros (CCTK_GFINDEX3D against the per-level
 * ghost cGH, e.g. inside g4dn() in derivative.h) keep working on
 * the current level.
 *
 * The metric fields are stored for all three time levels: [0] is
 * the current time, [1] is t - dt, [2] is t - 2dt (driver
 * convention).  Past levels are readable only when ADMBase keeps
 * them (timelevels >= 3) and Carpet fills them
 * (Carpet::init_fill_timelevels = "yes"); the acquire phase
 * tracks this in timelevels_ok, and metric_static records whether
 * levels 0 and 2 agree (a stationary metric, so d_t g = 0).
 *
 * Validity: filled during the per-level POSTSTEP call; read-only
 * afterwards during the same traversal (integration happens in the
 * finest-level call of the same iteration).  Safe for concurrent
 * read by OpenMP threads.
 * ============================================================ */
typedef struct {
  CCTK_INT  valid;                 /* data acquired                         */
  CCTK_INT  stamp;                 /* cctk_iteration of last acquisition    */
  CCTK_INT  timelevels_ok;         /* past time levels (1,2) were readable  */
  CCTK_INT  metric_static;         /* levels 0 and 2 agree within 1e-12     */

  /* grid geometry of this level (copied scalars, always valid) */
  CCTK_REAL origin[3];             /* physical coordinate of local cell 0   */
  CCTK_REAL dx[3];                 /* grid spacing of this level            */
  CCTK_REAL inv_dx[3];             /* 1/dx                                    */
  CCTK_INT  lsh[3];                /* cctk_lsh (includes ghost zones)       */
  CCTK_INT  lbnd[3];               /* cctk_lbnd                             */
  CCTK_INT  ubnd[3];               /* cctk_ubnd                             */
  CCTK_INT  nghostzones[3];        /* cctk_nghostzones                      */
  CCTK_INT  levfac[3];             /* cctk_levfac                           */
  CCTK_REAL dt;                    /* cctk_delta_time of this level         */
  CCTK_REAL r_lbnd[3];             /* safe region for particles: origin + 7*dx */
  CCTK_REAL r_ubnd[3];             /* safe region: origin + (lsh-8)*dx     */
  /* interior box of this rank's local patch (multi-process AMR
     ownership): the physical region, half-open on the upper end, so
     the interior boxes of all ranks at a level tile the refinement
     region exactly (boundary points belong to the "+"-side rank) */
  CCTK_REAL i_lbnd[3];
  CCTK_REAL i_ubnd[3];

  /* pointers to the per-level grid data (read-only for us), for all
     three time levels: [0] = now, [1] = t - dt, [2] = t - 2dt */
  CCTK_REAL *gxx[3], *gxy[3], *gxz[3], *gyy[3], *gyz[3], *gzz[3];
  CCTK_REAL *alp[3], *betax[3], *betay[3], *betaz[3];

  /* minimal ghost cGH: ONLY the fields assigned in the acquire phase are
     valid (geometry + time scalars); everything else is zero/NULL.       */
  cGH gh;
} LevelData;

/* ============================================================
 * Thorn global state: every run-time mutable scalar of the thorn
 * (cached parfile values, flags, warn-once counters, generation
 * counters, the per-level data descriptors, the particle cache,
 * the dump file handle).  The OpenMP-parallel particle loop only
 * reads this (plus the thread-local gview); all writes happen
 * single-threaded outside the loop.
 * ============================================================ */
typedef struct {
  CCTK_REAL ksm, ksa, sinitial_radius;
  CCTK_REAL xmin_safe, xmax_safe, ymin_safe, ymax_safe, zmin_safe, zmax_safe;
  unsigned int seed;      /* rand_r seed (unsigned by API requirement) */
  CCTK_INT dispno;
  bool sexact;            /* Exact=yes: built-in analytic Kerr-Schild metric */
  bool multi_exact;       /* multi-process + Exact=yes: grid safe-region
                             checks are skipped (each rank owns only a
                             portion of the global grid) */
  bool multi_amr;         /* multi-process + AMR: every rank integrates the
                             particles owned by its local patch (interior-box
                             ownership); the shared particle state is kept
                             consistent by MPI_Allgather (<= 4 ranks) */
  bool sgverbose;
  bool srverbose;
  bool check_velocity;    /* particle velocity check */
  /* spatial metric sampler (par Geodesic::metric_interp):
     0 = local27 (default), 1 = aei (AEILocalInterp), 2 = carpet
     (CarpetInterp).  Grid mode only (Exact = no). */
  CCTK_INT sinterp_mode;
  /* time interpolation of the grid metric (par Geodesic::time_interp):
     0 = off (default), 1 = linear blend of the two stored time levels. */
  CCTK_INT stime_interp;
  /* smoke test (par Geodesic::smoke_timedep_eps, default 0 = disabled):
     the sampled values of the OLDER time level are multiplied by
     (1 - smoke_eps) inside the interpolators so the two levels differ
     in a static run; see the param.ccl entry.  0.0 is a strict no-op. */
  CCTK_REAL smoke_eps;
  /* aei: interpolation + coordinate-system handles (resolved lazily on
     the first grid-mode integration; -1 until then) */
  int aei_handle;
  int aei_coordsys;
  /* carpet: the cGH of the integration call (CarpetInterp needs it);
     single-process only */
  cGH *cur_gh;
  CCTK_INT aei_warned, carpet_warned;  /* one-shot warning flags */
  CCTK_REAL sstep;
  CCTK_INT step_cap_warned, pastlevels_warned;
  CCTK_INT ld_generation, samr_levels, l0_seen_iter, last_integ_iter, nlev_seen;
  CCTK_INT init_done;          /* the one-shot u^mu init (re-solve) has run */
  LevelData LD[GEODESIC_MAX_LEVELS];
  ParticleGeodesic *pgeodesic;
  CCTK_INT pgeodesic_capacity;
  int *owner_of;          /* multi_amr: owning rank of each particle (NULL otherwise) */
  int *owner_lvl;         /* multi_amr: owning level of each particle (-1 = unowned) */
  CCTK_INT owner_cap;
  CCTK_REAL *mmpi_buf;    /* multi_amr: scratch for the particle-state Allgather */
  CCTK_INT mmpi_cap;
  FILE *dfp;
} GeodesicState;
extern GeodesicState gstate;

/* -------------------- Per-step ODE parameters for func_ode -------------------- */
typedef struct ODE_DATA {
  CCTK_INT m;          /* particle index */
  CCTK_REAL bg;        /* coordinate time at the start of this Cactus step */
  CCTK_REAL a;         /* du^t/dtau, tracked by func_ode for the tau_end prediction */
  GeodesicState *st;   /* thorn state (ODE callback boundary) */
} ode_data;

#endif /* GEODESIC_H */
