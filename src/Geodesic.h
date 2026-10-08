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
      data POINTERS and grid geometry into gstate.LD[map][level] (no copy).
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

/* -------------------- AMR: maximum number of refinement levels --------------------
   Match Carpet::max_refinement_levels so particles can be integrated on the
   finest AMR level of the simulation.  Default 16 covers typical runs. */
#ifndef GEODESIC_MAX_LEVELS
#define GEODESIC_MAX_LEVELS 16
#endif
#ifndef GEODESIC_MAX_MAPS
#define GEODESIC_MAX_MAPS 8   /* Thornburg04 = 7 maps (1 center + 6); 1 spare */
#endif

/* -------------------- Multi-process AMR: per-particle state payload --------------------
   Number of doubles exchanged per particle by the MPI_Allgather of the
   shared particle state (position, velocity, tau, loss flags):
   tt tx ty tz ttv txv tyv tzv ttw dead renorm_warn norm_bad freeze_warn
   rl_last outbd. outbd (particle crossed a local box edge this step) is
   needed by the chunked integrator on every rank to pick a common chunk
   boundary. */
#define GEODESIC_MPI_NPAY 16

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


  /* time_interp = linear: 1 if the OLDER time level's stencil samples
     are invalid (all-zero or non-finite) so the current level must be
     used for the metric/Christoffel blend (theta pinned to 1).  This
     handles the first-integration moment where the particle's coordinate
     time lags the grid and the past time level is not yet filled at the
     particle's AMR cell. */
  CCTK_INT  old_level_bad;

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

  /* sticky robust-path flag: 0 = cheap (historical) path, 1 = robust
     (blend-then-inverse) path.  Only meaningful for metric_interp =
     local27 (sinterp_mode 0) with adaptive_interp = on.  Set by the
     strong-curvature criterion (Geodesic::strong_curv_crit, see
     geodesic_integrate_particle) and never returned to 0 (sticky).
     Carried in the MPI payload so that ownership transfers across
     ranks in multi-process AMR inherit the correct path. */
  CCTK_INT  use_robust;

  /* per-particle, set by cf2_fitting: 1 = the local27 (27-point) fit at
     the current position was degenerate / invalid (all-zero or non-finite
     4-metric), so the official pgd/pgd1/pcf2 coefficients were NOT
     committed (the last good fit is kept).  While set, metric_at_point and
     christoffel_at_point fall back to the AEI extrapolant for this
     particle (metric AND Christoffel together, same frame).  Reset to 0
     when a valid fit is committed.  Derived state: recomputed whenever the
     AMR level / stencil changes, so it is NOT carried in the MPI payload. */
  CCTK_INT  l27_bad;

  /* current owning map (block).  0 = center Cartesian block for all
     particles initially.  Updated on a cross-block event; grid data is
     read from gstate.LD[cur_map][level], and the particle's y[] in
     func_ode is in this map's LOCAL coordinates. */
  CCTK_INT  cur_map;

  /* Phase B (multipath): transient crossing bookkeeping.  Set inside
     func_ode when a block boundary is crossed during this rank's
     integration of the particle; the integrator (geodesic_integrate_
     particle) reads crossed_y[] and restarts the ODE from it.  Both are
     per-particle-per-step, owned by exactly one thread, and are NOT
     part of the MPI payload (GEODESIC_MPI_NPAY). */
  CCTK_INT  crossed;
  CCTK_REAL crossed_y[8];
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
  CCTK_REAL time;                  /* data time of this level's tl0, i.e. the
                                     cctk_time at acquire: the three stored
                                     time levels span [time-2dt, time] (slots
                                     [0]=time, [1]=time-dt, [2]=time-2dt)     */
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
  CCTK_REAL amr_center[3];  /* AMR region centre (physical), par amr_center_* */
  CCTK_REAL amr_radius[3];  /* AMR region half-extent (physical), par amr_radius_* */
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
  /* adaptive_interp (par Geodesic::adaptive_interp): for the local27 sampler,
     1 = compute Christoffel by fitting the METRIC (blend-then-inverse: fit
     4-metric from the 27-cell stencil, invert, differentiate) -- robust in
     strong fields; 0 = historical behaviour (fit/blend the Christoffel
     directly).  Only affects sinterp_mode == 0 (local27); aei/carpet are
     unaffected.  Default 1 (on). */
  CCTK_INT sadaptive;
  /* strong-curvature criterion (par Geodesic::strong_curv_crit): which
     indicator promotes a local27 particle to the robust (blend-then-inverse)
     path.  0 = "u_dot_u" (trigger when raw u·u > u_dot_u_thr, default);
     1 = "alpha" (trigger when lapse < alpha_thr).  Sticky: once set,
     use_robust stays 1 for the particle's life.  Only affects
     metric_interp = local27 (sinterp_mode 0). */
  CCTK_INT  strong_mode;
  CCTK_REAL u_dot_u_thr;   /* par Geodesic::u_dot_u_threshold, default -0.01 */
  CCTK_REAL alpha_thr;     /* par Geodesic::alpha_threshold, default 0.5     */
  /* velocity-component magnitude checks (added discriminant on top of the
     u·u / lapse strong-curvature trigger).  The 4-velocity is normalized to
     u·u = -1, yet a particle near light speed has |u^0|,|u^i| >> 1 while
     u·u stays -1.  A positive threshold enables the corresponding check
     (flag when the magnitude exceeds it); 0 = disabled (default, so the
     existing behaviour is unchanged until the user tunes it).  Only used
     with metric_interp = local27 (sinterp_mode 0). */
  CCTK_REAL u0_thr;        /* par Geodesic::u0_threshold, default 0 (off)   */
  CCTK_REAL ui_thr;        /* par Geodesic::ui_threshold, default 0 (off)   */
  /* strong-region Christoffel source (par Geodesic::strong_christoffel): for
     the local27 robust (use_robust) path.  0 = "hybrid" (default): metric
     VALUE from AEI order-4, Christoffel derivatives from l27 poly27;
     1 = "l27": metric value AND derivatives both from l27 (blend-then-
     inverse).  Only affects metric_interp = local27 (sinterp_mode 0). */
  CCTK_INT  strong_christ_off;
  /* log-space metric interpolation (par Geodesic::log_metric_interp):
     for the local27 robust (use_robust) path, interpolate the metric VALUE
     in log|.| space for the positive ADM fields (gxx/gyy/gzz/lapse) to
     suppress strong-curvature oscillation of the 4th-order (Lagrange) fit;
     the six sign-changing fields (off-diagonal metric + shift) stay in
     regular space.  Same 5-node order-4 tensor-product molecule as AEI a4.
     0 = plain AEI values (default, unchanged behaviour);
     1 = log-space for those fields (qwen1.txt).  Only affects
     metric_interp = local27 (sinterp_mode 0), use_robust. */
  CCTK_INT  log_metric_interp;
  /* robust-path interpolation engine (par Geodesic::robust_metric_interp):
     for the local27 robust (use_robust) path with strong_christ_off = 0
     ("hybrid", the AEI-based strong-region path).
     0 = "aei"  : AEI order-4 pointwise (5-node space x 2/3-node time
                  blend; the current default, unchanged behaviour, optionally
                  in log-space when log_metric_interp is on).
     1 = "fourd": true 4D tensor product -- 5-node order-4 space x 3-node
                  order-2 time Lagrange over the three stored ADMBase levels,
                  giving both the value and all four derivatives from one
                  consistent interpolant.
     2 = "fourdh": same 4D tensor product applied to the nodal 4-metric
                  deviation h = g - eta (g composed exactly at each node
                  from that node's alpha/beta/gamma; eta = diag(-1,1,1,1)
                  added back at the sample point).  Slower (one 4-metric
                  composition per node), meant for the strong-curvature
                  robust path.
     3 = "poly" : node-exact 4D tensor product -- the AEI 5-node order-4
                  spatial Lagrange molecule (exact on each level's 5x5x5 =
                  125 stencil nodes, a strict superset of the 27-node local27
                  stencil) times a 3-node order-2 time Lagrange over the three
                  stored ADMBase levels [t, t-dt, t-2dt].  The full 125x3 =
                  375-spacetime-node interpolant is bit-exact on every stored
                  grid node (the 5-node molecule reproduces each level value on
                  its 5 nodes per axis, and the time weights are
                  Kronecker-delta on the 3 levels), so metric error on grid
                  points is exactly zero.  Spatial accuracy/order (O(dx^4)) is
                  identical to the plain AEI path, so it is numerically stable
                  (no renormalization failures); the 3-node time Lagrange is a
                  strict improvement over AEI's 2-level linear blend.  Same
                  fourd_tw/fourd_tw_a time weights as fourd, so the frame
                  agrees with fourd/fourdh/aei at the nodes.  Falls back to
                  the plain AEI path on boundary / level failure (old_level_bad
                  or metric_static pins the blend to the current level only).
     aei/fourd/fourdh/poly fall back to the plain AEI path on boundary / level
     failure.
     Only affects metric_interp = local27 (sinterp_mode 0), use_robust,
                  strong_christ_off = 0. */
  CCTK_INT  robust_metric_interp;
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
  CCTK_INT  ld_generation, samr_levels, l0_seen_iter, last_integ_iter,
           nlev_seen, last_dump_iter;
  /* subcycling call-batch tracking (Carpet calls POSTSTEP per level per
     substep; one "batch" = all calls sharing a cctk_iteration):
     batch_iter = iteration of the current batch, batch_maxlvl = highest
     level seen so far in this batch; last_integ_time = cctk_time of the
     last firing call (drives sstep = elapsed time). */
  CCTK_INT  batch_iter, batch_maxlvl;
  CCTK_REAL last_integ_time;
  CCTK_INT init_done;          /* the one-shot u^mu init (re-solve) has run */
  /* [map][level]: per-block, per-level grid data.  Only map 0 (center
     Cartesian block) is used while the map-0 filter is active; the multi-
     map extension (claude2.txt Step1) fills every block and lets particles
     move between them via locate_patch + velocity transform. */
  LevelData LD[GEODESIC_MAX_MAPS][GEODESIC_MAX_LEVELS];
  ParticleGeodesic *pgeodesic;
  CCTK_INT pgeodesic_capacity;
  int *owner_of;          /* multi_amr: owning rank of each particle (NULL otherwise) */
  int *owner_lvl;         /* multi_amr: owning level of each particle (-1 = unowned) */
  CCTK_INT owner_cap;
  CCTK_REAL *mmpi_buf;    /* multi_amr: scratch for the particle-state Allgather */
  CCTK_INT mmpi_cap;
  FILE *dfp;
  /* Number of MultiPatch blocks.  Set in the one-shot init from
     MultiPatch_GetMaps.  The multi-block crossing logic in func_ode is
     only valid when there is more than one block; a single-block run
     (per-hole AMR, plain unigrid) has nmaps == 1 and must never enter
     locate_and_cross().  Zero until init. */
  CCTK_INT nmaps;
} GeodesicState;
extern GeodesicState gstate;

/* -------------------- Per-step ODE parameters for func_ode -------------------- */
typedef struct ODE_DATA {
  CCTK_INT m;          /* particle index */
  CCTK_REAL bg;        /* coordinate time at the start of this Cactus step */
  CCTK_REAL a;         /* du^t/dtau, tracked by func_ode for the tau_end prediction */
  GeodesicState *st;   /* thorn state (ODE callback boundary) */
  cGH *cctkGH;         /* Phase B: current geometry for multipatch transforms */
} ode_data;

#endif /* GEODESIC_H */
