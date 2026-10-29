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

  Grid model (OpenMP multi-thread, <= GEODESIC_MAX_LEVELS Carpet
  refinement levels; multi-process runs use the owner-computes scheme
  described at the end of this comment, at most 4 ranks; a single-level
  run is the default case):

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
      (the safe region r_lbnd/r_ubnd, inset by 7 cells; the sampler-specific
      stencil margins are given by safe_margin_cells()) contains it.
      Without AMR (one level) this is
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

    * Multi-process CarpetX (gstate.multi_cx): no ownership at all.
      CarpetX_InterpGridArrays redistributes the interpolation points with
      MPI collectives (AMReX Redistribute + Alltoall/Alltoallv), so every
      rank must issue the identical call sequence, and an adaptive ODE step
      cannot be split between ranks.  Every rank therefore integrates every
      particle from the identical replicated state; the interpolation result
      is computed once on the rank owning the leaf and sent back, so all
      ranks stay bit-identical without any state exchange.  Rank 0 writes
      the dump and the warning summaries.
*/

#include "cctk.h"
#include <stdbool.h>
#include <stdio.h>

/* -------------------- Internal revision stamp --------------------
   Geodesic changes quickly during development and the Cactus build id does
   not say which Geodesic sources produced a run, so the thorn carries its
   own stamp.  It is logged once by Geodesic_ParamCheck and Geodesics_init
   and written into the header of every particle dump; bump it whenever
   behaviour changes.  Format: YYYY.MM.DD-<counter>. */
#ifndef GEODESIC_REVISION
#define GEODESIC_REVISION "2026.10.29-1"
#endif

/* -------------------- Polynomial interpolation order (fixed) -------------------- */
#ifndef LAGRANGE_POINTS_3
#define LAGRANGE_POINTS_3 3
#endif

#ifndef POLY27
#define POLY27 27
#endif

/* local27 (sinterp_mode 3): 125-node 5x5x5 quartic tensor-product polynomial */
#ifndef POLY125
#define POLY125 125
#endif

/* -------------------- Multi-process stencil halo width --------------------
   The local27 stencil window around a centred stencil centre spans
   [tp-5, tp+5] (5x5x5 nodes plus the 6th-order FD window); l27 spans
   [tp-4, tp+4].  A halo of HALO_W=5 cells around the local patch covers
   both, so the centred stencil stays available across a rank face. */
#ifndef GEODESIC_HALO_W
#define GEODESIC_HALO_W 5
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
   boundary.

   The four local27 fit-history flags (local27_cf2_bad, old_level_bad,
   local27_lvl2_bad, local27_bad) travel too.  They describe the state of
   the last refit, which the receiving rank has not performed: a rank that
   has never integrated a particle has them all zero, so when ownership
   moves, the new owner would pick a different Christoffel path (fitted
   poly vs analytic assembly) than the single-process run and the
   trajectory would branch.  The cached fits themselves (pgd125, pcf2_125)
   are NOT sent -- the new owner refits from its own grid data, which is
   identical once the halo is exchanged.

   The norm_dev diagnostic is deliberately NOT part of the payload: every
   rank resets it for all particles at the start of a step and only the
   owner writes it, so each rank's local maximum covers exactly the
   particles it integrated, and the reported maximum is obtained by an
   MPI_MAX reduction over ranks. */
#define GEODESIC_MPI_NPAY 19

/* -------------------- Base data containers at grid points -------------------- */
typedef struct CF2_DATA {
  CCTK_REAL cf2[4][4][4];
} cf2_data;

typedef struct GD_DATA {
  CCTK_REAL gd[4][4];
} gd_data;

/* -------------------- Per-sampler cache pools --------------------
   The stencil arrays below are used by exactly one sampler each: the
   27-node group only by metric_interp = l27 (sinterp_mode 0), the
   125-node group only by metric_interp = local27 (sinterp_mode 3);
   aei / carpet (pointwise) and Exact = yes (analytic) use neither.
   They are therefore NOT inlined in ParticleGeodesic: the pools are
   allocated once, only for the sampler actually selected, and each
   particle points into its own pool slot.  A particle of a mode that
   has no pool keeps NULL pointers (a wrong-mode access faults instead
   of reading stale data). */
typedef struct CACHE27 {
  cf2_data  cf2g[3][3][3];
  gd_data   gd[3][3][3];
  CCTK_REAL pcf2[4][4][4][POLY27];
  CCTK_INT  pcf2c[4][4][4];
  CCTK_REAL pgd[4][4][POLY27];
  CCTK_INT  pgdc[4][4];
} cache27;

typedef struct CACHE125 {
  CCTK_REAL pcf2_125[3][4][4][4][POLY125];
  CCTK_INT  pcf2_125c[3][4][4][4];
  CCTK_REAL pgd125[3][4][4][POLY125];
  CCTK_INT  pgd125c[3][4][4];
} cache125;

/* -------------------- Per-particle cached interpolation data --------------------
   The array members are pointers into the per-sampler pools above; the
   subscripting at the use sites is unchanged (pool[i].pcf2_125 decays to
   a pointer of exactly this member's type). */
typedef struct PARTICLEGEODESIC
{
  CCTK_INT  flag; /* cached data valid */
  CCTK_INT  outbd; /* out of bounds */
  CCTK_INT  norm_bad; /* Normalization check bad */
  CCTK_INT  level; /* AMR: refinement level of the current stencil */
  CCTK_INT  point[4]; /* base point grid (i,j,k) */
  CCTK_REAL below[4]; /* polynomial interpolation base point (physical coords) */
  CCTK_REAL dxp[3]; /* AMR: grid spacing of the particle's current level */

  /* ---- values on 3x3x3 stencil (grid-aligned): pool27, l27 only ---- */
  cf2_data   (*cf2g)[3][3]; /* Christoffel symbols (2nd kind) at grid points */
  gd_data    (*gd)[3][3];   /* 4-metric g_{mu nu} at grid points */

  /* ---- polynomial coefficients (27-term) for fast evaluation in func_ode ---- */
  CCTK_REAL (*pcf2)[4][4][POLY27]; /* Christoffel polynomial interpolation coeffs */
  CCTK_INT  (*pcf2c)[4][4];        /* Christoffel is constant? */

  CCTK_REAL (*pgd)[4][POLY27];     /* 4-metric polynomial interpolation coeffs */
  CCTK_INT  (*pgdc)[4];            /* 4-metric is constant? */

  /* 1 if the OLDER time level's stencil samples are invalid (all-zero or
     non-finite): the past time level is not yet filled at the particle's
     AMR cell (typical at the very first integration).  The local27 time
     blend then pins to the current level only (local27_time_weights). */
  CCTK_INT  old_level_bad;

  /* ---- local27 (sinterp_mode 3): 125-node 5x5x5
     quartic tensor-product fits (node-exact on the 5x5x5 stencil around
     the particle's nearest node).  Three stored time levels: [0] = t,
     [1] = t - dt, [2] = t - 2dt.  pcf2_125[lvl] fits the NODAL Christoffel
     symbols at time level lvl (6th-order FD metric derivative + nodal
     inverse, as in the local27 path); the particle Christoffel is the
     time-Lagrange blend of the level fits (fast path).  pgd125[lvl] fits
     the 4-metric at time level lvl; used for the metric value (time
     blend) and, when local27_cf2_bad, for the analytic Christoffel assembly
     (metric derivatives + inverse).  Pool: pool125, allocated only when
     metric_interp = local27. */
  CCTK_REAL (*pcf2_125)[4][4][4][POLY125];
  CCTK_INT  (*pcf2_125c)[4][4][4];
  CCTK_REAL (*pgd125)[4][4][POLY125];
  CCTK_INT  (*pgd125c)[4][4];

  /* local27 (sinterp_mode 3): 1 = the nodal Christoffel samples were
     non-finite at some refit, so the pcf2_125 fits are unusable and
     christoffel_at_point assembles the Christoffel analytically from
     the pgd125 metric fits instead.

     Recomputed at every refit (non-sticky): a particle uses the nodal
     fast path whenever that level's Christoffel samples are finite.  Two
     conditions can set it: the 6th-order FD window of a stencil node
     reaching the central singularity / excised region, and (by far the
     commonest) a stored past time level not yet filled at the
     particle's cell, which happens at the first refits of a run that
     starts from un-filled time levels.

     It used to be sticky, which was a real accuracy bug rather than a
     cost question: the analytic assembly samples the metric VALUE from
     level 0 only, so it is piecewise constant in time and first order in
     dt, while the fast path blends three levels.  On the u18 BBH grid the
     aei-vs-local27 trajectory difference scaled like dt (q = 0.95) purely
     because a start_time = 0 refit latched the flag for the whole run.
     The flag now clears as soon as the samples are good; switching paths
     mid-integration is harmless now that both paths blend in time.
     Cost: the nodal fast path is ~11% slower than the analytic assembly
     (measured on tp_far_local27_025_t80). */
  CCTK_INT  local27_cf2_bad;

  /* local27 (sinterp_mode 3): 1 = the level-2 (t - 2dt) metric samples are
     invalid (all-zero or non-finite), so the time blend uses the two
     levels [t, t-dt] only.  Recomputed at every refit (non-sticky). */
  CCTK_INT  local27_lvl2_bad;

  /* local27 (sinterp_mode 3): 1 = the staged (blended) metric at the particle
     position was invalid (degenerate all-zero / non-finite), so the pgd125
     fits were not committed and the particle is dropped (this mode has no
     AEI fallback by design).  Recomputed at every refit (non-sticky). */
  CCTK_INT  local27_bad;

  /* ---- per-step diagnostics: written inside the OpenMP particle loop
     (each particle by exactly one thread), aggregated single-threaded
     after the loop in geodesics_integrate ---- */
  CCTK_INT  renorm_warn;   /* u.u could not be renormalized this step */
  CCTK_INT  freeze_warn;   /* tau-loop iteration cap hit */
  CCTK_REAL rl_last;       /* last computed u_mu u^mu (diagnostic) */
  CCTK_REAL norm_dev;      /* |g(u,u)+1| of this step, measured before the
                              renormalization below absorbs it; 0 when the
                              particle was not integrated this step */

  /* persistent across steps (NOT reset by the per-step flag reset):
     set by the loss policy when the particle is dropped or flagged;
     a dead particle is skipped by the integration loop forever */
  CCTK_INT  dead;

  /* per-particle, set by cf2_fitting: 1 = the l27 (27-point) fit at the
     current position was degenerate / invalid (all-zero or non-finite
     4-metric, or NaN-poisoned Christoffel stencil samples), so the
     official pgd/pcf2 coefficients were NOT committed (the last good fit
     is kept).  While set, the particle has no usable metric/Christoffel
     in the l27 mode (no fallback interpolant): metric_at_point leaves gdn
     zeroed and the norm check / loss policy drop the particle.  Reset to
     0 when a valid fit is committed.  Derived state: recomputed whenever
     the AMR level / stencil changes, so it is NOT carried in the MPI
     payload. */
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
 * (Carpet::init_fill_timelevels = "yes"); metric_static records
 * whether levels 0 and 2 agree (a stationary metric, so d_t g = 0).
 *
 * Validity: filled during the per-level POSTSTEP call; read-only
 * afterwards during the same traversal (integration happens in the
 * finest-level call of the same iteration).  Safe for concurrent
 * read by OpenMP threads.
 * ============================================================ */
typedef struct {
  CCTK_INT  valid;                 /* data acquired                         */
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

  /* multi-process stencil halo (grid mode, l27/local27): padded copies
     of the ten fields for all three time levels.  The local patch's
     values are copied into the interior and the outer HALO_W-cell band
     is exchanged with the face-adjacent ranks, so a centred stencil
     window that sticks out over a rank face reads the neighbour's real
     values instead of being clipped (the rank-face symmetry-breaking
     bug reported by E. Schnetter).  halo is NULL when single-process
     (or a pointwise sampler): the view then points at the raw arrays.
     halo_alloc_lsh records the lsh the buffers were sized for. */
  CCTK_REAL *halo[10][3];
  CCTK_REAL *halo_base[10][3];   /* malloc base of halo[f][tl] (halo is offset) */
  CCTK_INT   halo_alloc_lsh[3];
  /* halo extent per direction (valid read indices into the padded
     arrays, local frame): [-HALO_W, lsh-1+HALO_W] */
  CCTK_INT   halo_lo[3], halo_hi[3];
  /* stride the padded arrays are laid out with: lsh+2*HALO_W while the
     halo buffers are active, lsh otherwise.  The ghost cGH's cctk_ash
     points here so CCTK_GFINDEX3D in derivative.h matches the layout
     the halo buffers (and the raw arrays when halo is NULL) use. */
  CCTK_INT   halo_ash[3];

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
  bool sexact;            /* Exact=yes: built-in analytic Kerr-Schild metric */
  bool multi_exact;       /* multi-process + Exact=yes: grid safe-region
                             checks are skipped (each rank owns only a
                             portion of the global grid) */
  bool multi_amr;         /* multi-process + AMR: every rank integrates the
                             particles owned by its local patch (interior-box
                             ownership); the shared particle state is kept
                             consistent by MPI_Allgather (<= 4 ranks) */
  bool multi_cx;          /* multi-process + metric_interp = carpetx: the
                             integration is replicated on every rank (the
                             CarpetX interpolator is an MPI collective), so
                             no owner map and no state exchange are needed
                             and only rank 0 reports */
  bool pstart_done;       /* particle_start_time: the particles have already
                             been released (the hold and the one-shot clock
                             re-anchoring happen once) */
  int nint_steps;         /* integration steps seen so far, used by
                             particle_start_steps */
  bool sgverbose;
  bool srverbose;
  bool check_velocity;    /* particle velocity check */
  CCTK_REAL norm_dev_max; /* largest |g(u,u)+1| seen before renormalization
                             over the whole run (diagnostic only: how much
                             one ODE step in the interpolated metric
                             violates the mass shell) */
  CCTK_REAL norm_dev_step; /* same, but only for the current integration
                             call; reset after each report */
  /* spatial metric sampler (par Geodesic::metric_interp):
     0 = l27 (simplest 27-point fit), 1 = aei (AEILocalInterp),
     2 = carpet, 3 = local27 (the self-contained 125-node fit),
     4 = carpetx (CarpetX InterpGridArrays on the ADMBaseX grid).
     Grid mode only (Exact = no); the default is local27. */
  CCTK_INT sinterp_mode;
  /* aei: interpolation + coordinate-system handles (resolved lazily on
     the first grid-mode integration; -1 until then) */
  int aei_handle;
  int aei_coordsys;
  /* carpetx: the CarpetX interpolation handle (resolved lazily; -1 until
     then) and the driver flag set once per integration call.  CarpetX
     does not hand Geodesic a level-mode cGH, so the level-0 LevelData is
     synthesised from the CarpetX grid parameters instead (see
     carpetx_fill_geometry). */
  int cx_handle;
  bool carpetx_mode;
  /* carpet: the cGH of the integration call (CarpetInterp needs it);
     single-process only */
  cGH *cur_gh;
  CCTK_INT aei_warned, carpet_warned;  /* one-shot warning flags */
  /* one-shot start-up report that the stencil windows of owned particles
     straddle a rank face and are served by the halo exchange (the
     situation of the bug reported by E. Schnetter) */
  CCTK_INT face_reported;
  CCTK_REAL sstep;
  CCTK_INT step_cap_warned, pastlevels_warned;
  CCTK_INT  ld_generation, samr_levels, l0_seen_iter, last_integ_iter,
           nlev_seen, last_dump_iter;
  /* time stamp of the last block written to the particle dump file; a block
     whose own time is not newer than this is a repeat (e.g. the first step of
     a restarted run re-emitting the recovered state) and is skipped */
  CCTK_REAL last_dump_time;
  /* 1 once CCTK_POST_RECOVER_VARIABLES has reported a real restart
     (cctk_iteration > 0).  Only then may the dump continue an existing file:
     a fresh run in a directory that still holds an old dump must start a new
     one, even when a checkpointing IO thorn is active. */
  CCTK_INT recovered;
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
  /* Per-sampler stencil pools, same capacity as pgeodesic: pool27 is
     allocated only for metric_interp = l27, pool125 only for
     metric_interp = local27; both stay NULL for aei / carpet and for
     Exact = yes.  Each particle's array members point at its own slot. */
  cache27  *pool27;
  cache125 *pool125;
  int *owner_of;          /* multi_amr: owning rank of each particle (NULL otherwise) */
  int *owner_lvl;         /* multi_amr: owning level of each particle (-1 = unowned) */
  CCTK_INT owner_cap;
  CCTK_REAL *mmpi_buf;    /* multi_amr: scratch for the particle-state Allgather */
  CCTK_INT mmpi_cap;
  FILE *dfp;
  /* Interpolation-error probe (Geodesic::probe_file).  probe_on is the
     one-shot parse of the (run-fixed) keyword: 0 = the probe is entirely
     inert and none of its code runs.  pfp is the sidecar, opened lazily at
     the first probe row and kept open for the whole run (closed by
     Geodesics_finalise).  probe_idx / probe_rows are the only gstate fields
     written from inside the OpenMP particle loop; both are touched through
     atomics or inside the probe's critical section. */
  FILE *pfp;
  CCTK_INT probe_on;
  CCTK_INT probe_stride;
  CCTK_INT probe_max_rows;
  CCTK_INT probe_idx;
  CCTK_INT probe_rows;
  /* Number of MultiPatch blocks.  Set in the one-shot init from
     MultiPatch_GetMaps.  The multi-block crossing logic in func_ode is
     only valid when there is more than one block; a single-block run
     (per-hole AMR, plain unigrid) has nmaps == 1 and must never enter
     locate_and_cross().  Zero until init. */
  CCTK_INT nmaps;
} GeodesicState;
extern GeodesicState gstate;

/* CarpetX grid parameters (see carpetx_fill_geometry in
   Geodesic_integrate.c); returns -1 for an invalid domain. */
int geodesic_carpetx_domain (double lo[3], double hi[3], int nc[3], int *ngz);

/* -------------------- Per-step ODE parameters for func_ode -------------------- */
typedef struct ODE_DATA {
  CCTK_INT m;          /* particle index */
  CCTK_REAL bg;        /* coordinate time at the start of this Cactus step */
  CCTK_REAL a;         /* du^t/dtau, tracked by func_ode for the tau_end prediction */
  GeodesicState *st;   /* thorn state (ODE callback boundary) */
  cGH *cctkGH;         /* Phase B: current geometry for multipatch transforms */
} ode_data;

#endif /* GEODESIC_H */
