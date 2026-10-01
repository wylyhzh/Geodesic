
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

/* ============================================================================================
   Geodesic_integrate.c
   Main driver of the Geodesic thorn:
     - per-level AMR grid acquisition (Carpet POSTSTEP traversal)
     - particle geodesic integration (GSL rkf45 in proper time)
     - 4-velocity renormalization (u_mu u^mu = -1, future-directed)
     - loss policy for out-of-bounds / bad-norm particles
       (drop / flag / respawn, selectable at run time)
   ============================================================================================ */

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"
#include "cctk_Functions.h"
#include <stddef.h>                     /* defines size_t */
#include <math.h>
#include <pthread.h>                    /* mutex guarding Util table creation */
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#ifdef CCTK_MPI
#include <mpi.h>
#endif
#include "cctk_Interp.h"       /* CCTK_InterpHandle / CCTK_InterpLocalUniform / handles */
/* CommOverloadables.h declares prototypes via the OVERLOADABLE(name) macro,
   which must be provided by OverloadMacros.h (mirrors cctk_Comm.h). */
#include "OverloadMacros.h"
#define OVERLOADABLE(name) OVERLOADABLE_PROTOTYPE(name)
#include "CommOverloadables.h" /* CCTK_InterpGridArrays (CarpetInterp driver) */
#undef OVERLOADABLE
#include "util_Table.h"        /* Util_Table* (AEI/Carpet parameter tables) */
#include "Geodesic.h"

/* --------------------------------------------------------------------
   MultiPatch (Coordinates thorn) function prototypes.

   Geodesic uses the Coordinates thorn's MultiPatch functions only when that
   thorn is present (all calls are guarded by CCTK_IsFunctionAliased).
   Geodesic's own include chain (cctk.h -> Geodesic_Prototypes.h) does not
   include Coordinates_Prototypes.h, so the calls would otherwise be
   implicitly declared.  GCC>=12 rejects implicit function declaration as
   an error; declare the two functions explicitly here.  The runtime guards
   ensure no call is ever made when the functions are not registered.
   -------------------------------------------------------------------- */
extern CCTK_INT MultiPatch_GetMap (const CCTK_POINTER_TO_CONST cctkGH);
extern CCTK_INT MultiPatch_GetMaps (const CCTK_POINTER_TO_CONST cctkGH);
extern CCTK_INT MultiPatch_GlobalToLocal (const CCTK_POINTER_TO_CONST cctkGH ,
 const CCTK_INT ndims ,
 const CCTK_INT npoints ,
 const CCTK_POINTER_TO_CONST globalcoords ,
 CCTK_INT* patch ,
 const CCTK_POINTER localcoords ,
 const CCTK_POINTER dadx ,
 const CCTK_POINTER ddadxdx);
extern CCTK_INT MultiPatch_LocalToGlobal (const CCTK_POINTER_TO_CONST cctkGH ,
 const CCTK_INT ndims ,
 const CCTK_INT npoints ,
 const CCTK_INT* patch ,
 const CCTK_POINTER_TO_CONST localcoords ,
 const CCTK_POINTER globalcoords ,
 const CCTK_POINTER dxda ,
 const CCTK_POINTER det_dxda ,
 const CCTK_POINTER dadx ,
 const CCTK_POINTER ddxdada ,
 const CCTK_POINTER ddadxdx ,
 const CCTK_POINTER dddxdadada);

#define lN      27                      /* number of data points to fit */

/* NOTE:
   All run-time state of the thorn lives in the single gstate object
   (GeodesicState, declared in Geodesic.h): cached parfile values,
   flags, warn-once counters, generation counters, the per-level grid
   data descriptors, the particle cache and the dump file handle.
   Grid access in the OpenMP-parallel code goes through the
   THREAD-LOCAL level view gview, defined further below together with
   the AMR machinery (each OpenMP thread points it at the level of the
   particle it is currently integrating, switch_to_level()).
*/
GeodesicState gstate = { .seed = 123456789u, .dispno = -10,
                         .l0_seen_iter = -1, .last_integ_iter = -1,
                         .last_dump_iter = -1,
                         .last_integ_time = 0.0,
                         .batch_iter = -1, .batch_maxlvl = -1,
                         .samr_levels = 1, .nlev_seen = 1,
                         .aei_handle = -1, .aei_coordsys = -1,
                         .log_metric_interp = 0,
                         .robust_metric_interp = 0 };

/* ============================ AMR support (max GEODESIC_MAX_LEVELS levels) ====

   Single process, multi-thread.  geodesics_integrate is called once per
   refinement level at CCTK_POSTSTEP (Carpet level-mode traversal); each
   call only stores the level's data POINTERS + geometry into LD[map][level]
   (no copy).  When the call pattern of the iteration is complete the
   particles are integrated once, each on its finest safe level.  A
   single-level run collapses to the plain unigrid behavior (one call,
   one integration per iteration).

   Multi-process, Exact=yes (multi_exact): only rank 0 integrates the
   particles and writes the dump (the acquire phase still runs on every
   rank, caching that rank's local grid view); the level selection and
   the local safe-region checks are disabled, because each rank owns
   only a portion of the global grid.

   Multi-process + AMR (multi_amr): every rank integrates the particles
   owned by its local patch (interior-box ownership; see the Phase 1
   owner map in geodesics_integrate), and the shared particle state is
   kept bit-identical on all ranks by an MPI_Allgather after the
   integration and the loss policy (geodesic_mmpi_sync).  At most 4
   ranks (a 2x2 y,z decomposition with one patch per rank per level).

   All grid access below goes through the THREAD-LOCAL "current level
   view" (gview) switched by switch_to_level().  Legacy code that uses
   CCTK_GFINDEX3D (e.g. g4dn() in derivative.h) keeps working via the
   per-level ghost cGH stored in the view.
   ============================================================================ */
#ifndef GEODESIC_TLS
#define GEODESIC_TLS __thread
#endif

/* Per-thread "current level view": a snapshot of the level an OpenMP
   thread is currently integrating on (data pointers for all three
   time levels + grid geometry + the level's ghost cGH), built by
   switch_to_level() from gstate.LD[map][level].  All grid access in this
   file and in derivative.h (g4dn, ndg4dn, cf2_grid, ...) resolves
   through it. */
typedef struct {
  cGH *GH;
  CCTK_REAL *gxx[3], *gxy[3], *gxz[3], *gyy[3], *gyz[3], *gzz[3];
  CCTK_REAL *alp[3], *betax[3], *betay[3], *betaz[3];
  CCTK_REAL dt;
  CCTK_INT metric_static;
  CCTK_REAL delta_inv[3], origin_space[3], delta_space[3], r_lbnd[3], r_ubnd[3];
  CCTK_INT lsh[3];
  CCTK_INT map, level, gen;
} GeodesicView;
static GEODESIC_TLS GeodesicView gview = { .map = -1, .level = -1, .gen = -1 };

/* linear index into the current thread's level arrays (i,j,k are 0-based) */
#define LD_INDEX(i,j,k) ((i) + gview.lsh[0]*((j) + gview.lsh[1]*(k)))

/* point this thread's grid view at (map, level l) (self-guarded, cheap) */
static void switch_to_level(int map, int l)
{
  if (gview.map == map && gview.level == l && gview.gen == gstate.ld_generation) return;
  gview.map = map;
  gview.level = l;
  gview.gen   = gstate.ld_generation;

  LevelData *L = &gstate.LD[map][l];
  gview.GH = &L->gh;
  {
    CCTK_REAL **pp[10] = { L->gxx, L->gxy, L->gxz, L->gyy, L->gyz, L->gzz,
                           L->alp, L->betax, L->betay, L->betaz };
    CCTK_REAL **qq[10] = { gview.gxx, gview.gxy, gview.gxz, gview.gyy, gview.gyz, gview.gzz,
                           gview.alp, gview.betax, gview.betay, gview.betaz };
    for (int i=0; i<10; i++)
      for (int tl=0; tl<3; tl++)
        qq[i][tl] = pp[i][tl];
  }
  gview.dt = L->dt;
  gview.metric_static = L->metric_static;
  for (int i=0; i<3; i++)
  {
    gview.lsh[i]         = L->lsh[i];
    gview.delta_space[i] = L->dx[i];
    gview.delta_inv[i]   = L->inv_dx[i];
    gview.origin_space[i]= L->origin[i];
    gview.r_lbnd[i]      = L->r_lbnd[i];
    gview.r_ubnd[i]      = L->r_ubnd[i];
  }
}

/* 1 if the metric stored at time levels 0 and 2 agrees (interior cells,
   every 4th cell, all 10 fields, relative tolerance 1e-12): a
   stationary metric, so d_t g = 0 and the time-derivative FD can be
   skipped. */
static int metric_is_static(const LevelData *L)
{
  const CCTK_REAL *const *f[10] = {
    L->gxx, L->gxy, L->gxz, L->gyy, L->gyz, L->gzz,
    L->alp, L->betax, L->betay, L->betaz };
  CCTK_INT i, j, k, fi;
  CCTK_REAL maxrel = 0.0;
  for (fi=0; fi<10; fi++)
    for (k=L->nghostzones[2]; k < L->lsh[2]-L->nghostzones[2]; k+=4)
      for (j=L->nghostzones[1]; j < L->lsh[1]-L->nghostzones[1]; j+=4)
        for (i=L->nghostzones[0]; i < L->lsh[0]-L->nghostzones[0]; i+=4)
        {
          const CCTK_INT idx = i + L->lsh[0]*(j + L->lsh[1]*k);
          const CCTK_REAL rel = fabs(f[fi][0][idx] - f[fi][2][idx]) / (1.0 + fabs(f[fi][0][idx]));
          if (rel > maxrel) maxrel = rel;
        }
  return maxrel < 1.0e-12;
}

/* safe-region margin in cells: the local27 (poly27) branch reads a 3x3x3
   stencil (metric fit) and a 6th-order central FD for metric derivatives,
   both centred on the node `point` and needing [point-3, point+3] (the
   FD uses nodes point-3..point+3; the 3x3x3 fit needs only point-1..point+1).
   Hence the centre must be at least 3 cells from each edge of the local
   patch -> margin 3.  (The earlier value of 7 over-shrinks the safe region
   and wrongly rejects valid particles that sit within 3..6 cells of the
   patch edge -- e.g. an equatorial particle on a z=0 symmetry plane inside
   a 1p MultiPatch block.  The probe confirmed: z=0 particle rejected when
   the safe region started 7 cells from the edge.)
   The aei/carpet branches (sinterp_mode 1/2) are pointwise (a 3-node
   quadratic {tp-2,tp-1,tp}), so a margin of 2 cells is correct for them. */
static int safe_margin_cells(void)
{
  return (gstate.sinterp_mode == 0) ? 3 : 2;
}

/* in-range test for one direction of one level:
   use the margin-appropriate safe region when it is non-empty; if the
   region is degenerate (level too thin in that direction), fall back to
   the interior box.  A particle on the symmetry plane there has its
   derivative in that direction vanish by symmetry, so the interior-box
   test is the correct "in-range" criterion. */
static int in_level_dir(const double y[], const LevelData *L, int d)
{
  const int margin = safe_margin_cells();
  double lo = L->origin[d] + (CCTK_REAL)margin * L->dx[d];
  double hi = L->origin[d] + (CCTK_REAL)(L->lsh[d]-1-margin) * L->dx[d];
  if (lo > hi) { lo = L->i_lbnd[d]; hi = L->i_ubnd[d]; }
  return (y[d+1] >= lo && y[d+1] <= hi);
}

/* finest level (of the given map) whose SAFE region contains y; -1 if none */
static int pick_level(const double y[], int map)
{
  for (int l = gstate.samr_levels-1; l >= 0; l--)
  {
    if (!gstate.LD[map][l].valid) continue;
    const LevelData *L = &gstate.LD[map][l];
    if (in_level_dir(y, L, 0) && in_level_dir(y, L, 1) &&
        in_level_dir(y, L, 2))
      return l;
  }
  return -1;
}

/* ============================================================================================
   Phase B: multipatch (Thornburg04) support.

   A particle's y[1..3] is expressed in the LOCAL coordinates of its current
   block (gstate.pgeodesic[m].cur_map).  For the center Cartesian block
   (map 0) these coincide with global physical coordinates; for the six outer
   blocks they are that block's own (curvilinear) coordinates, related to the
   global physical coordinates by the block Jacobian.

   The geodesic ODE is covariant: inside a single block the particle is
   integrated with that block's own metric + Christoffels (no transformation
   of the Christoffels is needed -- only the metric/Christoffel are re-sampled
   on the new block's grid).  Two things change when a particle crosses a block
   boundary:
     1. its position must be re-expressed in the new block's local coords;
     2. its spatial 4-velocity components transform between the two blocks'
        local bases as a vector:  u_new = J_new * J_old^{-1} * u_old,
        where J_block = d a_block / d x_global (the block Jacobian).
        The time component u^t is unchanged (the block time coord == global t).

   locate_and_cross():  on call, y[1..3] are in the current block's local coords.
     Returns  0  -> particle still inside the current block's coarse safe
                    region: nothing changed (fast path; the short single-block
                    run never leaves this branch, hence bit-identical).
     Returns  1  -> particle crossed into a neighbouring block; y[1..7] and
                    gstate.pgeodesic[m].cur_map are updated IN PLACE to the
                    new block's local coords + transformed velocity.  The
                    caller must then force a re-fit (the position/level change
                    does this via the existing cf2_fitting trigger).
     Returns -1  -> particle outside the whole multipatch domain, or in the
                    inter-block shadow zone (no block's coarse region covers
                    it): caller marks outbd.
   ============================================================================================ */
/* 3x3 inverse via the adjugate.  Returns 1 on success, 0 if (near) singular.
   r is not touched on failure. */
static int inv3x3(const double a[3][3], double r[3][3])
{
  double det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
             - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
             + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if (!isfinite(det) || fabs(det) < 1.0e-30) return 0;
  const double idet = 1.0/det;
  r[0][0] = (a[1][1]*a[2][2]-a[1][2]*a[2][1])*idet;
  r[0][1] = (a[1][2]*a[2][0]-a[0][2]*a[2][1])*idet;
  r[0][2] = (a[0][2]*a[1][1]-a[0][1]*a[1][2])*idet;
  r[1][0] = (a[1][2]*a[2][0]-a[1][0]*a[2][2])*idet;
  r[1][1] = (a[0][0]*a[2][2]-a[0][2]*a[2][0])*idet;
  r[1][2] = (a[0][2]*a[1][0]-a[0][0]*a[1][2])*idet;
  r[2][0] = (a[1][0]*a[2][1]-a[1][1]*a[2][0])*idet;
  r[2][1] = (a[0][1]*a[2][0]-a[0][0]*a[2][1])*idet;
  r[2][2] = (a[0][0]*a[1][1]-a[0][1]*a[1][0])*idet;
  return 1;
}

static int locate_and_cross(int m, double y[], cGH *cctkGH)
{
  const int ndims = 3;
  const int npoints = 1;
  const int pm = gstate.pgeodesic[m].cur_map;

  /* fast path: still inside the current block's coarse safe region? */
  const LevelData *L0 = &gstate.LD[pm][0];
  if (in_level_dir(y, L0, 0) && in_level_dir(y, L0, 1) && in_level_dir(y, L0, 2))
    return 0;

  /* 1) current block local coords -> global physical coords
        (also yields J_old = d a_old / d x_global, row-major: J[row][col] = dadx_old[row*3+col]).
        llama's Coordinates_LocalToGlobal / _GlobalToLocal interpret the coord & Jacobian
        args as *pointer arrays* (d-major): coords is CCTK_REAL*[ndims] (each pointer ->
        npoints doubles), dadx is CCTK_REAL*[ndims*ndims] (each -> npoints).  Geodesic's
        flat arrays are wired to those pointer arrays below (npoints==1 here). */
  double localv[ndims], xg[ndims], dadx_old[ndims*ndims];
  for (int d = 0; d < ndims; d++) localv[d] = y[d+1];
  CCTK_REAL* localp[ndims];
  for (int d = 0; d < ndims; d++) localp[d] = &localv[d];
  CCTK_REAL* xgp[ndims];
  for (int d = 0; d < ndims; d++) xgp[d] = &xg[d];
  CCTK_REAL* dadxop[ndims*ndims];
  for (int k = 0; k < ndims*ndims; k++) dadxop[k] = &dadx_old[k];
  {
    CCTK_INT patch = pm;
    int rc = MultiPatch_LocalToGlobal(cctkGH, ndims, npoints,
                                      &patch, localp, xgp,
                                      NULL, NULL, dadxop, NULL, NULL, NULL);
    if (rc != 0) return -1;
  }

  /* 2) global -> (block, block-local coords, J_new = d a_new / d x_global
         col-major: J[row][col] = dadx_new[col*ndims+row]) */
  double newy[ndims];
  double dadx_new[ndims*ndims];
  CCTK_REAL* newyp[ndims];
  for (int d = 0; d < ndims; d++) newyp[d] = &newy[d];
  CCTK_REAL* dadxnp[ndims*ndims];
  for (int k = 0; k < ndims*ndims; k++) dadxnp[k] = &dadx_new[k];
  int newmap;
  {
    CCTK_INT patch[1];
    int rc = MultiPatch_GlobalToLocal(cctkGH, ndims, npoints,
                                      xgp, &patch[0], newyp,
                                      dadxnp, NULL);
    if (rc != 0 || patch[0] < 0 || patch[0] >= GEODESIC_MAX_MAPS)
      return -1;   /* unknown point: outside the whole multipatch domain */
    newmap = patch[0];
  }

  /* 3) still the same block but outside its safe region -> shadow-zone point
        (stencil fits in neither block's grid) -> out of domain. */
  if (newmap == pm)
    return -1;

  /* 4) usable by the neighbour?  require the point to lie in the safe
        region (or, if that region is degenerate, the interior box) of SOME
        level of the new block -- exactly the criterion pick_level uses to
        decide whether the block can integrate the point there.  This is
        tighter than the old "coarsest-level interior box" test and stops
        a particle from being handed to a block that has no usable
        stencil at that spot (inter-block shadow zone). */
  {
    double nyp[4];
    nyp[0] = 0.0;
    for (int d = 0; d < ndims; d++) nyp[d+1] = newy[d];
    if (pick_level(nyp, newmap) < 0)
      return -1;   /* no usable level in the new block -> shadow zone / out of domain */
  }

  /* 5) build the two Jacobians. */
  double Jold[3][3], Jnew[3][3], Jinv[3][3];
  for (int row = 0; row < ndims; row++)
    for (int col = 0; col < ndims; col++)
    {
      Jold[row][col] = dadx_old[row*ndims + col];          /* row-major */
      Jnew[row][col] = dadx_new[col*ndims + row];          /* col-major  */
    }

  /* 6) transform the spatial velocity:  u_new = Jnew * inv(Jold) * u_old.
         v = inv(Jold) * u_old  (global physical velocity components).       */
  if (inv3x3(Jold, Jinv) != 1)
    return -1;
  double v[3];
  for (int col = 0; col < ndims; col++) {
    v[col] = 0.0;
    for (int j = 0; j < ndims; j++)
      v[col] += Jinv[col][j] * y[j+5];   /* spatial u = y[5..7] = (u^x,u^y,u^z) */
  }
  for (int row = 0; row < ndims; row++) {
    double u = 0.0;
    for (int col = 0; col < ndims; col++)
      u += Jnew[row][col] * v[col];
    y[row+5] = u;   /* write transformed spatial u into y[5..7] */
  }

  /* 7) adopt the new block.  Force a re-fit in the new block's grid by
         setting the level to a sentinel; the existing cf2_fitting trigger
         (lvl != level) will pick the new block's level and re-sample. */
  for (int d = 0; d < ndims; d++) y[d+1] = newy[d];
  gstate.pgeodesic[m].cur_map = newmap;
  gstate.pgeodesic[m].level    = -1;
  gstate.pgeodesic[m].flag     = 0;
  return 1;
}

/* Derivatives / analytic Kerr-Schild helpers, polynomial matrix MtxA etc. */
#include "derivative.h"

/* ------------------ forward declarations (existing) ------------------ */
CCTK_REAL ndg4dn_tl(CCTK_INT m, CCTK_INT d, CCTK_INT j1, CCTK_INT j2, CCTK_INT tl,
                    CCTK_INT ix, CCTK_INT iy, CCTK_INT iz);

double urand_range(unsigned int *seed, double a, double b);

void cf2_grid(int m, int i1, int i2, int i3);

int func_ode(double t, const double y[], double f[], void *params);

CCTK_REAL cf2_func(const double y[], int m, int j1, int j2, int j3);

int cf2_fitting(int m, const double y[]);

void geodesics_integrate(CCTK_ARGUMENTS);

/* forward declarations of the grid-mode sampler helpers defined
   further below (the time_interp = linear branch of cf2_fitting uses
   them before their definition) */
static void metric_27_sample_tl(int m, int tl, int i4, int i5, int i6, gd_data *out);
static void fit_metric_poly27(int m, const gd_data src[3][3][3],
                              CCTK_REAL dst[4][4][POLY27],
                              CCTK_INT dstc[4][4]);
static void fit_cf2_poly27(const double G[27][4][4][4],
                           CCTK_REAL dst[4][4][4][POLY27],
                           CCTK_INT dstc[4][4][4]);

/* ============================================================================================
   6th-order FD for metric derivatives (your original)
   ============================================================================================ */
CCTK_REAL ndg4dn_tl(CCTK_INT m, CCTK_INT d, CCTK_INT j1, CCTK_INT j2, CCTK_INT tl,
                    CCTK_INT ix, CCTK_INT iy, CCTK_INT iz)
{
  (void)m;
  CCTK_REAL od60, ret;

  ret = 0.0;
  if (d == 0)
  {
    /* time derivative of the metric: second-order backward difference
       over the three stored time levels (level 0 = now, 1 = t-dt,
       2 = t-2dt); 0 for a stationary metric (levels 0 and 2 agree) */
    if (gview.metric_static)
      ret = 0.0;
    else
    {
      const CCTK_REAL g0 = g4dn(0, j1, j2, ix, iy, iz);
      const CCTK_REAL g1 = g4dn(1, j1, j2, ix, iy, iz);
      const CCTK_REAL g2 = g4dn(2, j1, j2, ix, iy, iz);
      ret = (3.0*g0 - 4.0*g1 + g2) * (0.5 / gview.dt);
    }
  } else if (d == 1)
  {
    od60 = 1 / (60 * gview.delta_space[0]);
    ret = (g4dn(tl, j1, j2, ix+3, iy, iz) -  9*g4dn(tl, j1, j2, ix+2, iy, iz) +
        45*g4dn(tl, j1, j2, ix+1, iy, iz) -    g4dn(tl, j1, j2, ix-3, iy, iz) +
         9*g4dn(tl, j1, j2, ix-2, iy, iz) - 45*g4dn(tl, j1, j2, ix-1, iy, iz)) * od60;
  } else if (d == 2)
  {
    od60 = 1 / (60 * gview.delta_space[1]);
    ret = (g4dn(tl, j1, j2, ix, iy+3, iz) -  9*g4dn(tl, j1, j2, ix, iy+2, iz) +
        45*g4dn(tl, j1, j2, ix, iy+1, iz) -    g4dn(tl, j1, j2, ix, iy-3, iz) +
         9*g4dn(tl, j1, j2, ix, iy-2, iz) - 45*g4dn(tl, j1, j2, ix, iy-1, iz)) * od60;
  } else if (d == 3)
  {
    od60 = 1 / (60 * gview.delta_space[2]);
    ret = (g4dn(tl, j1, j2, ix, iy, iz+3) -  9*g4dn(tl, j1, j2, ix, iy, iz+2) +
        45*g4dn(tl, j1, j2, ix, iy, iz+1) -    g4dn(tl, j1, j2, ix, iy, iz-3) +
         9*g4dn(tl, j1, j2, ix, iy, iz-2) - 45*g4dn(tl, j1, j2, ix, iy, iz-1)) * od60;
  }
  return ret;
}

/* ============================================================================================
   Polynomial interpolation evaluation for Christoffel (your original cf2_func)
   ============================================================================================ */
CCTK_REAL cf2_func(const double y[], int m, int j1, int j2, int j3)
{
  CCTK_REAL tx1, tx2;
  CCTK_REAL ty1, ty2;
  CCTK_REAL tz1, tz2;
  CCTK_REAL ret;

  /*
  f = a01 + a02*x + a03*y + a04*z + a05*x^2 + a06*y^2 + a07*z^2 + a08*x*y + a09*x*z + \
       a10*y*z + a11*x^2*y + a12*x^2*z + a13*y^2*x + a14*y^2*z + a15*z^2*x + a16*z^2*y + \
       a17*x*y*z + a18*x^2*y^2 + a19*x^2*z^2 + a20*y^2*z^2 + a21*x^2*y*z + a22*y^2*x*z + \
       a23*z^2*x*y + a24*x^2*y^2*z + a25*x^2*z^2*y + a26*y^2*z^2*x + a27*x^2*y^2*z^2
  */

  if (gstate.pgeodesic[m].pcf2c[j1][j2][j3])
  {
    ret = gstate.pgeodesic[m].pcf2[j1][j2][j3][0];
  }
  else
  {
    tx1 = (y[1] - gstate.pgeodesic[m].below[1])/gview.delta_space[0];
    tx2 = tx1*tx1;
    ty1 = (y[2] - gstate.pgeodesic[m].below[2])/gview.delta_space[1];
    ty2 = ty1*ty1;
    tz1 = (y[3] - gstate.pgeodesic[m].below[3])/gview.delta_space[2];
    tz2 = tz1*tz1;

    ret = gstate.pgeodesic[m].pcf2[j1][j2][j3][ 0] +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 1]*tx1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 2]*ty1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 3]*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 4]*tx2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 5]*ty2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 6]*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 7]*tx1*ty1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 8]*tx1*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][ 9]*ty1*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][10]*tx2*ty1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][11]*tx2*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][12]*tx1*ty2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][13]*ty2*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][14]*tx1*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][15]*ty1*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][16]*tx1*ty1*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][17]*tx2*ty2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][18]*tx2*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][19]*ty2*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][20]*tx2*ty1*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][21]*tx1*ty2*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][22]*tx1*ty1*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][23]*tx2*ty2*tz1 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][24]*tx2*ty1*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][25]*tx1*ty2*tz2 +
          gstate.pgeodesic[m].pcf2[j1][j2][j3][26]*tx2*ty2*tz2;
  }

  return ret;
}

/* ============================================================================================
   Generic 27-term polynomial evaluation helper
   (same basis as cf2_func, but for scalar fields).
   ============================================================================================ */
static inline CCTK_REAL poly27_eval(const double y[], int m,
                                   const CCTK_REAL c[POLY27],
                                   const CCTK_INT is_const)
{
  if (is_const)
    return c[0];

  const CCTK_REAL tx1 = (y[1] - gstate.pgeodesic[m].below[1]) / gview.delta_space[0];
  const CCTK_REAL ty1 = (y[2] - gstate.pgeodesic[m].below[2]) / gview.delta_space[1];
  const CCTK_REAL tz1 = (y[3] - gstate.pgeodesic[m].below[3]) / gview.delta_space[2];

  const CCTK_REAL tx2 = tx1*tx1;
  const CCTK_REAL ty2 = ty1*ty1;
  const CCTK_REAL tz2 = tz1*tz1;

  return
    c[ 0] +
    c[ 1]*tx1 +
    c[ 2]*ty1 +
    c[ 3]*tz1 +
    c[ 4]*tx2 +
    c[ 5]*ty2 +
    c[ 6]*tz2 +
    c[ 7]*tx1*ty1 +
    c[ 8]*tx1*tz1 +
    c[ 9]*ty1*tz1 +
    c[10]*tx2*ty1 +
    c[11]*tx2*tz1 +
    c[12]*tx1*ty2 +
    c[13]*ty2*tz1 +
    c[14]*tx1*tz2 +
    c[15]*ty1*tz2 +
    c[16]*tx1*ty1*tz1 +
    c[17]*tx2*ty2 +
    c[18]*tx2*tz2 +
    c[19]*ty2*tz2 +
    c[20]*tx2*ty1*tz1 +
    c[21]*tx1*ty2*tz1 +
    c[22]*tx1*ty1*tz2 +
    c[23]*tx2*ty2*tz1 +
    c[24]*tx2*ty1*tz2 +
    c[25]*tx1*ty2*tz2 +
    c[26]*tx2*ty2*tz2;
}

/* Poly27 basis exponents: term k = tx^A[k] ty^B[k] tz^C[k], the tensor
   product {0,1,2}^3 in EXACTLY the order of the c[0..26] terms used by
   poly27_eval (monomials: 1, tx, ty, tz, tx^2, ty^2, tz^2, tx ty, ...). */
static const int P27_A[27] = {0,1,0,0,2,0,0,1,1,0,2,2,1,0,1,0,1,2,2,0,2,1,1,2,2,1,2};
static const int P27_B[27] = {0,0,1,0,0,2,0,1,0,1,1,0,2,2,0,1,1,2,0,2,1,2,1,2,1,2,2};
static const int P27_C[27] = {0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 1, 0, 1, 2, 2, 1, 0, 2, 2, 1, 1, 2, 1, 2, 2, 2};

/* value of tx^a ty^b tz^c for a,b,c in {0,1,2} (normalized coords) */
static inline double p27_monom(int a, int b, int c, double tx, double ty, double tz)
{
  double x = 1.0; if (a==1) x = tx; else if (a==2) x = tx*tx;
  double y = 1.0; if (b==1) y = ty; else if (b==2) y = ty*ty;
  double z = 1.0; if (c==1) z = tz; else if (c==2) z = tz*tz;
  return x*y*z;
}

/* Derivatives of the 27-term polynomial w.r.t. the NORMALIZED coords tx,ty,tz.
   dP/dtx etc. (physical derivatives require division by dx,dy,dz at caller). */
static void p27_deriv(double tx, double ty, double tz,
                      const double c[27],
                      double *dtx, double *dty, double *dtz)
{
  *dtx = *dty = *dtz = 0.0;
  for (int k = 0; k < 27; ++k)
  {
    const double v = c[k];
    if (v == 0.0) continue;
    const int a = P27_A[k], b = P27_B[k], cc = P27_C[k];
    if (a >= 1) *dtx += v * (double)a * p27_monom(a-1, b, cc, tx, ty, tz);
    if (b >= 1) *dty += v * (double)b * p27_monom(a, b-1, cc, tx, ty, tz);
    if (cc>= 1) *dtz += v * (double)cc * p27_monom(a, b, cc-1, tx, ty, tz);
  }
}

/* ============================================================================================
   cf2_grid: compute and cache on 3x3x3 stencil:
     - metric g_{mu nu}
     - Christoffel Gamma^mu_{ab}
   ============================================================================================ */
void cf2_grid (int m, int i1, int i2, int i3)
{
  int index, j1, j2, j3, j4, i4, i5, i6;
  CCTK_REAL g4up[4][4];
  CCTK_REAL dgd[4][4][4];
  CCTK_REAL cf1[4][4][4];
  double gxxL, gxyL, gxzL, gyyL, gyzL, gzzL, det, invdet, gupxxL, gupxyL, gupyyL, gupxzL, gupyzL, gupzzL;
  double lapse, lapse_1, lapse_11, shiftx, shifty, shiftz;

  /* local indices 0..2 in stencil */
  i4 = i1;
  i5 = i2;
  i6 = i3;

  /* absolute grid indices (1-based in your code)
   below[i] = origin + (point[i]-1)*dx, so the stencil must start
   at point-1 to match the polynomial basis at normalized coords 0,1,2. */
  i1 = i1 + gstate.pgeodesic[m].point[1] - 1;
  i2 = i2 + gstate.pgeodesic[m].point[2] - 1;
  i3 = i3 + gstate.pgeodesic[m].point[3] - 1;

  index = LD_INDEX(i1-1, i2-1, i3-1);

  /* ---------------- metric from ADMBase 3+1 ---------------- */
  gxxL = gview.gxx[0][index];
  gxyL = gview.gxy[0][index];
  gxzL = gview.gxz[0][index];
  gyyL = gview.gyy[0][index];
  gyzL = gview.gyz[0][index];
  gzzL = gview.gzz[0][index];

  det = -gxzL*gxzL*gyyL + 2*gxyL*gxzL*gyzL - gxxL*gyzL*gyzL - gxyL*gxyL*gzzL + gxxL*gyyL*gzzL;
  invdet = 1.0 / det;
  gupxxL = (-gyzL*gyzL + gyyL*gzzL)*invdet;
  gupxyL = ( gxzL*gyzL - gxyL*gzzL)*invdet;
  gupyyL = (-gxzL*gxzL + gxxL*gzzL)*invdet;
  gupxzL = (-gxzL*gyyL + gxyL*gyzL)*invdet;
  gupyzL = ( gxyL*gxzL - gxxL*gyzL)*invdet;
  gupzzL = (-gxyL*gxyL + gxxL*gyyL)*invdet;

  lapse    = gview.alp[0][index];
  lapse_1  = 1.0/lapse;
  lapse_11 = lapse_1*lapse_1;
  shiftx   = gview.betax[0][index];
  shifty   = gview.betay[0][index];
  shiftz   = gview.betaz[0][index];

  /* g^{mu nu} from 3+1 */
  g4up[0][0] =              -lapse_11;
  g4up[0][1] = g4up[1][0] = lapse_11*shiftx;
  g4up[0][2] = g4up[2][0] = lapse_11*shifty;
  g4up[0][3] = g4up[3][0] = lapse_11*shiftz;
  g4up[1][1] =              gupxxL - lapse_11*shiftx*shiftx;
  g4up[1][2] = g4up[2][1] = gupxyL - lapse_11*shiftx*shifty;
  g4up[1][3] = g4up[3][1] = gupxzL - lapse_11*shiftx*shiftz;
  g4up[2][2] =              gupyyL - lapse_11*shifty*shifty;
  g4up[2][3] = g4up[3][2] = gupyzL - lapse_11*shifty*shiftz;
  g4up[3][3] =              gupzzL - lapse_11*shiftz*shiftz;

  /* g_{mu nu} */
  for(j1=0; j1<4; j1++)
  {
    for(j2=j1; j2<4; j2++)
    {
      if (j1 == 0)
      {
        if (j2 == 0)
        {
          double shift_x, shift_y, shift_z;
          shift_x = (gxxL*(shiftx) + gxyL*(shifty) + gxzL*(shiftz));
          shift_y = (gxyL*(shiftx) + gyyL*(shifty) + gyzL*(shiftz));
          shift_z = (gxzL*(shiftx) + gyzL*(shifty) + gzzL*(shiftz));
          gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = -lapse*lapse + (shiftx*shift_x + shifty*shift_y + shiftz*shift_z);
        }
        else
        {
          if (j2 == 1)
            gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxxL*shiftx + gxyL*shifty + gxzL*shiftz;
          else if (j2 == 2)
            gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxyL*shiftx + gyyL*shifty + gyzL*shiftz;
          else
            gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxzL*shiftx + gyzL*shifty + gzzL*shiftz;
        }
      }
      else
      {
        if (j1 == 1)
        {
          if (j2 == 1)      gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxxL;
          else if (j2 == 2) gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxyL;
          else              gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxzL;
        }
        else if (j1 == 2)
        {
          if (j2 == 2)      gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gyyL;
          else              gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gyzL;
        }
        else
        {
          gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gzzL;
        }
      }
    }
  }

  for(j1=1; j1<4; j1++)
    for(j2=0; j2<j1; j2++)
      gstate.pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gstate.pgeodesic[m].gd[i4][i5][i6].gd[j2][j1];

  /* ---------------- derivatives of metric (your ndg4dn) ---------------- */
  for(j1=0; j1<4; j1++)
    for(j2=0; j2<4; j2++)
      for(j3=j2; j3<4; j3++)
        dgd[j1][j2][j3] = ndg4dn_tl(0, j1, j2, j3, 0, i1, i2, i3);

  for(j1=0; j1<4; j1++)
    for(j2=1; j2<4; j2++)
      for(j3=0; j3<j2; j3++)
        dgd[j1][j2][j3] = dgd[j1][j3][j2];

  /* ---------------- Christoffel ---------------- */
  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<4; j3++)
      {
        cf1[j1][j2][j3] = 0.0;
        gstate.pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] = 0.0;
      }

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=j2; j3<4; j3++)
        cf1[j1][j2][j3] += 0.5*(dgd[j3][j2][j1] + dgd[j2][j1][j3] - dgd[j1][j2][j3]);

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        cf1[j1][j2][j3] = cf1[j1][j3][j2];

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=j2; j3<4; j3++)
        for (j4=0; j4<4; j4++)
          gstate.pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] += g4up[j1][j4] * cf1[j4][j2][j3];

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        gstate.pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] =
          gstate.pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j3][j2];
}

/* ============================================================================================
   pointwise_cf2_tl: Christoffel symbols (2nd kind) at ONE grid point
   (absolute 1-based indices i1,i2,i3) from a single metric time level
   tl (0 = t, 1 = t - dt), for the time_interp = linear path.
   Operation-identical to the cf2_grid core (3+1 inverse from the level
   fields, 6th-order FD spatial metric derivatives at the same level)
   except:
     - the metric time level is tl (cf2_grid hard-codes level 0),
     - the metric time derivative is the 2-level difference
       (g0 - g1)/dt (cf2_grid uses 0 for a stationary metric and a
       3-level backward difference otherwise),
     - the result goes to the caller's array (cf2_grid caches in cf2g).
   For a stationary metric (levels 0 and 1 bit-identical) this is
   bit-identical to cf2_grid at level 0.
   ============================================================================================ */
static void pointwise_cf2_tl(int m, CCTK_INT tl, CCTK_INT i1, CCTK_INT i2, CCTK_INT i3,
                             double cf2out[4][4][4])
{
  double g4up[4][4];
  double dgd[4][4][4];
  double cf1[4][4][4];
  double gxxL, gxyL, gxzL, gyyL, gyzL, gzzL, det, invdet;
  double gupxxL, gupxyL, gupyyL, gupxzL, gupyzL, gupzzL;
  double lapse, lapse_1, lapse_11, shiftx, shifty, shiftz;
  int index, j1, j2, j3, j4;
  (void)m;

  index = LD_INDEX(i1-1, i2-1, i3-1);

  gxxL = gview.gxx[tl][index];  gxyL = gview.gxy[tl][index];  gxzL = gview.gxz[tl][index];
  gyyL = gview.gyy[tl][index];  gyzL = gview.gyz[tl][index];  gzzL = gview.gzz[tl][index];

  det = -gxzL*gxzL*gyyL + 2*gxyL*gxzL*gyzL - gxxL*gyzL*gyzL - gxyL*gxyL*gzzL + gxxL*gyyL*gzzL;
  invdet = 1.0 / det;
  gupxxL = (-gyzL*gyzL + gyyL*gzzL)*invdet;
  gupxyL = ( gxzL*gyzL - gxyL*gzzL)*invdet;
  gupyyL = (-gxzL*gxzL + gxxL*gzzL)*invdet;
  gupxzL = (-gxzL*gyyL + gxyL*gyzL)*invdet;
  gupyzL = ( gxyL*gxzL - gxxL*gyzL)*invdet;
  gupzzL = (-gxyL*gxyL + gxxL*gyyL)*invdet;

  lapse    = gview.alp[tl][index];
  lapse_1  = 1.0/lapse;
  lapse_11 = lapse_1*lapse_1;
  shiftx   = gview.betax[tl][index];
  shifty   = gview.betay[tl][index];
  shiftz   = gview.betaz[tl][index];

  g4up[0][0] =              -lapse_11;
  g4up[0][1] = g4up[1][0] = lapse_11*shiftx;
  g4up[0][2] = g4up[2][0] = lapse_11*shifty;
  g4up[0][3] = g4up[3][0] = lapse_11*shiftz;
  g4up[1][1] =              gupxxL - lapse_11*shiftx*shiftx;
  g4up[1][2] = g4up[2][1] = gupxyL - lapse_11*shiftx*shifty;
  g4up[1][3] = g4up[3][1] = gupxzL - lapse_11*shiftx*shiftz;
  g4up[2][2] =              gupyyL - lapse_11*shifty*shifty;
  g4up[2][3] = g4up[3][2] = gupyzL - lapse_11*shifty*shiftz;
  g4up[3][3] =              gupzzL - lapse_11*shiftz*shiftz;

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=j2; j3<4; j3++)
      {
        if (j1 == 0)
          dgd[j1][j2][j3] = (g4dn(0, j2, j3, i1, i2, i3)
                             - g4dn(1, j2, j3, i1, i2, i3)) / gview.dt;
        else
          dgd[j1][j2][j3] = ndg4dn_tl(m, j1, j2, j3, tl, i1, i2, i3);
      }
  for (j1=0; j1<4; j1++)
    for (j2=1; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        dgd[j1][j2][j3] = dgd[j1][j3][j2];

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<4; j3++)
      {
        cf1[j1][j2][j3] = 0.0;
        cf2out[j1][j2][j3] = 0.0;
      }
  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=j2; j3<4; j3++)
        cf1[j1][j2][j3] += 0.5*(dgd[j3][j2][j1] + dgd[j2][j1][j3] - dgd[j1][j2][j3]);
  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        cf1[j1][j2][j3] = cf1[j1][j3][j2];
  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=j2; j3<4; j3++)
        for (j4=0; j4<4; j4++)
          cf2out[j1][j2][j3] += g4up[j1][j4] * cf1[j4][j2][j3];
  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        cf2out[j1][j2][j3] = cf2out[j1][j3][j2];
}

/*
  Calculate interpolation coefficients
  int m            Particle index
  const double y[] Particle current state vector (t,x,y,z, u^t,u^x,u^y,u^z)
*/
int cf2_fitting (int m, const double y[])
{
  CCTK_INT i, j1, j2, j3, j4, j5, j6, j7, j8;
  const int pm = gstate.pgeodesic[m].cur_map;   /* owning block */
  gd_data gd1buf[3][3][3]; /* time-level-1 metric samples (local27+linear) */

  /* threshold for "constant field" detection */
  const double rel_eps = 1e-8;

  /* ---------------- AMR: pick the finest level whose safe region contains y ----------------
     The 7-cell safe margin of the chosen level guarantees that the
     3x3x3 stencil plus the 6th-order FD padding fits inside that
     level's grid (see the safe-region definition in the acquire phase).
     In the sexact branch the metric/Christoffel symbols are exact
     analytic values, but the level is still chosen by position (as in
     the !sexact branch; always 0 for a single-level, non-AMR run).
     Multi-process Exact runs (multi_exact): each rank owns only its own
     local patch, so a level whose geometry was never acquired on this
     rank must not be considered; pin to level 0 (the only level whose
     geometry is guaranteed present on every rank).  Multi-process AMR
     (multi_amr): the particle is pinned to its owning level (the one
     whose local patch contains it; the Phase 1 owner map), because the
     other ranks' patches are not visible on this rank. */
  const int lvl = gstate.multi_amr ? gstate.owner_lvl[m]
                     : (gstate.multi_exact ? 0 : pick_level(y, pm));
  if (lvl < 0)
  {
    gstate.pgeodesic[m].outbd = 1;
    {
      static int dbg_outbd = -1, n_dbg_outbd = 0;
      if (dbg_outbd < 0) dbg_outbd = (getenv("GEODESIC_DBG_OUTBD") != NULL);
      if (dbg_outbd && n_dbg_outbd < 40)
      {
        n_dbg_outbd++;
        fprintf(stderr, "[OUTBD] m=%d y=(%.6f %.6f %.6f) pm=%d lvl=%d samr=%d ",
                m, y[1], y[2], y[3], pm, lvl, gstate.samr_levels);
        for (int l = 0; l < gstate.samr_levels; l++)
        {
          const LevelData *L = &gstate.LD[pm][l];
          fprintf(stderr, "  l=%d valid=%d ", l, (int)L->valid);
          if (L->valid) {
            int in0 = in_level_dir(y, L, 0);
            int in1 = in_level_dir(y, L, 1);
            int in2 = in_level_dir(y, L, 2);
            double lo[3], hi[3];
            const int margin = safe_margin_cells();
            for (int d = 0; d < 3; d++) {
              lo[d] = L->origin[d] + (CCTK_REAL)margin * L->dx[d];
              hi[d] = L->origin[d] + (CCTK_REAL)(L->lsh[d]-1-margin) * L->dx[d];
              if (lo[d] > hi[d]) { lo[d] = L->i_lbnd[d]; hi[d] = L->i_ubnd[d]; }
            }
            fprintf(stderr, "in=(%d %d %d) safe=[%.6g %.6g][%.6g %.6g][%.6g %.6g] ",
                    in0, in1, in2, lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
          }
        }
        fprintf(stderr, "\n");
      }
    }
    return 0;
  }
  /* make sure this thread's grid view matches the particle's (map, level)
     (self-guarded; also covers the no-rebuild path when this thread
     previously worked on a different particle/map/level) */
  switch_to_level(pm, lvl);

  /* compute base grid point index for the current particle position (on level lvl) */
  CCTK_INT tpoint[4];
  tpoint[0] = 0;
  for (i=1; i<4; i++)
  {
    tpoint[i] = lrint (floor ((y[i] - gstate.LD[pm][lvl].origin[i-1]) * gstate.LD[pm][lvl].inv_dx[i-1] + 0.5));
  }

  /* multi_amr: clip the index into the owner's stencil-safe range.
     The owner box [i_lbnd, i_ubnd) = [origin+(ng-0.5)dx, origin+(lsh-ng-0.5)dx)
     maps to tpoint in [4, lsh-4], but every grid read here goes through
     g4dn(), which accepts only valid cells (1-based ix in [1, lsh]) and
     the full read window is [tp-4, tp+4] (the 3x3x3 cf2_grid stencil plus
     the 6th-order FD padding) -- so the safe range is [5, lsh-4], one
     cell inside the box's first boundary face.  A particle sitting in that
     face cell (e.g. a z=0 particle on the z+ rank, the shared boundary of
     the 2x2 decomposition) would map to tp=4 and read ix=0 out of bounds.
     Clipping tp to [5, lsh-4] keeps every read (including the direct
     LD_INDEX reads in cf2_grid) inside valid cells.  The polynomial is
     still evaluated at the true y: for a boundary-face particle the local
     coordinate reaches -0.5, a half-cell extrapolation of the quartic
     surface fit that is negligible for a smooth metric; for interior
     particles the clip never fires. */
  if (gstate.multi_amr && !gstate.sexact)
  {
    for (i=1; i<4; i++)
    {
      if (tpoint[i] < 5)
        tpoint[i] = 5;
      if (tpoint[i] > gstate.LD[pm][lvl].lsh[i-1] - 4)
        tpoint[i] = gstate.LD[pm][lvl].lsh[i-1] - 4;
    }
  }

  /* ------------------------------------------------------------------
     need_rebuild == true  <=>  the particle changed refinement level,
     or moved into a new 3x3x3 stencil cube on the same level, or this
     is the first call for it in the current Cactus step (flag == 0).
     Only in this case do we resample cf2_grid and refit the 27-term
     polynomials.
     ------------------------------------------------------------------ */
  const int need_rebuild = (!gstate.pgeodesic[m].flag ||
                             lvl != gstate.pgeodesic[m].level ||
                             tpoint[1] != gstate.pgeodesic[m].point[1] ||
                             tpoint[2] != gstate.pgeodesic[m].point[2] ||
                             tpoint[3] != gstate.pgeodesic[m].point[3]);

  if (need_rebuild)
  {
    gstate.pgeodesic[m].level    = lvl;
    gstate.pgeodesic[m].point[1] = tpoint[1];
    gstate.pgeodesic[m].point[2] = tpoint[2];
    gstate.pgeodesic[m].point[3] = tpoint[3];

    for (i=1; i<4; i++)
    {
      gstate.pgeodesic[m].below[i] = gstate.LD[pm][lvl].origin[i-1] + (tpoint[i] - 1) * gstate.LD[pm][lvl].dx[i-1];
      gstate.pgeodesic[m].dxp[i-1] = gstate.LD[pm][lvl].dx[i-1];
    }

    /* !sexact: sample metric + Christoffel symbols on the stencil.
       In the sexact branch both are exact Kerr-Schild analytic values,
       evaluated directly by tg4dn/gdn_derivatives inside func_ode, so
       no sampling is needed. */
    if (!gstate.sexact && gstate.sinterp_mode == 0)
    {
      if (!gstate.stime_interp)
      {
        /* original path: metric + Christoffel symbols on the stencil
           (level 0 / current time).  Also sample the OLDER (t-dt) level
           into gd1buf: the robust (blend-then-inverse) path needs the
           1st-order time derivative (g0 - g1)/dt, which requires both
           time levels.  The historical off path never read the old level,
           so pgd1 stayed all-zero and the (g0 - g1)/dt term was spurious
           (the l27 particle-drop bug). */
        for (j1=0; j1<3; j1++)
          for (j2=0; j2<3; j2++)
            for (j3=0; j3<3; j3++)
            {
              cf2_grid(m, j1, j2, j3);
              metric_27_sample_tl(m, 1, j1, j2, j3, &gd1buf[j1][j2][j3]);
            }
        if (gstate.smoke_eps != 0.0)
        {
          const double s1 = 1.0 - gstate.smoke_eps;
          for (j1=0; j1<3; j1++)
            for (j2=0; j2<3; j2++)
              for (j3=0; j3<3; j3++)
                for (i=0; i<4; i++)
                  for (j4=0; j4<4; j4++)
                    gd1buf[j1][j2][j3].gd[i][j4] *= s1;
        }
        /* old-level validity: all-zero or non-finite stencil samples mean
           the past time level is not filled at this AMR cell (first
           integration, or a newly-grown refinement region).  Flag it so
           the robust path pins theta to 1 (current level).  Same detection
           as the linear branch. */
        {
          int old_bad = 0;
          double maxabs = 0.0;
          for (j1=0; j1<3 && !old_bad; j1++)
            for (j2=0; j2<3 && !old_bad; j2++)
              for (j3=0; j3<3 && !old_bad; j3++)
                for (i=0; i<4; i++)
                  for (j4=0; j4<4; j4++)
                  {
                    double v = gd1buf[j1][j2][j3].gd[i][j4];
                    if (!isfinite(v))
                      old_bad = 1;
                    else if (fabs(v) > maxabs)
                      maxabs = fabs(v);
                  }
          if (!old_bad && maxabs < 1e-100)
            old_bad = 1;
          gstate.pgeodesic[m].old_level_bad = old_bad ? 1 : 0;
        }
      }
      else
      {
        /* local27+linear: 4-metric only, both time levels
           (time level 0 -> pgeodesic[m].gd, time level 1 -> gd1buf) */
        for (j1=0; j1<3; j1++)
          for (j2=0; j2<3; j2++)
            for (j3=0; j3<3; j3++)
            {
              metric_27_sample_tl(m, 0, j1, j2, j3, &gstate.pgeodesic[m].gd[j1][j2][j3]);
              metric_27_sample_tl(m, 1, j1, j2, j3, &gd1buf[j1][j2][j3]);
            }
        if (gstate.smoke_eps != 0.0)
        {
          const double s1 = 1.0 - gstate.smoke_eps;
          for (j1=0; j1<3; j1++)
            for (j2=0; j2<3; j2++)
              for (j3=0; j3<3; j3++)
                for (i=0; i<4; i++)
                  for (j4=0; j4<4; j4++)
                    gd1buf[j1][j2][j3].gd[i][j4] *= s1;
        }
        /*
         * Detect an invalid OLDER time level: if every one of the 27
         * stencil cells holds an all-zero (or non-finite) 4-metric on
         * level 1, the past time level is not filled at this AMR cell
         * (typical at the very first integration, when the particle's
         * coordinate time lags the grid, or when the refinement region
         * has just grown over the particle).  Flag it so the linear
         * blend below (and christoffel_at_point) can fall back to the
         * current level. */
        {
          int old_bad = 0;
          double maxabs = 0.0;
          for (j1=0; j1<3 && !old_bad; j1++)
            for (j2=0; j2<3 && !old_bad; j2++)
              for (j3=0; j3<3 && !old_bad; j3++)
                for (i=0; i<4; i++)
                  for (j4=0; j4<4; j4++)
                  {
                    double v = gd1buf[j1][j2][j3].gd[i][j4];
                    if (!isfinite(v))
                      old_bad = 1;
                    else if (fabs(v) > maxabs)
                      maxabs = fabs(v);
                  }
          if (!old_bad && maxabs < 1e-100)
            old_bad = 1;
          gstate.pgeodesic[m].old_level_bad = old_bad ? 1 : 0;
        }
        /* [REBUILD_DBG] env GEODESIC_DBG_REBUILD: dump the 27x16 stencil
           samples (current level pgeodesic[m].gd + old level gd1buf) with
           their non-finite counts and magnitudes, to locate whether the
           nan enters through grid sampling (current/old level) at this
           level/patch or only appears in the fits. */
        if (getenv("GEODESIC_DBG_REBUILD") != NULL && m < 20)
        {
          int cur_bad = 0, obad = 0;
          double cmax = 0.0, omax = 0.0;
          for (j1=0; j1<3; j1++)
            for (j2=0; j2<3; j2++)
              for (j3=0; j3<3; j3++)
                for (i=0; i<4; i++)
                  for (j4=0; j4<4; j4++)
                  {
                    double vc = gstate.pgeodesic[m].gd[j1][j2][j3].gd[i][j4];
                    double vo = gd1buf[j1][j2][j3].gd[i][j4];
                    if (!isfinite(vc)) cur_bad++;
                    if (!isfinite(vo)) obad++;
                    if (isfinite(vc) && fabs(vc) > cmax) cmax = fabs(vc);
                    if (isfinite(vo) && fabs(vo) > omax) omax = fabs(vo);
                  }
          fprintf(stderr,
            "[REBUILD_DBG] m=%d lvl=%d t=(%d,%d,%d) outbd=%d cur_bad=%d "
            "cur_maxabs=%.6g old_bad=%d old_maxabs=%.6g\n",
            m, (int)lvl, tpoint[1], tpoint[2], tpoint[3],
            (int)gstate.pgeodesic[m].outbd, cur_bad, cmax, obad, omax);
        }
      }
    }
    /* sinterp_mode 1/2 (aei/carpet): no stencil sampling (pointwise) */
  }

  if (!need_rebuild)
  {
    /* Stencil unchanged: the pcf2/pgd coefficients are still valid in
       the !sexact branch -- reuse them and return. */
    return 1;
  }

  /* ---------------------------------------------
     The block below runs only when the stencil was just rebuilt
     (i.e. need_rebuild == true).
     --------------------------------------------- */
  double ly[27];   /* 27 stencil values, shared by the Christoffel/metric fits */

  /* ---------------------------------------------
     27-term polynomial fits for the Christoffel symbols and the metric:
     executed only in the !sexact branch (in the sexact branch both are
     exact Kerr-Schild analytic values, evaluated directly by
     tg4dn/gdn_derivatives inside func_ode).
     --------------------------------------------- */
  if (!gstate.sexact)
  {
  /* the legacy local27+off path fits Christoffel + metric;
     local27+linear fits only the metric (both time levels);
     aei/carpet fit nothing (pointwise interpolation) */
  if (gstate.sinterp_mode == 0 && !gstate.stime_interp)
  {
  /* ---------------------------------------------
     Polynomial fitting: Christoffel coefficients
     --------------------------------------------- */
  for (j1=0; j1<4; j1++)
  {
    for (j2=0; j2<4; j2++)
    {
      for (j3=j2; j3<4; j3++)
      {
        j8 = 0;
        double tmpy  = 0.0;
        double tol   = 0.0;
        int    vary  = 0;

        for (j5=0; j5<3; j5++)
        for (j6=0; j6<3; j6++)
        for (j7=0; j7<3; j7++)
        {
          ly[j8] = gstate.pgeodesic[m].cf2g[j7][j6][j5].cf2[j1][j2][j3];

          if (j8 == 0)
          {
            tmpy = ly[0];
            tol  = fabs(tmpy)*rel_eps + rel_eps;
            vary = 0;
          }
          else if (!vary)
          {
            if (ly[j8] > tmpy + tol || ly[j8] < tmpy - tol)
              vary = 1;
          }
          j8++;
        }

        if (vary)
        {
          for (j4=0; j4<lN; j4++)
          {
            gstate.pgeodesic[m].pcf2[j1][j2][j3][j4] = 0.0;
            for (j5=0; j5<lN; j5++)
              gstate.pgeodesic[m].pcf2[j1][j2][j3][j4] += MtxA[j4][j5]*ly[j5];
          }
          gstate.pgeodesic[m].pcf2c[j1][j2][j3] = 0;
        }
        else
        {
          for (j4=1; j4<lN; j4++)
            gstate.pgeodesic[m].pcf2[j1][j2][j3][j4] = 0.0;

          gstate.pgeodesic[m].pcf2[j1][j2][j3][0] = tmpy;
          gstate.pgeodesic[m].pcf2c[j1][j2][j3] = 1;
        }
      }
    }
  }

  /* enforce symmetry in lower indices for Christoffel: Gamma^mu_{ab} = Gamma^mu_{ba} */
  for (j1=0; j1<4; j1++)
  {
    for (j2=1; j2<4; j2++)
    {
      for (j3=0; j3<j2; j3++)
      {
        for (j4=0; j4<lN; j4++)
          gstate.pgeodesic[m].pcf2[j1][j2][j3][j4] = gstate.pgeodesic[m].pcf2[j1][j3][j2][j4];
        gstate.pgeodesic[m].pcf2c[j1][j2][j3] = gstate.pgeodesic[m].pcf2c[j1][j3][j2];
      }
    }
  }

  /* ---------------------------------------------
     Polynomial fitting: metric coefficients g_{mu nu}
     --------------------------------------------- */
  for (j1=0; j1<4; j1++)
  {
    for (j2=j1; j2<4; j2++)
    {
      j8 = 0;
      double tmpy  = 0.0;
      double tol   = 0.0;
      int    vary  = 0;

      for (j5=0; j5<3; j5++)
      for (j6=0; j6<3; j6++)
      for (j7=0; j7<3; j7++)
      {
        ly[j8] = gstate.pgeodesic[m].gd[j7][j6][j5].gd[j1][j2];

        if (j8 == 0)
        {
          tmpy = ly[0];
          tol  = fabs(tmpy)*rel_eps + rel_eps;
          vary = 0;
        }
        else if (!vary)
        {
          if (ly[j8] > tmpy + tol || ly[j8] < tmpy - tol)
            vary = 1;
        }

        j8++;
      }

      if (vary)
      {
        for (j4=0; j4<lN; j4++)
        {
          gstate.pgeodesic[m].pgd[j1][j2][j4] = 0.0;
          for (j5=0; j5<lN; j5++)
            gstate.pgeodesic[m].pgd[j1][j2][j4] += MtxA[j4][j5]*ly[j5];
        }
        gstate.pgeodesic[m].pgdc[j1][j2] = 0;
      }
      else
      {
        for (j4=1; j4<lN; j4++)
          gstate.pgeodesic[m].pgd[j1][j2][j4] = 0.0;

        gstate.pgeodesic[m].pgd[j1][j2][0] = tmpy;
        gstate.pgeodesic[m].pgdc[j1][j2] = 1;
      }
    }
  }

  /* metric symmetry */
  for (j1=1; j1<4; j1++)
  {
    for (j2=0; j2<j1; j2++)
    {
      for (j4=0; j4<lN; j4++)
        gstate.pgeodesic[m].pgd[j1][j2][j4] = gstate.pgeodesic[m].pgd[j2][j1][j4];
      gstate.pgeodesic[m].pgdc[j1][j2] = gstate.pgeodesic[m].pgdc[j2][j1];
    }
  }

  /* old level (t - dt): fit its metric from the gd1buf stencil samples.
     Operation-identical to the current-level pgd fit (same constant-field
     shortcut).  Used only by the robust (blend-then-inverse) path, which
     needs the 1st-order time derivative (g0 - g1)/dt.  When
     adaptive_interp = off or the particle never enters the strong-curvature
     region this array is simply unused. */
  fit_metric_poly27(m, gd1buf,
                    gstate.pgeodesic[m].pgd1, gstate.pgeodesic[m].pgd1c);
  } /* end local27+off: Christoffel/metric fitting */
  else if (gstate.sinterp_mode == 0)
  {
    /* local27+linear: fit the 4-metric on both time levels (used by
       metric_at_point), and fit the pointwise Christoffel symbols at
       the 27 stencil cells on each time level into pcf2 (t) and
       pcf2_1 (t - dt); the particle Christoffel is then the linear
       time blend of the two polynomial evaluations (see
       christoffel_at_point) */
    fit_metric_poly27(m, gstate.pgeodesic[m].gd,
                      gstate.pgeodesic[m].pgd, gstate.pgeodesic[m].pgdc);
    fit_metric_poly27(m, gd1buf,
                      gstate.pgeodesic[m].pgd1, gstate.pgeodesic[m].pgd1c);
    {
      double G0[27][4][4][4], G1[27][4][4][4];
      CCTK_INT i1, i2, i3;
      int cc = 0;
      for (j5=0; j5<3; j5++)
      for (j6=0; j6<3; j6++)
      for (j7=0; j7<3; j7++)
      {
        i1 = j5+gstate.pgeodesic[m].point[1]-1;
        i2 = j6+gstate.pgeodesic[m].point[2]-1;
        i3 = j7+gstate.pgeodesic[m].point[3]-1;
        pointwise_cf2_tl(m, 0, i1, i2, i3, G0[cc]);
        pointwise_cf2_tl(m, 1, i1, i2, i3, G1[cc]);
        /* smoke hook: same perturbation of the OLD level (a direct scaling
           of the Christoffel samples, not a recomputation from the scaled
           metric -- a perturbation to exercise the blend, not physics) */
        if (gstate.smoke_eps != 0.0)
        {
          const double s1 = 1.0 - gstate.smoke_eps;
          for (j1=0; j1<4; j1++)
            for (j2=0; j2<4; j2++)
              for (j3=0; j3<4; j3++)
                G1[cc][j1][j2][j3] *= s1;
        }
        cc++;
      }
      fit_cf2_poly27(G0, gstate.pgeodesic[m].pcf2, gstate.pgeodesic[m].pcf2c);
      fit_cf2_poly27(G1, gstate.pgeodesic[m].pcf2_1, gstate.pgeodesic[m].pcf2_1c);
    }
  }
  } /* end if (!gstate.sexact): Christoffel/metric fitting */

  /* temporary debug: fit sanity (gated by GEODESIC_DEBUG_FIT, particle 0 only) */
  {
    static int dbg_fit = -1, n_dbg_fit = 0;
    if (dbg_fit < 0) dbg_fit = (getenv("GEODESIC_DEBUG_FIT") != NULL);
    if (dbg_fit && m == 0 && n_dbg_fit < 5 && !gstate.sexact)
    {
      n_dbg_fit++;
      int a, b, c2, d2;
      double vmin = 1e300, vmax = -1e300, pmin = 1e300, pmax = -1e300;
      int ncf = 0, npl = 0;
      for (a=0;a<4;a++) for (b=0;b<4;b++) for (c2=0;c2<4;c2++)
        for (d2=0; d2<27; d2++)
        {
          const double v = (double)gstate.pgeodesic[m].cf2g[d2/9][(d2/3)%3][d2%3].cf2[a][b][c2];
          if (!isfinite(v))
          {
            ncf++;
            if (ncf<=4)
              fprintf(stderr, "FIT_DEBUG m=%d cf2g NONFINITE comp=(%d,%d,%d) node=(%d,%d,%d) v=%.17g\n",
                      m, a,b,c2, d2%3,(d2/3)%3,d2/9, v);
          }
          else if (v<vmin) vmin=v; else if (v>vmax) vmax=v;
        }
      for (a=0;a<4;a++) for (b=0;b<4;b++) for (c2=0;c2<4;c2++)
        for (d2=0; d2<27; d2++)
        {
          const double v = (double)gstate.pgeodesic[m].pcf2[a][b][c2][d2];
          if (!isfinite(v))
          {
            npl++;
            if (npl<=4)
              fprintf(stderr, "FIT_DEBUG m=%d pcf2 NONFINITE comp=(%d,%d,%d) k=%d v=%.17g\n",
                      m, a,b,c2, d2, v);
          }
          else if (v<pmin) pmin=v; else if (v>pmax) pmax=v;
        }
      {
        const int cx = gstate.pgeodesic[m].point[1];
        const int cy = gstate.pgeodesic[m].point[2];
        const int cz = gstate.pgeodesic[m].point[3];
        fprintf(stderr, "FIT_DEBUG m=%d n=%d lvl=%d static=%d dt=%.17g point=(%d,%d,%d) mode=%d timeinterp=%d\n",
                m, n_dbg_fit, (int)gstate.pgeodesic[m].level, (int)gview.metric_static,
                (double)gview.dt, cx, cy, cz, (int)gstate.sinterp_mode, (int)gstate.stime_interp);
        fprintf(stderr, "FIT_DEBUG g4dn gxx tl0=%.17g tl1=%.17g tl2=%.17g\n",
                (double)g4dn(0,1,1,cx,cy,cz), (double)g4dn(1,1,1,cx,cy,cz), (double)g4dn(2,1,1,cx,cy,cz));
        fprintf(stderr, "FIT_DEBUG g4dn gtt tl0=%.17g tl1=%.17g tl2=%.17g\n",
                (double)g4dn(0,0,0,cx,cy,cz), (double)g4dn(1,0,0,cx,cy,cz), (double)g4dn(2,0,0,cx,cy,cz));
        fprintf(stderr, "FIT_DEBUG cf2g finite: min=%.17g max=%.17g nonfinite=%d\n", vmin, vmax, ncf);
        fprintf(stderr, "FIT_DEBUG pcf2 finite: min=%.17g max=%.17g nonfinite=%d\n", pmin, pmax, npl);
      }
    }
  }

  /* temporary debug: Kerr-Schild index/origin cross-check, gated by
     GEODESIC_KSCHECK.  For a subset of particles (m % 4 == 0) print the
     current-level origin/dx/lsh, the 1-based center node, the particle
     position, and the full spatial metric read via g4dn at that node.
     The closed-form Kerr-Schild (a=0) values at the same node are compared
     in scratch/ks_check.py to expose any index/origin offset.  DELETE ME. */
  {
    static int ksc_dbg = -1;
    if (ksc_dbg < 0) ksc_dbg = (getenv("GEODESIC_KSCHECK") != NULL);
    if (ksc_dbg)
    {
      const int cx = gstate.pgeodesic[m].point[1];
      const int cy = gstate.pgeodesic[m].point[2];
      const int cz = gstate.pgeodesic[m].point[3];
      if (m % 4 == 0)
      {
        /* pos = origin + (point-1)*dx is what g4dn reads; the conversion is
           exactly this, so print origin/dx/point and let ks_check.py form
           the physical position.  (particle position comes from the dump.) */
        fprintf(stderr, "KSCHECK m=%d lvl=%d origin=(%g %g %g) dx=(%g %g %g) "
                "lsh=(%d %d %d) point=(%d %d %d)\n",
                m, (int)gstate.pgeodesic[m].level,
                (double)gview.origin_space[0], (double)gview.origin_space[1], (double)gview.origin_space[2],
                (double)gview.delta_space[0], (double)gview.delta_space[1], (double)gview.delta_space[2],
                gview.lsh[0], gview.lsh[1], gview.lsh[2],
                cx, cy, cz);
        fprintf(stderr, "KSCHECK m=%d gxx=%.17g gxy=%.17g gxz=%.17g gyy=%.17g "
                "gyz=%.17g gzz=%.17g\n",
                m,
                (double)g4dn(0,1,1,cx,cy,cz), (double)g4dn(0,1,2,cx,cy,cz),
                (double)g4dn(0,1,3,cx,cy,cz), (double)g4dn(0,2,2,cx,cy,cz),
                (double)g4dn(0,2,3,cx,cy,cz), (double)g4dn(0,3,3,cx,cy,cz));
      }
    }
  }

  {
    /* temporary debug: 3x3x3 metric-neighborhood cross-check, gated by
       GEODESIC_DEBUG_NBG, for the first 16 particles.  Prints gxx(tl=0) on a
       3x3x3 stencil about the particle's current node so we can see whether
       a NaN is localised to one grid node (data corruption at that node) or
       is an artefact of the read.  DELETE ME. */
    static int nbg_dbg = -1;
    if (nbg_dbg < 0) nbg_dbg = (getenv("GEODESIC_DEBUG_NBG") != NULL);
    if (nbg_dbg && m < 16)
    {
      const int cx = gstate.pgeodesic[m].point[1];
      const int cy = gstate.pgeodesic[m].point[2];
      const int cz = gstate.pgeodesic[m].point[3];
      fprintf(stderr, "NBG m=%d lvl=%d point=(%d %d %d) centre gxx(tl0)=%.17g\n",
              m, (int)gstate.pgeodesic[m].level, cx, cy, cz,
              (double)g4dn(0,1,1,cx,cy,cz));
      for (int di=-1; di<=1; di++)
        for (int dj=-1; dj<=1; dj++)
          for (int dk=-1; dk<=1; dk++)
            fprintf(stderr, "NBG   d( %d %d %d) gxx(tl0)=%.17g\n",
                    di, dj, dk, (double)g4dn(0,1,1,cx+di,cy+dj,cz+dk));
    }
  }

  gstate.pgeodesic[m].flag = 1;
  return 1;
}

/* ============================================================================================
   func_ode: geodesic equation in proper time tau:
   dy^mu/dtau = u^mu,   du^mu/dtau = -Gamma^mu_{ab} u^a u^b
   ============================================================================================ */

/* ============================================================================================
   Helpers: build 3+1 quantities and g^{mu nu} from interpolated 4-metric g_{mu nu}
   We avoid full 4x4 inversion; only invert spatial 3x3 metric gamma_{ij}.
   ============================================================================================ */
static int invert_spatial_metric_3x3(const double gxx, const double gxy, const double gxz,
                                     const double gyy, const double gyz, const double gzz,
                                     double gu[3][3], double *det_out)
{
  /* det = gxx*(gyy*gzz-gyz*gyz) - gxy*(gxy*gzz-gxz*gyz) + gxz*(gxy*gyz-gxz*gyy) */
  const double det = gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gxz*gyz) + gxz*(gxy*gyz - gxz*gyy);
  if (det_out) *det_out = det;
  if (!isfinite(det) || fabs(det) < 1e-300) return 0;

  const double invdet = 1.0/det;

  gu[0][0] = (gyy*gzz - gyz*gyz) * invdet;
  gu[0][1] = (gxz*gyz - gxy*gzz) * invdet;
  gu[0][2] = (gxy*gyz - gxz*gyy) * invdet;

  gu[1][0] = gu[0][1];
  gu[1][1] = (gxx*gzz - gxz*gxz) * invdet;
  gu[1][2] = (gxy*gxz - gxx*gyz) * invdet;

  gu[2][0] = gu[0][2];
  gu[2][1] = gu[1][2];
  gu[2][2] = (gxx*gyy - gxy*gxy) * invdet;

  return 1;
}

static int build_gup_from_gdn(const double gdn[4][4],
                             double *alpha_out,
                             double beta_up[3],
                             double gamma_dn[3][3],
                             double gamma_up[3][3],
                             double gup[4][4])
{
  /* gamma_ij = g_{ij} */
  gamma_dn[0][0] = gdn[1][1]; gamma_dn[0][1] = gdn[1][2]; gamma_dn[0][2] = gdn[1][3];
  gamma_dn[1][0] = gdn[2][1]; gamma_dn[1][1] = gdn[2][2]; gamma_dn[1][2] = gdn[2][3];
  gamma_dn[2][0] = gdn[3][1]; gamma_dn[2][1] = gdn[3][2]; gamma_dn[2][2] = gdn[3][3];

  double det;
  if (!invert_spatial_metric_3x3(gamma_dn[0][0],gamma_dn[0][1],gamma_dn[0][2],
                                 gamma_dn[1][1],gamma_dn[1][2],gamma_dn[2][2],
                                 gamma_up,&det))
    return 0;

  /* beta_i = g_{0i} */
  const double beta_dn[3] = { gdn[0][1], gdn[0][2], gdn[0][3] };

  /* beta^i = gamma^{ij} beta_j */
  for (int i=0;i<3;i++)
  {
    beta_up[i] = 0.0;
    for (int j=0;j<3;j++) beta_up[i] += gamma_up[i][j]*beta_dn[j];
  }

  /* alpha^2 = -(g00 - gamma_ij beta^i beta^j) */
  double betabeta = 0.0;
  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
      betabeta += gamma_dn[i][j]*beta_up[i]*beta_up[j];

  const double alpha2 = -(gdn[0][0] - betabeta);
  if (!isfinite(alpha2) || alpha2 <= 0.0) return 0;

  const double alpha = sqrt(alpha2);
  if (alpha_out) *alpha_out = alpha;

  /* g^{00}, g^{0i}, g^{ij} */
  const double inva2 = 1.0/(alpha2);

  gup[0][0] = -inva2;
  for (int i=0;i<3;i++)
  {
    gup[0][i+1] = beta_up[i]*inva2;
    gup[i+1][0] = gup[0][i+1];
  }
  for (int i=0;i<3;i++)
  for (int j=0;j<3;j++)
    gup[i+1][j+1] = gamma_up[i][j] - beta_up[i]*beta_up[j]*inva2;

  return 1;
}

/* ============================================================================================
   Grid-mode sampler helpers (metric_interp = local27/aei/carpet,
   time_interp = off/linear).  See the parfile keywords; the
   local27+off combination reproduces the original code path
   bit-for-bit (those branches are left untouched above).
   ============================================================================================ */

/* ADMBase fields in the fixed order used by aei/carpet interpolation
   (the betax/betay/betaz fields hold the contravariant shift beta^i) */
static const char *const adm_field[10] = { "ADMBASE::gxx", "ADMBASE::gxy", "ADMBASE::gxz",
                                           "ADMBASE::gyy", "ADMBASE::gyz", "ADMBASE::gzz",
                                           "ADMBASE::alp", "ADMBASE::betax", "ADMBASE::betay",
                                           "ADMBASE::betaz" };
static int adm_varidx[10];
static int adm_varidx_ok = 0;

/* 27-term basis: value and normalized derivatives at (x,y,z) =
   (y[]-below)/dx  (physical derivatives = returned values / dx_i) */
static inline void poly27_eval_d(const double y[], int m, const CCTK_REAL c[POLY27],
                                 const CCTK_INT is_const,
                                 CCTK_REAL *val, CCTK_REAL *dvx, CCTK_REAL *dvy, CCTK_REAL *dvz)
{
  if (is_const)
  {
    *val = c[0];
    *dvx = *dvy = *dvz = 0.0;
    return;
  }
  const CCTK_REAL x  = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
  const CCTK_REAL yy = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
  const CCTK_REAL zz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
  *val = c[0]+c[1]*x+c[2]*yy+c[3]*zz+c[4]*x*x+c[5]*yy*yy+c[6]*zz*zz
       + c[7]*x*yy+c[8]*x*zz+c[9]*yy*zz
       + c[10]*x*x*yy+c[11]*x*x*zz+c[12]*x*yy*yy+c[13]*yy*yy*zz
       + c[14]*x*zz*zz+c[15]*yy*zz*zz+c[16]*x*yy*zz
       + c[17]*x*x*yy*yy+c[18]*x*x*zz*zz+c[19]*yy*yy*zz*zz
       + c[20]*x*x*yy*zz+c[21]*x*yy*yy*zz+c[22]*x*yy*zz*zz
       + c[23]*x*x*yy*yy*zz+c[24]*x*x*yy*zz*zz+c[25]*x*yy*yy*zz*zz+c[26]*x*x*yy*yy*zz*zz;
  *dvx = c[1]+2.0*c[4]*x+c[7]*yy+c[8]*zz+2.0*c[10]*x*yy+2.0*c[11]*x*zz+c[12]*yy*yy
       + c[14]*zz*zz+c[16]*yy*zz+2.0*c[17]*x*yy*yy+2.0*c[18]*x*zz*zz
       + 2.0*c[20]*x*yy*zz+c[21]*yy*yy*zz+c[22]*yy*zz*zz+2.0*c[23]*x*yy*yy*zz
       + 2.0*c[24]*x*yy*zz*zz+c[25]*yy*yy*zz*zz+2.0*c[26]*x*yy*yy*zz*zz;
  *dvy = c[2]+2.0*c[5]*yy+c[7]*x+c[9]*zz+c[10]*x*x+2.0*c[12]*x*yy+2.0*c[13]*yy*zz
       + c[15]*zz*zz+c[16]*x*zz
       + 2.0*c[17]*x*x*yy+2.0*c[19]*yy*zz*zz+c[20]*x*x*zz+2.0*c[21]*x*yy*zz
       + c[22]*x*zz*zz+2.0*c[23]*x*x*yy*zz+c[24]*x*x*zz*zz+2.0*c[25]*x*yy*zz*zz
       + 2.0*c[26]*x*x*yy*zz*zz;
  *dvz = c[3]+c[8]*x+c[9]*yy+2.0*c[6]*zz+c[11]*x*x+c[13]*yy*yy+2.0*c[14]*x*zz
       + 2.0*c[15]*yy*zz+c[16]*x*yy+2.0*c[18]*x*x*zz+2.0*c[19]*yy*yy*zz
       + c[20]*x*x*yy+c[21]*x*yy*yy+2.0*c[22]*x*yy*zz+c[23]*x*x*yy*yy
       + 2.0*c[24]*x*x*yy*zz+2.0*c[25]*x*yy*yy*zz+2.0*c[26]*x*x*yy*yy*zz*zz;
}

/* 3+1 -> 4-metric at one cell of time level tl (mirrors the metric block
   of cf2_grid, which reads time level 0) */
static void metric_at_cell_tl(int tl, int ix, int iy, int iz, gd_data *out)
{
  const int index = LD_INDEX(ix, iy, iz);
  const double gxxL=gview.gxx[tl][index], gxyL=gview.gxy[tl][index], gxzL=gview.gxz[tl][index];
  const double gyyL=gview.gyy[tl][index], gyzL=gview.gyz[tl][index], gzzL=gview.gzz[tl][index];
  const double lapse=gview.alp[tl][index], shiftx=gview.betax[tl][index];
  const double shifty=gview.betay[tl][index], shiftz=gview.betaz[tl][index];
  const double shift_x = gxxL*shiftx+gxyL*shifty+gxzL*shiftz;
  const double shift_y = gxyL*shiftx+gyyL*shifty+gyzL*shiftz;
  const double shift_z = gxzL*shiftx+gyzL*shifty+gzzL*shiftz;
  out->gd[0][0] = -lapse*lapse+(shiftx*shift_x+shifty*shift_y+shiftz*shift_z);
  out->gd[0][1]=out->gd[1][0]=shift_x;
  out->gd[0][2]=out->gd[2][0]=shift_y;
  out->gd[0][3]=out->gd[3][0]=shift_z;
  out->gd[1][1]=gxxL; out->gd[1][2]=out->gd[2][1]=gxyL; out->gd[1][3]=out->gd[3][1]=gxzL;
  out->gd[2][2]=gyyL; out->gd[2][3]=out->gd[3][2]=gyzL; out->gd[3][3]=gzzL;
}

/* metric sample of the 3x3x3 stencil (local indices i4,i5,i6 in 0..2)
   at time level tl: raw index = i + point[] - 2, as in cf2_grid */
static void metric_27_sample_tl(int m, int tl, int i4, int i5, int i6, gd_data *out)
{
  metric_at_cell_tl(tl, i4+gstate.pgeodesic[m].point[1]-2,
                    i5+gstate.pgeodesic[m].point[2]-2,
                    i6+gstate.pgeodesic[m].point[3]-2, out);
}

/* compose 4-metric, dgd[deriv][mu][nu] (row 0 left zero for the caller)
   and inverse metric from the 3+1 fields g31[adm_field order] and their
   spatial derivatives dg31[f][k]; dgd/dg31/gup may be NULL */
static void compose_4metric(const double g31[10], const double dg31[10][3],
                            double gdn[4][4], double dgd[4][4][4], double gup[4][4])
{
  double gamma_dn[3][3], gamma_up[3][3], dgamma[3][3][3];
  const double beta_up[3] = { g31[7], g31[8], g31[9] };
  const double alp = g31[6];
  double betabeta = 0.0, det;
  int i, j, k;
  gamma_dn[0][0]=g31[0]; gamma_dn[0][1]=gamma_dn[1][0]=g31[1];
  gamma_dn[0][2]=gamma_dn[2][0]=g31[2]; gamma_dn[1][1]=g31[3];
  gamma_dn[1][2]=gamma_dn[2][1]=g31[4]; gamma_dn[2][2]=g31[5];
  invert_spatial_metric_3x3(g31[0],g31[1],g31[2],g31[3],g31[4],g31[5],gamma_up,&det);
  for (i=0;i<3;i++) for (j=0;j<3;j++) betabeta += gamma_dn[i][j]*beta_up[i]*beta_up[j];
  gdn[0][0] = -alp*alp+betabeta;
  for (i=0;i<3;i++)
    gdn[0][i+1]=gdn[i+1][0]=gamma_dn[i][0]*beta_up[0]+gamma_dn[i][1]*beta_up[1]+gamma_dn[i][2]*beta_up[2];
  for (i=0;i<3;i++) for (j=0;j<3;j++) gdn[i+1][j+1]=gamma_dn[i][j];
  if (gup)
  {
    const double inva2 = 1.0/(alp*alp);
    gup[0][0] = -inva2;
    for (i=0;i<3;i++) gup[0][i+1]=gup[i+1][0]=beta_up[i]*inva2;
    for (i=0;i<3;i++) for (j=0;j<3;j++) gup[i+1][j+1]=gamma_up[i][j]-beta_up[i]*beta_up[j]*inva2;
  }
  if (dgd)
  {
    for (i=0;i<4;i++) for (j=0;j<4;j++) for (k=0;k<4;k++) dgd[i][j][k]=0.0;
    if (dg31)
    {
      for (k=0;k<3;k++)
      {
        dgamma[0][0][k]=dg31[0][k]; dgamma[0][1][k]=dgamma[1][0][k]=dg31[1][k];
        dgamma[0][2][k]=dgamma[2][0][k]=dg31[2][k]; dgamma[1][1][k]=dg31[3][k];
        dgamma[1][2][k]=dgamma[2][1][k]=dg31[4][k]; dgamma[2][2][k]=dg31[5][k];
      }
      for (k=0;k<3;k++)
      {
        double dgamma_bb=0.0, dgamma_beta=0.0;
        for (i=0;i<3;i++) for (j=0;j<3;j++)
        {
          dgamma_bb += dgamma[i][j][k]*beta_up[i]*beta_up[j];
          dgamma_beta += gamma_dn[i][j]*dg31[7+i][k]*beta_up[j];
        }
        dgd[k+1][0][0] = -2.0*alp*dg31[6][k]+dgamma_bb+2.0*dgamma_beta;
        for (i=0;i<3;i++)
        {
          double dg0i=0.0;
          for (j=0;j<3;j++) dg0i += dgamma[j][i][k]*beta_up[j]+gamma_dn[j][i]*dg31[7+j][k];
          dgd[k+1][0][i+1]=dg0i; dgd[k+1][i+1][0]=dg0i;
        }
        for (i=0;i<3;i++) for (j=0;j<3;j++) dgd[k+1][i+1][j+1]=dgamma[i][j][k];
      }
    }
  }
}

/* Christoffel symbols from dgd and gup (mirrors the cf2_grid formula,
   lower indices symmetric) */
static void cf2_pointwise(const double dgd[4][4][4], const double gup[4][4], double cf2[4][4][4])
{
  double cf1[4][4][4];
  int j1,j2,j3,j4;
  for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) for (j3=0;j3<4;j3++) { cf1[j1][j2][j3]=0.0; cf2[j1][j2][j3]=0.0; }
  for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) for (j3=j2;j3<4;j3++)
    cf1[j1][j2][j3] = 0.5*(dgd[j3][j2][j1]+dgd[j2][j1][j3]-dgd[j1][j2][j3]);
  for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) for (j3=0;j3<j2;j3++) cf1[j1][j2][j3]=cf1[j1][j3][j2];
  for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) for (j3=j2;j3<4;j3++) for (j4=0;j4<4;j4++)
    cf2[j1][j2][j3] += gup[j1][j4]*cf1[j4][j2][j3];
  for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) for (j3=0;j3<j2;j3++) cf2[j1][j2][j3]=cf2[j1][j3][j2];
}

/* Christoffel at the particle position from the local27 METRIC fits
   (pgd, pgd1), i.e. the adaptive (blend-then-inverse) approach at a
   single spatial level.  Metric value, spatial derivatives and inverse
   come from the current level (pgd); the time derivative is the 2-level
   difference (g0 - g1)/dt using the old level (pgd1), or 0 when the old
   level is unavailable (old_level_bad).  This is the robust replacement
   for the direct Christoffel-polynomial fit (cf2_func / pcf2), which was
   inaccurate in strong-field regions.  Used by the local27 sampler when
   adaptive_interp = on (default). */
static int cf2_metric_off(int m, const double y[], double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
  double g0, g1, dx, dy, dz;
  int mu, nu, d;
  const int oldbad = gstate.pgeodesic[m].old_level_bad;
  const double tx = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
  const double ty = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
  const double tz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
  for (mu=0;mu<4;mu++) for (nu=0;nu<4;nu++) for (d=0;d<4;d++) dgd[d][mu][nu]=0.0;
  for (mu=0;mu<4;mu++) for (nu=0;nu<4;nu++) gdn[mu][nu]=0.0;
  for (mu=0;mu<4;mu++)
    for (nu=mu;nu<4;nu++)
    {
      g0 = poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],gstate.pgeodesic[m].pgdc[mu][nu]);
      gdn[mu][nu]=g0; gdn[nu][mu]=g0;
      p27_deriv(tx,ty,tz,gstate.pgeodesic[m].pgd[mu][nu], &dx,&dy,&dz);
      dgd[1][mu][nu]=dx/gview.delta_space[0]; dgd[1][nu][mu]=dgd[1][mu][nu];
      dgd[2][mu][nu]=dy/gview.delta_space[1]; dgd[2][nu][mu]=dgd[2][mu][nu];
      dgd[3][mu][nu]=dz/gview.delta_space[2]; dgd[3][nu][mu]=dgd[3][mu][nu];
      if (oldbad) g1 = g0;
      else
        g1 = poly27_eval(y,m,gstate.pgeodesic[m].pgd1[mu][nu],gstate.pgeodesic[m].pgd1c[mu][nu]);
      dgd[0][mu][nu]=(g0-g1)/gview.dt; dgd[0][nu][mu]=dgd[0][mu][nu];
    }
  if (!build_gup_from_gdn(gdn,NULL,beta_up,gamma_dn,gamma_up,gup))
    return 0;
  cf2_pointwise(dgd,gup,cf2);
  return 1;
}

/* 27-point metric fit (operation-identical to the original pgd fit in
   cf2_fitting; for a static metric both time levels fit bit-identically) */
static void fit_metric_poly27(int m, const gd_data src[3][3][3],
                              CCTK_REAL dst[4][4][POLY27],
                              CCTK_INT dstc[4][4])
{
  const double rel_eps = 1e-8;
  double ly[27];
  int j1,j2,j4,j5,j6,j7,j8;
  (void)m;
  for (j1=0;j1<4;j1++) for (j2=j1;j2<4;j2++)
  {
    j8=0; double tmpy=0.0, tol=0.0; int vary=0;
    for (j5=0;j5<3;j5++) for (j6=0;j6<3;j6++) for (j7=0;j7<3;j7++)
    {
      ly[j8]=src[j7][j6][j5].gd[j1][j2];
      if (j8==0) { tmpy=ly[0]; tol=fabs(tmpy)*rel_eps+rel_eps; vary=0; }
      else if (!vary) { if (ly[j8]>tmpy+tol || ly[j8]<tmpy-tol) vary=1; }
      j8++;
    }
    if (vary)
    {
      for (j4=0;j4<lN;j4++) { dst[j1][j2][j4]=0.0; for (j5=0;j5<lN;j5++) dst[j1][j2][j4]+=MtxA[j4][j5]*ly[j5]; }
      dstc[j1][j2]=0;
    }
    else
    {
      for (j4=1;j4<lN;j4++) dst[j1][j2][j4]=0.0;
      dst[j1][j2][0]=tmpy;
      dstc[j1][j2]=1;
    }
  }
  for (j1=1;j1<4;j1++) for (j2=0;j2<j1;j2++)
  {
    for (j4=0;j4<lN;j4++) dst[j1][j2][j4]=dst[j2][j1][j4];
    dstc[j1][j2]=dstc[j2][j1];
  }
}

/* 27-point fit of the pointwise Christoffel symbols (2nd kind) at the
   stencil cells, for one time level (the caller precomputes both
   levels with pointwise_cf2_tl); operation-identical to the off-path
   Christoffel fit (same constant-field shortcut).  Fitting the
   Christoffels directly -- instead of fitting the metric and its
   6th-order FD derivatives separately and combining pointwise -- is
   what brings the time_interp = linear Christoffel to off-path
   accuracy. */
static void fit_cf2_poly27(const double G[27][4][4][4],
                           CCTK_REAL dst[4][4][4][POLY27],
                           CCTK_INT dstc[4][4][4])
{
  const double rel_eps = 1e-8;
  double ly[27];
  int j1,j2,j3,j4,j5,j6,j7,j8;
  for (j1=0;j1<4;j1++)
  for (j2=0;j2<4;j2++)
  for (j3=j2;j3<4;j3++)
  {
    j8=0; double tmpy=0.0, tol=0.0; int vary=0;
    for (j5=0;j5<3;j5++) for (j6=0;j6<3;j6++) for (j7=0;j7<3;j7++)
    {
      ly[j8] = G[j7*9+j6*3+j5][j1][j2][j3];
      if (j8==0) { tmpy=ly[0]; tol=fabs(tmpy)*rel_eps+rel_eps; vary=0; }
      else if (!vary) { if (ly[j8]>tmpy+tol || ly[j8]<tmpy-tol) vary=1; }
      j8++;
    }
    if (vary)
    {
      for (j4=0;j4<lN;j4++)
      {
        dst[j1][j2][j3][j4] = 0.0;
        for (j5=0;j5<lN;j5++)
          dst[j1][j2][j3][j4] += MtxA[j4][j5]*ly[j5];
      }
      dstc[j1][j2][j3] = 0;
    }
    else
    {
      for (j4=1;j4<lN;j4++) dst[j1][j2][j3][j4] = 0.0;
      dst[j1][j2][j3][0] = tmpy;
      dstc[j1][j2][j3] = 1;
    }
  }
  /* symmetry in the lower indices: Gamma^mu_{ab} = Gamma^mu_{ba} */
  for (j1=0;j1<4;j1++)
  for (j2=1;j2<4;j2++)
  for (j3=0;j3<j2;j3++)
  {
    for (j4=0;j4<lN;j4++) dst[j1][j2][j3][j4] = dst[j1][j3][j2][j4];
    dstc[j1][j2][j3] = dstc[j1][j3][j2];
  }
}

/* AEI pointwise interpolation (Lagrange tensor product, order 4):
   one point, 10 input fields x operations {0,1,2,3} = 40 outputs in
   outbuf[4*f+op]; returns 0 on success, nonzero on failure */
static int aei_interp_point(int tl, double x, double y, double z,
                            double g31[10], double dg31[10][3])
{
  static GEODESIC_TLS int aei_tab = -1;
  /* Util's table API is not thread-safe: concurrent Util_TableCreate
     calls can hand two threads the same handle, and every Util_Table*
     call then resolves through the shared handle array, so the two
     threads would concurrently modify one table's entry list.  The
     mutex below guarantees each thread gets a unique handle; after
     creation the table is touched only by its owning thread. */
  static pthread_mutex_t aei_tab_lock = PTHREAD_MUTEX_INITIALIZER;
  double outbuf[40];
  void *out[40];
  const void *in[10];
  const void *coords[3] = { &x, &y, &z };
  CCTK_INT dims[3] = { gview.lsh[0], gview.lsh[1], gview.lsh[2] };
  CCTK_INT types[40];
  int f, k, status;
  if (aei_tab < 0)
  {
    pthread_mutex_lock(&aei_tab_lock);
    if (aei_tab < 0)
    {
      CCTK_INT opcodes[40], opidx[40];
      for (f=0;f<10;f++) for (k=0;k<4;k++) { opcodes[4*f+k]=k; opidx[4*f+k]=f; }
      aei_tab = Util_TableCreate(UTIL_TABLE_FLAGS_DEFAULT);
      if (aei_tab >= 0)
      {
        Util_TableSetInt(aei_tab, 4, "order");
        Util_TableSetIntArray(aei_tab, 40, opcodes, "operation_codes");
        Util_TableSetIntArray(aei_tab, 40, opidx, "operand_indices");
      }
    }
    pthread_mutex_unlock(&aei_tab_lock);
  }
  if (aei_tab < 0) return -1;
  in[0]=gview.gxx[tl]; in[1]=gview.gxy[tl]; in[2]=gview.gxz[tl];
  in[3]=gview.gyy[tl]; in[4]=gview.gyz[tl]; in[5]=gview.gzz[tl];
  in[6]=gview.alp[tl]; in[7]=gview.betax[tl]; in[8]=gview.betay[tl]; in[9]=gview.betaz[tl];
  for (f=0;f<40;f++) { types[f]=CCTK_VARIABLE_REAL; out[f]=&outbuf[f]; }
  status = CCTK_InterpLocalUniform(3, gstate.aei_handle, aei_tab,
                                   gview.origin_space, gview.delta_space,
                                   1, CCTK_VARIABLE_REAL, coords,
                                   10, dims, types, in, 40, types, out);
  if (status != 0) return status;
  for (f=0;f<10;f++)
  {
    g31[f]=outbuf[4*f];
    if (dg31)
    {
      dg31[f][0]=outbuf[4*f+1];
      dg31[f][1]=outbuf[4*f+2];
      dg31[f][2]=outbuf[4*f+3];
    }
  }
  return 0;
}

/* CarpetInterp pointwise: order-2 (quadratic) Lagrange tensor product on
   the thread-local gview slot-0 arrays, all 10 fields at once.  The point
   maps to the nearest node tp (0-based, as in cf2_fitting; clamped to
   [2, lsh-1] so the 3x3x3 stencil never leaves the interior cells; a
   boundary particle gets the same half-cell-scale extrapolation as the
   multi_amr clip in cf2_fitting), and the value is a tensor product of
   the one-dimensional quadratic weights through the stencil nodes
   {tp-2, tp-1, tp}, evaluated at the true point.  Spatial derivatives are the analytic derivatives of the same
   quadratic interpolant (same order as the old 7-point centered
   differences).  Only the TLS gview copy is touched, so the routine is
   thread-safe and works on every rank (the old implementation called
   the CarpetInterp driver, which does collective MPI and is neither). */
static int carpet_interp_point(int tl, double x, double y, double z,
                               double g31[10], double dg31[10][3])
{
  const double *fp[10] = {
    gview.gxx[tl], gview.gxy[tl], gview.gxz[tl], gview.gyy[tl],
    gview.gyz[tl], gview.gzz[tl], gview.alp[tl],
    gview.betax[tl], gview.betay[tl], gview.betaz[tl]
  };
  const double q[3] = { x, y, z };
  double w[3][3], dw[3][3];
  int i0[3], f, ax, ay, az, k;
  for (k = 0; k < 3; k++)
  {
    CCTK_INT tp;
    double u = (q[k] - gview.origin_space[k]) * gview.delta_inv[k];
    double r;
    tp = (CCTK_INT)lrint(floor(u + 0.5));   /* 0-based nearest node, as in
                                               cf2_fitting (node n sits at
                                               origin + n*dx) */
    if (tp < 2) tp = 2;
    if (tp > gview.lsh[k] - 1) tp = gview.lsh[k] - 1;
    i0[k] = (int)(tp - 1);                  /* stencil center node, in [1, lsh-2]
                                               (stencil = {i0-1, i0, i0+1}) */
    r = u - i0[k];                          /* in [0.5, 1.5] (the particle is
                                               within half a cell of node tp =
                                               i0+1); [0, 1] / [1, 2] at the
                                               domain boundary (extrapolation,
                                               cf2_fitting) */
    w[k][0]  = 0.5 * r * (r - 1.0);
    w[k][1]  = 1.0 - r * r;
    w[k][2]  = 0.5 * r * (r + 1.0);
    dw[k][0] = r - 0.5;
    dw[k][1] = -2.0 * r;
    dw[k][2] = r + 0.5;
  }
  for (f = 0; f < 10; f++)
  {
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (ax = 0; ax < 3; ax++)
    for (ay = 0; ay < 3; ay++)
    for (az = 0; az < 3; az++)
    {
      double v = fp[f][LD_INDEX(i0[0]+ax-1, i0[1]+ay-1, i0[2]+az-1)];
      double wv = w[0][ax] * w[1][ay] * w[2][az];
      s[0] += wv * v;
      s[1] += dw[0][ax] * w[1][ay] * w[2][az] * v;
      s[2] += w[0][ax] * dw[1][ay] * w[2][az] * v;
      s[3] += w[0][ax] * w[1][ay] * dw[2][az] * v;
    }
    g31[f] = s[0];
    if (dg31)
    {
      dg31[f][0] = s[1] * gview.delta_inv[0];
      dg31[f][1] = s[2] * gview.delta_inv[1];
      dg31[f][2] = s[3] * gview.delta_inv[2];
    }
  }
  return 0;
}

/* theta of the particle's coordinate time y[0] within the level's two
   stored time levels.  L = the particle's level; its levels span
   [L->time - L->dt, L->time].  Absolute mapping (independent of sstep
   or bg) keeps the blend phase-correct under time subcycling, where a
   particle may sit anywhere in the window [t-dt, t].  Clamps to [0,1]:
   below the window -> oldest level, above -> newest (clamped). */
static inline double geodesic_time_theta(int m, double y0)
{
  const LevelData *L =
    &gstate.LD[gstate.pgeodesic[m].cur_map][gstate.pgeodesic[m].level];
  const double dt = L->dt;
  if (dt <= 0.0)
    return 1.0;
  double theta = (y0 - (L->time - dt)) / dt;
  if (theta < 0.0)
    theta = 0.0;
  if (theta > 1.0)
    theta = 1.0;
  return theta;
}

/* AEI order-4 pointwise 4-metric at a particle position.  The strong-
   curvature replacement for the local27 (order-2, 27-point) fit.  Must be
   called with the same particle (m, y) as aei_christoffel_at_point so the
   two produce identical metric frames (u.u = -1 renormalization).  Returns
   0 on success, -1 if the point is outside AEI's boundary/extrapolation
   tolerance (gdn left untouched in that case). */
static int aei_metric_at_point(int m, const double y[8], double gdn[4][4])
{
  double g31[10], dg31[10][3];
  if (aei_interp_point(0, y[1], y[2], y[3], g31, dg31) != 0)
    return -1;
  if (!gstate.stime_interp)
  {
    compose_4metric(g31, dg31, gdn, NULL, NULL);
    return 0;
  }
  {
    double g31a[10], dg31a[10][3];
    double theta = geodesic_time_theta(m, y[0]);
    if (aei_interp_point(1, y[1], y[2], y[3], g31a, dg31a) != 0)
      return -1;
    /* smoke hook: perturb the OLD level's samples (par smoke_timedep_eps);
       0 = strict no-op */
    if (gstate.smoke_eps != 0.0)
    {
      const double s1 = 1.0 - gstate.smoke_eps;
      for (int mu = 0; mu < 10; mu++)
      {
        g31a[mu] *= s1;
        for (int nu = 0; nu < 3; nu++) dg31a[mu][nu] *= s1;
      }
    }
    /* the two stored ADMBase time levels span [t-dt, t]; theta is the
       particle's absolute phase in that window (geodesic_time_theta),
       clamped to [0,1] */
    for (int mu = 0; mu < 10; mu++)
    {
      g31[mu] = g31a[mu] + theta * (g31[mu] - g31a[mu]);
      for (int nu = 0; nu < 3; nu++)
        dg31[mu][nu] = dg31a[mu][nu] + theta * (dg31[mu][nu] - dg31a[mu][nu]);
    }
    compose_4metric(g31, dg31, gdn, NULL, NULL);
    return 0;
  }
}

/* ===================== log-space (qwen1.txt) robust metric value ==== */

/* 1D 5-node (order-4) Lagrange weight for node offset a in {-2..+2} at the
   local coordinate p in [-0.5,0.5) -- the AEI a4 (3d cube size5, centered)
   convention: center = round(fp), p = fp - center. */
static inline double log_w5(int a, double p)
{
  /* 5-node order-4 Lagrange basis at nodes {-2,-1,0,1,2}, local coord p.
     Denominators are prod_{k!=a} (a - k): a=-2 -> 24, a=-1 -> -6, a=0 -> 4,
     a=1 -> -6, a=2 -> 24. */
  switch (a) {
    case -2: return (p+1.0)*p*(p-1.0)*(p-2.0)/24.0;
    case -1: return (p+2.0)*p*(p-1.0)*(p-2.0)/(-6.0);
    case  0: return (p+2.0)*(p+1.0)*(p-1.0)*(p-2.0)/4.0;
    case  1: return (p+2.0)*(p+1.0)*p*(p-2.0)/(-6.0);
    case  2: return (p+2.0)*(p+1.0)*p*(p-1.0)/24.0;
  }
  return 0.0;
}

/* Interpolate all 10 ADM fields at the particle (level tl) with the 5-node
   order-4 tensor-product molecule: log|.| space for the positive fields
   (gxx/gyy/gzz/lapse) and regular space for the six sign-changing fields.
   Fills g31; returns 0 on success, -1 if the point is too close to the grid
   boundary for the 5-node stencil (or a log field is non-positive). */
static int interp_log_field_all(int tl, const double y[8], double g31[10])
{
  const double x = y[1], yy = y[2], z = y[3];
  const CCTK_REAL *fields[10] = {
      gview.gxx[tl], gview.gxy[tl], gview.gxz[tl],
      gview.gyy[tl], gview.gyz[tl], gview.gzz[tl],
      gview.alp[tl], gview.betax[tl], gview.betay[tl], gview.betaz[tl]
  };
  static const int log_field[10] = { 1,0,0,  1,0,0,  1,0,0,0 };
  const double o[3] = {gview.origin_space[0], gview.origin_space[1],
                       gview.origin_space[2]};
  const double d[3] = {gview.delta_space[0], gview.delta_space[1],
                       gview.delta_space[2]};
  const double rlb[3] = {gview.r_lbnd[0], gview.r_lbnd[1], gview.r_lbnd[2]};
  const double rub[3] = {gview.r_ubnd[0], gview.r_ubnd[1], gview.r_ubnd[2]};
  double fp[3] = {(x-o[0])/d[0], (yy-o[1])/d[1], (z-o[2])/d[2]};
  int  c[3]   = {(int)lround(fp[0]), (int)lround(fp[1]), (int)lround(fp[2])};
  double p[3] = {fp[0]-c[0], fp[1]-c[1], fp[2]-c[2]};
  /* the 5-node stencil must lie inside the safe (non-ghost) region */
  for (int ax = 0; ax < 3; ax++) {
    double sli = (rlb[ax] - o[ax])/d[ax];
    double sui = (rub[ax] - o[ax])/d[ax];
    if (c[ax]-2 < sli || c[ax]+2 > sui) return -1;
  }
  double w[3][5];
  for (int ax = 0; ax < 3; ax++)
    for (int a = -2; a <= 2; a++)
      w[ax][a+2] = log_w5(a, p[ax]);
  for (int f = 0; f < 10; f++) {
    double V = 0.0;
    for (int ii = 0; ii < 5; ii++)
      for (int jj = 0; jj < 5; jj++)
        for (int kk = 0; kk < 5; kk++) {
          int gi = c[0]+(ii-2), gj = c[1]+(jj-2), gk = c[2]+(kk-2);
          double fval = fields[f][LD_INDEX(gi,gj,gk)];
          if (log_field[f]) {
            if (fval <= 0.0) return -1;   /* guard: log of non-positive */
            V += w[0][ii]*w[1][jj]*w[2][kk] * log(fval);
          } else {
            V += w[0][ii]*w[1][jj]*w[2][kk] * fval;
          }
        }
    g31[f] = log_field[f] ? exp(V) : V;
  }
  return 0;
}

/* Log-space robust-path metric value (qwen1.txt).  The four positive ADM
   fields (gxx, gyy, gzz, lapse) are interpolated in log|.| space to suppress
   strong-curvature oscillation; the six sign-changing fields are in regular
   space.  On a smooth metric this agrees with aei_metric_at_point to O(dx^4)
   (same 5-node order-4 molecule).  Returns 0 on success, -1 if the point is
   too close to the grid boundary for the 5-node stencil (caller falls back
   to AEI, which extrapolates). */
static int log_metric_at_point(int m, const double y[8], double gdn[4][4])
{
  double g31[10];
  double theta = geodesic_time_theta(m, y[0]);
  if (gstate.pgeodesic[m].old_level_bad) theta = 1.0;
  if (gstate.stime_interp) {
    double g31a[10];
    if (interp_log_field_all(0, y, g31)  != 0) return -1;
    if (interp_log_field_all(1, y, g31a) != 0) return -1;
    if (gstate.smoke_eps != 0.0)
      for (int mu = 0; mu < 10; mu++) g31a[mu] *= (1.0 - gstate.smoke_eps);
    for (int mu = 0; mu < 10; mu++)
      g31[mu] = g31a[mu] + theta*(g31[mu] - g31a[mu]);
  } else {
    if (interp_log_field_all(0, y, g31) != 0) return -1;
  }
  compose_4metric(g31, NULL, gdn, NULL, NULL);
  if (m < 2 && getenv("GEODESIC_LOG_DBG") != NULL)
  {
    double aei_gdn[4][4];
    if (aei_metric_at_point(m, y, aei_gdn) == 0)
      for (int mu = 0; mu < 4; mu++)
        for (int nu = mu; nu < 4; nu++)
          fprintf(stderr, "[LOG_DBG] m=%d t=%.9f x=%.7f y=%.7f z=%.7f g[%d,%d] log=%.15g aei=%.15g |d|=%.15e\n",
                m, y[0], y[1], y[2], y[3], mu, nu,
                gdn[mu][nu], aei_gdn[mu][nu], fabs(gdn[mu][nu] - aei_gdn[mu][nu]));
  }
  return 0;
}

/* forward declarations (the 4D robust-interpolation functions are defined
   further below, after aei_christoffel_at_point) */
static int fourd_metric_at_point(int m, const double y[8], double gdn[4][4]);
static int fourd_christoffel_at_point(int m, const double y[8],
                                      double cf2[4][4][4]);
static int fourdh_metric_at_point(int m, const double y[8], double gdn[4][4]);
static int fourdh_christoffel_at_point(int m, const double y[8],
                                       double cf2[4][4][4]);

/* Robust-path metric value: 4D tensor product when robust_metric_interp is
   on (mode 1 = fourd: interpolate the ADM fields, compose g at the sample
   point; mode 2 = fourdh: compose g at each node and interpolate h = g - eta,
   eta added back at the sample point), else log-space when
   log_metric_interp is on, else plain AEI; on any boundary failure fall back
   to AEI (which has boundary extrapolation). */
static int robust_metric_at_point(int m, const double y[8], double gdn[4][4])
{
  if (gstate.robust_metric_interp == 1 && fourd_metric_at_point(m, y, gdn) == 0)
    return 0;
  if (gstate.robust_metric_interp == 2 && fourdh_metric_at_point(m, y, gdn) == 0)
    return 0;
  if (gstate.log_metric_interp && log_metric_at_point(m, y, gdn) == 0)
    return 0;
  return aei_metric_at_point(m, y, gdn);
}

/* AEI order-4 pointwise Christoffel symbols at a particle position.  The
   strong-curvature replacement for the local27 blend-then-inversion path;
   also the core of the dedicated aei sampler's (sinterp_mode 1)
   `christoffel_at_point` branch.  Must be called with the same particle
   (m, y) as aei_metric_at_point so the metric frames agree.  Returns 1 on
   success, 0 on interpolation failure. */
static int aei_christoffel_at_point(int m, const double y[8], double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double g31[10], dg31[10][3];
  int j1, j2;
  if (aei_interp_point(0, y[1], y[2], y[3], g31, dg31) != 0)
    return 0;
  if (!gstate.stime_interp)
  {
    compose_4metric(g31, dg31, gdn, dgd, gup);
    if (gview.metric_static)
    {
      for (j1 = 0; j1 < 4; j1++)
        for (j2 = 0; j2 < 4; j2++)
          dgd[0][j1][j2] = 0.0;
    }
    else
    {
      double g31a[10], g31b[10], gnn1[4][4], gnn2[4][4];
      if (aei_interp_point(1, y[1], y[2], y[3], g31a, NULL) != 0)
        return 0;
      if (aei_interp_point(2, y[1], y[2], y[3], g31b, NULL) != 0)
        return 0;
      compose_4metric(g31a, NULL, gnn1, NULL, NULL);
      compose_4metric(g31b, NULL, gnn2, NULL, NULL);
      /* 2nd-order 3-point time difference of the full 4-metric:
         gdn = level 0 (t), gnn1 = level 1 (t-dt), gnn2 = level 2 (t-2dt) */
      for (j1 = 0; j1 < 4; j1++)
        for (j2 = 0; j2 < 4; j2++)
          dgd[0][j1][j2] = (3.0 * gdn[j1][j2] - 4.0 * gnn1[j1][j2] + gnn2[j1][j2]) * (0.5 / gview.dt);
    }
    cf2_pointwise(dgd, gup, cf2);
    return 1;
  }
  {
    double g31a[10], dg31a[10][3], gnn0[4][4], gnn1[4][4];
    double theta = geodesic_time_theta(m, y[0]);
    if (aei_interp_point(1, y[1], y[2], y[3], g31a, dg31a) != 0)
      return 0;
    /* smoke hook: perturb the OLD level's samples (par smoke_timedep_eps);
       0 = strict no-op */
    if (gstate.smoke_eps != 0.0)
    {
      const double s1 = 1.0 - gstate.smoke_eps;
      for (j1 = 0; j1 < 10; j1++)
      {
        g31a[j1] *= s1;
        for (j2 = 0; j2 < 3; j2++) dg31a[j1][j2] *= s1;
      }
    }
    compose_4metric(g31, NULL, gnn0, NULL, NULL);
    compose_4metric(g31a, NULL, gnn1, NULL, NULL);
    for (j1 = 0; j1 < 10; j1++)
    {
      g31[j1] = g31a[j1] + theta * (g31[j1] - g31a[j1]);
      for (j2 = 0; j2 < 3; j2++)
        dg31[j1][j2] = dg31a[j1][j2] + theta * (dg31[j1][j2] - dg31a[j1][j2]);
    }
    compose_4metric(g31, dg31, gdn, dgd, gup);
    for (j1 = 0; j1 < 4; j1++)
      for (j2 = 0; j2 < 4; j2++)
        dgd[0][j1][j2] = (gnn0[j1][j2] - gnn1[j1][j2]) / gview.dt;
    cf2_pointwise(dgd, gup, cf2);
    return 1;
  }
}

/* ===================== 4D tensor-product robust interpolation ========== */
/* True 4D (space 3D x time 1D) Lagrange interpolation for the local27
   robust (use_robust) path, selected by Geodesic::robust_metric_interp.
   Space: the 5-node order-4 Lagrange molecule (log_w5, identical to the
   AEI a4).  Time: a 3-node order-2 Lagrange over the three stored ADMBase
   levels [0]=t, [1]=t-dt, [2]=t-2dt, evaluated at the particle's fractional
   time s = geodesic_time_theta in [0,1] (between levels 1 and 0).  The
   full tensor product gives both the value and all four derivatives (3
   spatial + 1 time) from one consistent interpolant: value O(dx^4) x
   O(dt^3), spatial derivatives O(dx^3) x O(dt^3), time derivative O(dt^2).
   This is strictly more consistent than AEI (value and derivative from the
   same interpolant evaluated at the particle's true time, not a 2-level
   blend with a separate difference).  Falls back to the plain AEI path on
   boundary / level failure.  Only used for strong_christ_off = 0 (the AEI
   strong-region path); the pure-l27 path is unaffected. */

/* 1D 3-node (order-2) Lagrange weight for time level lvl at fractional time
   s in [0,1].  Local node positions: level0 (t) = +1, level1 (t-dt) = 0,
   level2 (t-2dt) = -1.  Sum over lvl of fourd_tw(lvl, s) == 1 for all s. */
static inline double fourd_tw(int lvl, double s)
{
  switch (lvl) {
    case 0: return s * (s + 1.0) / 2.0;   /* level 0 at s = +1  */
    case 1: return 1.0 - s * s;           /* level 1 at s =  0  */
    case 2: return s * (s - 1.0) / 2.0;   /* level 2 at s = -1  */
  }
  return 0.0;
}

/* derivative of fourd_tw with respect to the local fractional time s; the
   physical time derivative is (this) / gview.dt. */
static inline double fourd_tw_a(int lvl, double s)
{
  switch (lvl) {
    case 0: return (2.0 * s + 1.0) / 2.0;
    case 1: return -2.0 * s;
    case 2: return (2.0 * s - 1.0) / 2.0;
  }
  return 0.0;
}

/* analytic derivative of the 5-node order-4 weight log_w5(a, p) with respect
   to the local coordinate p.  The log_w5 polynomials are exactly the Lagrange
   basis at nodes -2..2, so dL_j/dp = L_j * sum_{k!=j} 1/(p-k).  Safe for all
   p because the sum excludes k == a (no division by zero). */
static inline double log_w5a(int a, double p)
{
  double w = log_w5(a, p);
  double s = 0.0;
  for (int k = -2; k <= 2; k++)
    if (k != a) s += 1.0 / (p - (double)k);
  return w * s;
}

/* 5-node order-4 spatial tensor-product interpolation of all 10 ADM fields at
   level tl.  Fills g31[] with values and, if dg31 is non-NULL, the spatial
   derivatives d/dx, d/dy, d/dz (in physical units) in dg31[][3].  Returns
   -1 if the 5-node stencil does not lie inside the safe (non-ghost) region;
   the caller then falls back to AEI, which extrapolates past the boundary. */
static int interp_4d_space(int tl, double x, double y, double z,
                           double g31[10], double dg31[10][3])
{
  const CCTK_REAL *fields[10] = {
      gview.gxx[tl], gview.gxy[tl], gview.gxz[tl],
      gview.gyy[tl], gview.gyz[tl], gview.gzz[tl],
      gview.alp[tl], gview.betax[tl], gview.betay[tl], gview.betaz[tl]
  };
  const double o[3] = {gview.origin_space[0], gview.origin_space[1], gview.origin_space[2]};
  const double dinv[3] = {gview.delta_inv[0], gview.delta_inv[1], gview.delta_inv[2]};
  const double rlb[3] = {gview.r_lbnd[0], gview.r_lbnd[1], gview.r_lbnd[2]};
  const double rub[3] = {gview.r_ubnd[0], gview.r_ubnd[1], gview.r_ubnd[2]};
  const double fp[3] = {(x - o[0]) * dinv[0], (y - o[1]) * dinv[1], (z - o[2]) * dinv[2]};
  const int c[3] = {(int)lround(fp[0]), (int)lround(fp[1]), (int)lround(fp[2])};
  const double p[3] = {fp[0] - (double)c[0], fp[1] - (double)c[1], fp[2] - (double)c[2]};
  /* the 5-node stencil must lie inside the safe (non-ghost) region */
  for (int ax = 0; ax < 3; ax++) {
    double sli = (rlb[ax] - o[ax]) * dinv[ax];
    double sui = (rub[ax] - o[ax]) * dinv[ax];
    if (c[ax] - 2 < sli || c[ax] + 2 > sui) return -1;
  }
  double w[3][5], dw[3][5];
  for (int ax = 0; ax < 3; ax++)
    for (int a = -2; a <= 2; a++) {
      w[ax][a + 2] = log_w5(a, p[ax]);
      dw[ax][a + 2] = log_w5a(a, p[ax]) * dinv[ax];   /* dW/d(coord) */
    }
  for (int f = 0; f < 10; f++) {
    double V = 0.0, D0 = 0.0, D1 = 0.0, D2 = 0.0;
    for (int ii = 0; ii < 5; ii++)
      for (int jj = 0; jj < 5; jj++)
        for (int kk = 0; kk < 5; kk++) {
          double v = fields[f][LD_INDEX(c[0] + ii - 2, c[1] + jj - 2, c[2] + kk - 2)];
          V  += w[0][ii] * w[1][jj] * w[2][kk] * v;
          D0 += dw[0][ii] * w[1][jj] * w[2][kk] * v;
          D1 += w[0][ii] * dw[1][jj] * w[2][kk] * v;
          D2 += w[0][ii] * w[1][jj] * dw[2][kk] * v;
        }
    g31[f] = V;
    if (dg31) { dg31[f][0] = D0; dg31[f][1] = D1; dg31[f][2] = D2; }
  }
  return 0;
}

/* 4D robust-path metric value.  The value is the full tensor product: for each
   of the 3 time levels, the 5-node spatial interpolation is blended with the
   3-node time weights, then compose_4metric gives the 4-metric.  When the past
   time levels are unusable (metric_static: not filled yet / unreadable, or the
   particle's old_level_bad) only level 0 is read (s = 1 pins the time weights
   to (1,0,0)); reading the garbage past levels would poison the blend through
   0 * NaN.  Returns 0 on success, -1 on boundary / level failure (caller falls
   back to AEI). */
static int fourd_metric_at_point(int m, const double y[8], double gdn[4][4])
{
  const int t1only = (gstate.pgeodesic[m].old_level_bad || gview.metric_static) ? 1 : 0;
  const double s = t1only ? 1.0 : geodesic_time_theta(m, y[0]);
  const int nlvl = t1only ? 1 : 3;
  double g31t[3][10];
  for (int lvl = 0; lvl < nlvl; lvl++) {
    if (interp_4d_space(lvl, y[1], y[2], y[3], g31t[lvl], NULL) != 0)
      return -1;
  }
  double g31tot[10];
  for (int f = 0; f < 10; f++) {
    double V = 0.0;
    for (int lvl = 0; lvl < nlvl; lvl++) V += fourd_tw(lvl, s) * g31t[lvl][f];
    g31tot[f] = V;
  }
  compose_4metric(g31tot, NULL, gdn, NULL, NULL);
  return 0;
}

/* 4D robust-path Christoffel symbols.  For each time level, the 5-node spatial
   interpolation (value + spatial derivatives) is blended with the 3-node time
   weights for the metric value AND spatial derivatives; the time derivative is
   the 3-node Lagrange derivative of the per-level metric values.  When the past
   time levels are unusable (metric_static / old_level_bad) only level 0 is
   read and d_t g = 0, matching the AEI branch's static handling.  Returns 1 on
   success, 0 on boundary / level failure (caller falls back to AEI). */
static int fourd_christoffel_at_point(int m, const double y[8], double cf2[4][4][4])
{
  double g31t[3][10], dg31t[3][10][3];
  double gdn_lvl[3][4][4];
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double g31tot[10], dg31tot[10][3];
  const double dt = gview.dt;
  if (dt <= 0.0) return 0;
  const int t1only = (gstate.pgeodesic[m].old_level_bad || gview.metric_static) ? 1 : 0;
  const double s = t1only ? 1.0 : geodesic_time_theta(m, y[0]);
  const int nlvl = t1only ? 1 : 3;
  for (int lvl = 0; lvl < nlvl; lvl++) {
    if (interp_4d_space(lvl, y[1], y[2], y[3], g31t[lvl], dg31t[lvl]) != 0)
      return 0;
    compose_4metric(g31t[lvl], dg31t[lvl], gdn_lvl[lvl], NULL, NULL);
  }
  for (int f = 0; f < 10; f++) {
    double V = 0.0;
    for (int lvl = 0; lvl < nlvl; lvl++) V += fourd_tw(lvl, s) * g31t[lvl][f];
    g31tot[f] = V;
    for (int a = 0; a < 3; a++) {
      double D = 0.0;
      for (int lvl = 0; lvl < nlvl; lvl++) D += fourd_tw(lvl, s) * dg31t[lvl][f][a];
      dg31tot[f][a] = D;
    }
  }
  compose_4metric(g31tot, dg31tot, gdn, dgd, gup);
  /* overwrite dgd[0] (zeroed by compose_4metric) with the 3-node Lagrange
     time derivative of the per-level metric values, in physical time units
     (0 when the past levels are unusable).  gdn_lvl[lvl] is symmetric, so
     dgd[0][mu][nu] is already symmetric. */
  if (t1only)
  {
    for (int mu = 0; mu < 4; mu++)
      for (int nu = 0; nu < 4; nu++)
        dgd[0][mu][nu] = 0.0;
  }
  else
  {
    for (int lvl = 0; lvl < 3; lvl++)
      for (int mu = 0; mu < 4; mu++)
        for (int nu = 0; nu < 4; nu++)
          dgd[0][mu][nu] = fourd_tw_a(lvl, s) / dt * gdn_lvl[lvl][mu][nu];
  }
  cf2_pointwise(dgd, gup, cf2);
  return 1;
}

/* ===================== fourdh: 4D interpolation of h = g - eta ========== */
/* Same 4D tensor-product machinery as fourd above, but the interpolated
   field is the metric deviation h_mn = g_mn - eta_mn (eta = diag(-1,1,1,1)):
   at each interpolation NODE the 4-metric is composed exactly from that
   node's alpha/beta/gamma (compose_4metric), h is taken, and the 10
   independent components of h are interpolated with the same 5-node space x
   3-node time Lagrange weights; eta is added back at the sample point.
   eta is constant, so dh = dg and the derivative path needs no change.
   Compared with fourd (interpolate the ADM fields, compose g at the sample
   point) this keeps the interpolation in the composed-metric frame, which is
   more consistent; it is slower (one 4-metric composition per node, 125 per
   time level) and is meant for the strong-curvature robust path where speed
   does not matter.  Falls back to the plain AEI path on boundary / level
   failure. */

/* the 10 independent components (mu <= nu) of a symmetric 4-tensor */
static const int fourdh_mu[10] = {0,0,0,0,1,1,1,2,2,3};
static const int fourdh_nu[10] = {0,1,2,3,1,2,3,2,3,3};

/* h <-> g conversion in place on a symmetric 4-tensor: h = g - eta,
   eta = diag(-1,1,1,1) */
static inline void fourdh_sub_eta(double g[4][4])
{
  g[0][0] += 1.0; g[1][1] -= 1.0; g[2][2] -= 1.0; g[3][3] -= 1.0;
}
static inline void fourdh_add_eta(double h[4][4])
{
  h[0][0] -= 1.0; h[1][1] += 1.0; h[2][2] += 1.0; h[3][3] += 1.0;
}

/* 5-node order-4 spatial tensor-product interpolation of the nodal 4-metric
   deviation h = g - eta at level tl.  Same stencil / weight logic as
   interp_4d_space (log_w5 / log_w5a), but the nodal values are the 10
   independent components of h_mn with g composed exactly at each node from
   that node's alpha/beta/gamma.  Fills h (symmetric) and, if dh is non-NULL,
   the spatial derivatives dh[mu][nu][k] = d_k h_mn in physical units
   (symmetric in mu,nu).  Returns -1 if the 5-node stencil does not lie
   inside the safe (non-ghost) region; the caller then falls back to AEI. */
static int interp_4d_space_h(int tl, double x, double y, double z,
                             double h[4][4], double dh[4][4][3])
{
  const CCTK_REAL *fields[10] = {
      gview.gxx[tl], gview.gxy[tl], gview.gxz[tl],
      gview.gyy[tl], gview.gyz[tl], gview.gzz[tl],
      gview.alp[tl], gview.betax[tl], gview.betay[tl], gview.betaz[tl]
  };
  const double o[3] = {gview.origin_space[0], gview.origin_space[1], gview.origin_space[2]};
  const double dinv[3] = {gview.delta_inv[0], gview.delta_inv[1], gview.delta_inv[2]};
  const double rlb[3] = {gview.r_lbnd[0], gview.r_lbnd[1], gview.r_lbnd[2]};
  const double rub[3] = {gview.r_ubnd[0], gview.r_ubnd[1], gview.r_ubnd[2]};
  const double fp[3] = {(x - o[0]) * dinv[0], (y - o[1]) * dinv[1], (z - o[2]) * dinv[2]};
  const int c[3] = {(int)lround(fp[0]), (int)lround(fp[1]), (int)lround(fp[2])};
  const double p[3] = {fp[0] - (double)c[0], fp[1] - (double)c[1], fp[2] - (double)c[2]};
  /* the 5-node stencil must lie inside the safe (non-ghost) region */
  for (int ax = 0; ax < 3; ax++) {
    double sli = (rlb[ax] - o[ax]) * dinv[ax];
    double sui = (rub[ax] - o[ax]) * dinv[ax];
    if (c[ax] - 2 < sli || c[ax] + 2 > sui) {
      if (getenv("GEODESIC_DBG_STENCIL")) { static int n_st = 0;
        if (n_st < 10) { fprintf(stderr, "[STEN] tl=%d ax=%d c=%d sli=%.3f sui=%.3f pos=(%.4f %.4f %.4f) lsh=(%d %d %d) o=(%.4f %.4f %.4f) dx=(%.6f %.6f %.6f) dt=%g lvl=%d\n",
                                 tl, ax, c[ax], sli, sui, x, y, z,
                                 (int)gview.lsh[0], (int)gview.lsh[1], (int)gview.lsh[2],
                                 o[0], o[1], o[2],
                                 1.0/dinv[0], 1.0/dinv[1], 1.0/dinv[2], gview.dt,
                                 (int)gview.level); n_st++; } }
      return -1;
    }
  }
  double w[3][5], dw[3][5];
  for (int ax = 0; ax < 3; ax++)
    for (int a = -2; a <= 2; a++) {
      w[ax][a + 2] = log_w5(a, p[ax]);
      dw[ax][a + 2] = log_w5a(a, p[ax]) * dinv[ax];   /* dW/d(coord) */
    }
  for (int mu = 0; mu < 4; mu++)
    for (int nu = 0; nu < 4; nu++) {
      h[mu][nu] = 0.0;
      if (dh) { dh[mu][nu][0] = 0.0; dh[mu][nu][1] = 0.0; dh[mu][nu][2] = 0.0; }
    }
  for (int ii = 0; ii < 5; ii++)
    for (int jj = 0; jj < 5; jj++)
      for (int kk = 0; kk < 5; kk++) {
        /* nodal 4-metric deviation: compose g exactly at this node, h = g - eta */
        double g31node[10], gn[4][4];
        for (int f = 0; f < 10; f++)
          g31node[f] = fields[f][LD_INDEX(c[0] + ii - 2, c[1] + jj - 2, c[2] + kk - 2)];
        compose_4metric(g31node, NULL, gn, NULL, NULL);
        fourdh_sub_eta(gn);
        const double wv = w[0][ii] * w[1][jj] * w[2][kk];
        const double d0 = dw[0][ii] * w[1][jj] * w[2][kk];
        const double d1 = w[0][ii] * dw[1][jj] * w[2][kk];
        const double d2 = w[0][ii] * w[1][jj] * dw[2][kk];
        for (int cmp = 0; cmp < 10; cmp++) {
          const int mu = fourdh_mu[cmp], nu = fourdh_nu[cmp];
          const double hv = gn[mu][nu];
          h[mu][nu] += wv * hv;
          if (dh) {
            dh[mu][nu][0] += d0 * hv;
            dh[mu][nu][1] += d1 * hv;
            dh[mu][nu][2] += d2 * hv;
          }
        }
      }
  /* symmetric lower triangle */
  for (int mu = 1; mu < 4; mu++)
    for (int nu = 0; nu < mu; nu++) {
      h[mu][nu] = h[nu][mu];
      if (dh) {
        dh[mu][nu][0] = dh[nu][mu][0];
        dh[mu][nu][1] = dh[nu][mu][1];
        dh[mu][nu][2] = dh[nu][mu][2];
      }
    }
  return 0;
}

/* fourdh robust-path metric value: per time level the nodal-h spatial
   interpolation (values only), blended with the 3-node time weights, eta
   added back at the sample point.  When the past time levels are unusable
   (metric_static / old_level_bad) only level 0 is read (see
   fourd_metric_at_point).  Returns 0 on success, -1 on boundary / level
   failure (caller falls back to AEI). */
static int fourdh_metric_at_point(int m, const double y[8], double gdn[4][4])
{
  const int t1only = (gstate.pgeodesic[m].old_level_bad || gview.metric_static) ? 1 : 0;
  const double s = t1only ? 1.0 : geodesic_time_theta(m, y[0]);
  const int nlvl = t1only ? 1 : 3;
  double h_lvl[3][4][4];
  for (int lvl = 0; lvl < nlvl; lvl++) {
    if (interp_4d_space_h(lvl, y[1], y[2], y[3], h_lvl[lvl], NULL) != 0)
      return -1;
  }
  for (int cmp = 0; cmp < 10; cmp++) {
    const int mu = fourdh_mu[cmp], nu = fourdh_nu[cmp];
    double V = 0.0;
    for (int lvl = 0; lvl < nlvl; lvl++) V += fourd_tw(lvl, s) * h_lvl[lvl][mu][nu];
    gdn[mu][nu] = V;
  }
  for (int mu = 1; mu < 4; mu++)
    for (int nu = 0; nu < mu; nu++)
      gdn[mu][nu] = gdn[nu][mu];
  fourdh_add_eta(gdn);
  return 0;
}

/* fourdh robust-path Christoffel symbols: per time level the nodal-h
   interpolation (value + spatial derivatives) gives the per-level 4-metric
   g_lvl = h_lvl + eta; the metric value and spatial derivatives are blended
   with the 3-node time weights (eta is constant, so the blend of h equals
   the blend of g minus eta and dh = dg); the time derivative is the 3-node
   Lagrange derivative of the per-level metric values.  When the past time
   levels are unusable (metric_static / old_level_bad) only level 0 is read
   and d_t g = 0, matching the AEI branch's static handling.  Returns 1 on
   success, 0 on boundary / level / inversion failure (caller falls back to
   AEI). */
static int fourdh_christoffel_at_point(int m, const double y[8], double cf2[4][4][4])
{
  double h_lvl[3][4][4], dh_lvl[3][4][4][3];
  double gdn_lvl[3][4][4];
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
  const double dt = gview.dt;
  if (dt <= 0.0) return 0;
  const int t1only = (gstate.pgeodesic[m].old_level_bad || gview.metric_static) ? 1 : 0;
  const double s = t1only ? 1.0 : geodesic_time_theta(m, y[0]);
  const int nlvl = t1only ? 1 : 3;
  for (int lvl = 0; lvl < nlvl; lvl++) {
    if (interp_4d_space_h(lvl, y[1], y[2], y[3], h_lvl[lvl], dh_lvl[lvl]) != 0)
      return 0;
    for (int mu = 0; mu < 4; mu++)
      for (int nu = 0; nu < 4; nu++)
        gdn_lvl[lvl][mu][nu] = h_lvl[lvl][mu][nu];
    fourdh_add_eta(gdn_lvl[lvl]);
  }
  for (int cmp = 0; cmp < 10; cmp++) {
    const int mu = fourdh_mu[cmp], nu = fourdh_nu[cmp];
    double V = 0.0;
    for (int lvl = 0; lvl < nlvl; lvl++) V += fourd_tw(lvl, s) * h_lvl[lvl][mu][nu];
    gdn[mu][nu] = gdn[nu][mu] = V;
    for (int a = 0; a < 3; a++) {
      double D = 0.0;
      for (int lvl = 0; lvl < nlvl; lvl++) D += fourd_tw(lvl, s) * dh_lvl[lvl][mu][nu][a];
      dgd[a + 1][mu][nu] = dgd[a + 1][nu][mu] = D;
    }
  }
  fourdh_add_eta(gdn);
  /* time derivative: 3-node Lagrange derivative of the per-level metric
     values, in physical time units; 0 when the past levels are unusable
     (dgd[0] symmetric by construction) */
  if (t1only)
  {
    for (int mu = 0; mu < 4; mu++)
      for (int nu = 0; nu < 4; nu++)
        dgd[0][mu][nu] = 0.0;
  }
  else
  {
    for (int mu = 0; mu < 4; mu++)
      for (int nu = 0; nu < 4; nu++) {
        double D = 0.0;
        for (int lvl = 0; lvl < 3; lvl++)
          D += fourd_tw_a(lvl, s) / dt * gdn_lvl[lvl][mu][nu];
        dgd[0][mu][nu] = D;
      }
  }
  if (!build_gup_from_gdn(gdn, NULL, beta_up, gamma_dn, gamma_up, gup))
    return 0;
  cf2_pointwise(dgd, gup, cf2);
  return 1;
}

/* Hybrid strong-curvature Christoffel (the user's mixed AEI+l27 scheme):
   the 4-metric VALUE comes from the AEI order-4 pointwise interpolant --
   identical to the strong-curvature branch of metric_at_point, so the
   u.u = -1 renormalization stays in the same frame -- while the
   derivatives come from the local27 metric fits: the spatial derivatives
   are the analytic derivatives of the poly27 fits (p27_deriv), blended
   between the two time levels with theta, and the time derivative is the
   2-level difference (g0 - g1)/dt of the poly27 evaluations (0 when the
   old level is unavailable, old_level_bad).  The Christoffel is then
   recomputed pointwise (build_gup_from_gdn + cf2_pointwise).  Returns 1
   on success, 0 when the AEI metric interpolation fails (the caller falls
   back to the pure local27 robust path). */
static int hybrid_christoffel_at_point(int m, const double y[8], double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
  const int oldbad = gstate.pgeodesic[m].old_level_bad;
  double theta = geodesic_time_theta(m, y[0]);
  if (oldbad) theta = 1.0;

  /* metric value from the robust interpolant (AEI or log-space, same frame
     as the metric_at_point strong-curvature branch) */
  if (robust_metric_at_point(m, y, gdn) != 0)
    return 0;

  const double tx = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
  const double ty = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
  const double tz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
  for (int a=0;a<4;a++) for (int b=0;b<4;b++) for (int d=0;d<4;d++) dgd[a][b][d]=0.0;
  for (int mu=0;mu<4;mu++)
    for (int nu=mu;nu<4;nu++)
    {
      double dcur[3], dold[3];
      const double g0 = poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],
                                    gstate.pgeodesic[m].pgdc[mu][nu]);
      p27_deriv(tx,ty,tz,gstate.pgeodesic[m].pgd[mu][nu], &dcur[0], &dcur[1], &dcur[2]);
      if (oldbad)
      {
        dgd[0][mu][nu] = 0.0;
        dold[0]=dcur[0]; dold[1]=dcur[1]; dold[2]=dcur[2];
      }
      else
      {
        const double g1 = poly27_eval(y,m,gstate.pgeodesic[m].pgd1[mu][nu],
                                      gstate.pgeodesic[m].pgd1c[mu][nu]);
        dgd[0][mu][nu] = (g0-g1)/gview.dt;
        p27_deriv(tx,ty,tz,gstate.pgeodesic[m].pgd1[mu][nu], &dold[0], &dold[1], &dold[2]);
      }
      dgd[0][nu][mu] = dgd[0][mu][nu];
      dgd[1][mu][nu] = (dold[0]+theta*(dcur[0]-dold[0]))/gview.delta_space[0];
      dgd[1][nu][mu] = dgd[1][mu][nu];
      dgd[2][mu][nu] = (dold[1]+theta*(dcur[1]-dold[1]))/gview.delta_space[1];
      dgd[2][nu][mu] = dgd[2][mu][nu];
      dgd[3][mu][nu] = (dold[2]+theta*(dcur[2]-dold[2]))/gview.delta_space[2];
      dgd[3][nu][mu] = dgd[3][mu][nu];
    }
  if (!build_gup_from_gdn(gdn, NULL, beta_up, gamma_dn, gamma_up, gup))
    return 0;
  cf2_pointwise(dgd, gup, cf2);
  return 1;
}

/* 4-metric at a particle position (renormalization; void).
   sexact: exact Kerr-Schild.  Grid modes: level selection via
   cf2_fitting (switch_to_level); interpolation failure flags the
   particle out-of-bounds and leaves gdn zeroed by the caller. */
static void metric_at_point(int m, const double y[8], double bg, double gdn[4][4])
{
  int mu, nu;
  if (m < 3 && getenv("GEODESIC_DBG_METRIC") != NULL)
    fprintf(stderr, "[MET_DBG] m=%d sinterp_mode=%d sexact=%d outbd=%d stime_interp=%d y=(%.6f %.6f %.6f)\n",
      m, gstate.sinterp_mode, gstate.sexact, (int)gstate.pgeodesic[m].outbd,
      gstate.stime_interp, y[1], y[2], y[3]);
  if (gstate.sexact)
  {
    const double tx=y[1], ty=y[2], tz=y[3];
    for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++) gdn[mu][nu]=tg4dn(0,tx,ty,tz,mu,nu);
    for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
    return;
  }
  cf2_fitting(m, y);
  if (gstate.pgeodesic[m].outbd) return;
  if (gstate.sinterp_mode == 0)
  {
    /* strong-curvature point: use the AEI order-4 pointwise interpolant
       (higher order than the local27 order-2 fit) for the metric VALUE when
       strong_christ_off == 0 ("hybrid").  The same value is used by
       hybrid_christoffel_at_point (which takes the derivatives from the
       local27 poly27 fits instead), so the per-step renormalization
       (u.u = -1) and the geodesic update share one metric frame.  For
       strong_christ_off == 1 ("l27") we skip this and take the pure-l27
       blended metric below, so the norm check frame matches the l27
       robust Christoffel.  On AEI failure we fall through to the l27 path
       below whose stencil clamp keeps the particle. */
    if (gstate.pgeodesic[m].use_robust && gstate.strong_christ_off == 0)
    {
      if (robust_metric_at_point(m, y, gdn) == 0)
        return;
    }

    /* local27 + off + cheap (use_robust not yet set): the historical path
       uses only the current time level (pgd); no blend, no pgd1 read.  This
       is exactly what the pre-fix off path returned, kept for the cheap
       default and for A/B comparison. */
    if (!gstate.pgeodesic[m].use_robust && !gstate.stime_interp)
    {
      for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++)
        gdn[mu][nu] =
          poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],
                      gstate.pgeodesic[m].pgdc[mu][nu]);
      for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
      return;
    }

    /* the two stored ADMBase time levels span [t-dt, t] with t the level
       data time; theta is the particle's absolute phase in that window
       (geodesic_time_theta), clamped to [0,1].  This metric MUST be
       identical to
       the one used by christoffel_at_point (robust path) to compute the
       Christoffel, so that the per-step renormalization (u.u = -1) is done
       in the same frame as the geodesic update.  The old stime_interp=off
       special case that used only the current level was consistent with the
       previous cf2_metric_off Christoffel, but after the off-path switch to
       christoffel_at_point it became inconsistent and drove the 4-velocity
       off the u.u=-1 surface, causing the tau-loop cap.  Unifying both
       stime_interp values to the blended metric restores consistency. */
    double theta = geodesic_time_theta(m, y[0]);
    /* old level invalid (all-zero / non-finite) -> blend is undefined;
       pin to the current level instead.  This is the safe, exact limit
       for a static metric and the only sensible choice when the past
       level is not filled at this AMR cell. */
    if (gstate.pgeodesic[m].old_level_bad) theta=1.0;
    {
      static int dbg_mpt = -1, n_dbg_mpt = 0;
      if (dbg_mpt < 0) dbg_mpt = (getenv("GEODESIC_DEBUG_METRIC") != NULL);
      if (dbg_mpt && m < 20)
      {
        const double g0c = poly27_eval(y,m,gstate.pgeodesic[m].pgd[0][0],gstate.pgeodesic[m].pgdc[0][0]);
        const double g1c = poly27_eval(y,m,gstate.pgeodesic[m].pgd1[0][0],gstate.pgeodesic[m].pgd1c[0][0]);
        n_dbg_mpt++;
        fprintf(stderr,
          "[METRIC_DBG] m=%d n=%d y0=%.17g bg=%.17g sstep=%.17g theta=%.17g "
          "old_level_bad=%d g0=%.17g g1=%.17g -> %.17g lvl=%d outbd=%d\n",
          m, n_dbg_mpt, y[0], bg, gstate.sstep, theta,
          (int)gstate.pgeodesic[m].old_level_bad, g0c, g1c, g1c+theta*(g0c-g1c),
          (int)gstate.pgeodesic[m].level, (int)gstate.pgeodesic[m].outbd);
      }
    }
    for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++)
    {
      const double g0=poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],gstate.pgeodesic[m].pgdc[mu][nu]);
      const double g1=poly27_eval(y,m,gstate.pgeodesic[m].pgd1[mu][nu],gstate.pgeodesic[m].pgd1c[mu][nu]);
      gdn[mu][nu]=g1+theta*(g0-g1);
    }
    for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
    return;
  }
  if (gstate.sinterp_mode == 1)
  {
    double g31[10], dg31[10][3];
    {
      int st = aei_interp_point(0,y[1],y[2],y[3],g31,dg31);
      if (m < 3 && getenv("GEODESIC_DBG_METRIC") != NULL)
        fprintf(stderr, "[AEI_DBG] m=%d y=(%.6f %.6f %.6f) st=%d lsh=(%d %d %d) origin=(%.6f %.6f %.6f) delta=(%.6f %.6f %.6f) lvl=%d map=%d outbd=%d\n",
          m, y[1], y[2], y[3], st,
          (int)gview.lsh[0], (int)gview.lsh[1], (int)gview.lsh[2],
          gview.origin_space[0], gview.origin_space[1], gview.origin_space[2],
          gview.delta_space[0], gview.delta_space[1], gview.delta_space[2],
          gstate.pgeodesic[m].level, gstate.pgeodesic[m].cur_map, (int)gstate.pgeodesic[m].outbd);
      if (st != 0) { gstate.pgeodesic[m].outbd=1; return; }
    }
    if (!gstate.stime_interp)
    {
      compose_4metric(g31,dg31,gdn,NULL,NULL);
      return;
    }
    {
      double g31a[10], dg31a[10][3];
      /* the two stored ADMBase time levels span [t-dt, t] with t the level
         data time; theta is the particle's absolute phase in that window
         (geodesic_time_theta), clamped to [0,1] */
      double theta = geodesic_time_theta(m, y[0]);
      if (aei_interp_point(1,y[1],y[2],y[3],g31a,dg31a) != 0) { gstate.pgeodesic[m].outbd=1; return; }
      /* smoke hook: perturb the OLD level's samples (par smoke_timedep_eps);
         0 = strict no-op */
      if (gstate.smoke_eps != 0.0)
      {
        const double s1 = 1.0 - gstate.smoke_eps;
        for (mu=0;mu<10;mu++)
        {
          g31a[mu] *= s1;
          for (nu=0;nu<3;nu++) dg31a[mu][nu] *= s1;
        }
      }
      for (mu=0;mu<10;mu++)
      {
        g31[mu]=g31a[mu]+theta*(g31[mu]-g31a[mu]);
        for (nu=0;nu<3;nu++) dg31[mu][nu]=dg31a[mu][nu]+theta*(dg31[mu][nu]-dg31a[mu][nu]);
      }
      compose_4metric(g31,dg31,gdn,NULL,NULL);
    }
    return;
  }
  { /* carpet (time_interp = linear is rejected in ParamCheck) */
    double g31[10], dg31[10][3];
    if (gstate.stime_interp) { gstate.pgeodesic[m].outbd=1; return; }
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) { gstate.pgeodesic[m].outbd=1; return; }
    compose_4metric(g31,dg31,gdn,NULL,NULL);
  }
}

/* Christoffel symbols at a particle position for func_ode (the
   non local27+off paths).  Returns 1 on success, 0 on failure (the
   caller flags the particle and aborts the step). */
static int christoffel_at_point(int m, const double y[8], double bg, double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
  int j1, j2, j3;
  if (gstate.sinterp_mode == 0)
  {
    if (gstate.strong_christ_off == 0)
    {
      /* strong-curvature point (the "stable algorithm" for the robust path).
         With Geodesic::robust_metric_interp = "fourd" use the true 4D
         tensor-product Christoffel (fourd_christoffel_at_point), whose value
         and derivatives come from one consistent 4D interpolant and whose metric
         value agrees with fourd_metric_at_point (the value used by
         metric_at_point's robust branch, so the u.u = -1 renormalization is in
         the same frame).  "fourdh" is the same machinery applied to the nodal
         4-metric deviation h = g - eta (fourdh_christoffel_at_point), matching
         fourdh_metric_at_point's frame.  Otherwise use the AEI order-4
         pointwise Christoffel
         (aei_christoffel_at_point), which interpolates the metric in AEI's
         5-node order-4 Lagrange stencil in *both* space and time.  The AEI
         value matches the robust interpolant in metric_at_point (AEI and/or
         the log-space variant when Geodesic::log_metric_interp is set), so the
         two share the same frame in the default (fourd off) case.  This path
         is only invoked in mode 0 when the particle's use_robust flag is set
         (sadaptive && use_robust, func_ode), so it is exactly the strong-
         curvature / abnormal-lr region.  On engine failure we fall through to
         the local27 robust path. */
      if (gstate.robust_metric_interp == 1) {
        if (fourd_christoffel_at_point(m, y, cf2))
        {
          if (getenv("GEODESIC_DBG_ENGINE")) { static int n_eng = 0;
            if (n_eng < 40) { fprintf(stderr, "[ENG] m=%d t=%.6f mode=%d eng=fourd OK\n",
                                      m, y[0], (int)gstate.robust_metric_interp); n_eng++; } }
          return 1;
        }
      }
      if (gstate.robust_metric_interp == 2) {
        if (fourdh_christoffel_at_point(m, y, cf2))
        {
          if (getenv("GEODESIC_DBG_ENGINE")) { static int n_eng = 0;
            if (n_eng < 40) { fprintf(stderr, "[ENG] m=%d t=%.6f mode=%d eng=fourdh OK\n",
                                      m, y[0], (int)gstate.robust_metric_interp); n_eng++; } }
          return 1;
        }
      }
      if (aei_christoffel_at_point(m, y, cf2))
      {
        if (getenv("GEODESIC_DBG_ENGINE")) { static int n_eng2 = 0;
          if (n_eng2 < 40) { fprintf(stderr, "[ENG] m=%d t=%.6f mode=%d eng=aei FALLBACK\n",
                                    m, y[0], (int)gstate.robust_metric_interp); n_eng2++; } }
        return 1;
      }
    }
    else
    {
      /* strong_christ_off == 1 ("l27"): skip the AEI-value hybrid path entirely
         and use the pure-local27 robust (blend-then-inverse) Christoffel below.
         The metric frame must then come from l27 too -- see metric_at_point. */
    }
    /* local27+linear: blend the two ADMBase time levels.
       Fast/old path: blend the two fitted pointwise-Christoffel
       polynomials (for a stationary metric the two fits are identical
       and the blend is exact).
       Robust/new path: when the metric changes significantly between the
       two levels (large delta), the Christoffel-blend is inaccurate (the
       root cause of the superluminal particle drops near AMR level
       changes / strong curvature); there we recompute the Christoffel from
       the blended metric (blend metric, then invert), which is more robust. */
    /* the two stored ADMBase time levels span [t-dt, t] with t the level
       data time; theta is the particle's absolute phase in that window
       (geodesic_time_theta), clamped to [0,1] */
    double theta = geodesic_time_theta(m, y[0]);
    /* old level invalid -> blend undefined; pin to the current level. */
    const int oldbad = gstate.pgeodesic[m].old_level_bad;
    if (oldbad) theta=1.0;

    /* threshold above which the metric changed enough to warrant the
       robust blend-then-inverse path instead of the Christoffel blend.
       Estimated from the g_tt component of the two levels. */
    /* Always prefer the robust blend-then-inverse path when the old level
       is valid.  It is strictly more accurate than the Christoffel blend
       (fit the smoother 4-metric, then differentiate, rather than fitting
       the Christoffel directly) and it is cheaper (10 components x 2 levels
       vs 40 Christoffel components x 2 levels).  The delta is kept for
       diagnostics only.  This path is controlled by Geodesic::adaptive_interp
       (default "on"); "off" falls back to the historical Christoffel blend
       (the direct-fit behaviour). */
    int use_new = (gstate.sadaptive && !oldbad) ? 1 : 0;
    if (!oldbad)
    {
      const double g0c = poly27_eval(y,m,gstate.pgeodesic[m].pgd[0][0],gstate.pgeodesic[m].pgdc[0][0]);
      const double g1c = poly27_eval(y,m,gstate.pgeodesic[m].pgd1[0][0],gstate.pgeodesic[m].pgd1c[0][0]);
      double delta = 0.0;
      if (isnormal(g0c) && isnormal(g1c))
      {
        double d = fabs(g0c-g1c);
        double dmax = fabs(g0c);
        if (dmax<1.0) dmax=1.0;
        delta = d/dmax;
      }
      static int dbg_delta = -1;
      if (dbg_delta < 0) dbg_delta = (getenv("GEODESIC_DBG_DELTA") != NULL);
      if (dbg_delta && m < 5)
      {
        fprintf(stderr, "[DELTA_DBG] m=%d y=(%.3f %.3f %.3f) bg=%.15g sstep=%.15g "
                "theta=%.4f g_tt0=%.15g g_tt1=%.15g delta=%.9g\n",
                m, y[1], y[2], y[3], bg, gstate.sstep, theta, g0c, g1c, delta);
      }
    }

    if (!use_new)
    {
      /* old (fast) path: blend the two Christoffel polynomials.
         With theta=1 (oldbad) this is exactly the current level. */
      for (j1=0;j1<4;j1++)
      for (j2=0;j2<4;j2++)
      for (j3=j2;j3<4;j3++)
      {
        CCTK_REAL c0=poly27_eval(y,m,gstate.pgeodesic[m].pcf2[j1][j2][j3],gstate.pgeodesic[m].pcf2c[j1][j2][j3]);
        CCTK_REAL c1=poly27_eval(y,m,gstate.pgeodesic[m].pcf2_1[j1][j2][j3],gstate.pgeodesic[m].pcf2_1c[j1][j2][j3]);
        cf2[j1][j2][j3]=c1+theta*(c0-c1);
      }
      for (j1=0;j1<4;j1++)
      for (j2=1;j2<4;j2++)
      for (j3=0;j3<j2;j3++)
        cf2[j1][j2][j3]=cf2[j1][j3][j2];
      return 1;
    }

    /* ---- robust path: blend the metric, recompute the Christoffel ----
       gdn = blended 4-metric; dgd[dir][mu][nu] = (d/d x^dir) g_{mu nu},
       dir=0 is time (forward 2-level difference), dir=1,2,3 spatial. */
    {
      const double tx = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
      const double ty = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
      const double tz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
      const double dxp0 = gview.delta_space[0];
      const double dxp1 = gview.delta_space[1];
      const double dxp2 = gview.delta_space[2];
      const double dt   = gview.dt;
      for (int a=0;a<4;a++) for (int b=0;b<4;b++) for (int d=0;d<4;d++) dgd[a][b][d]=0.0;
      for (int a=0;a<4;a++) for (int b=0;b<4;b++) gdn[a][b]=0.0;
      for (int mu=0;mu<4;mu++)
      for (int nu=mu;nu<4;nu++)
      {
        const double g0 = poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],gstate.pgeodesic[m].pgdc[mu][nu]);
        const double g1 = poly27_eval(y,m,gstate.pgeodesic[m].pgd1[mu][nu],gstate.pgeodesic[m].pgd1c[mu][nu]);
        double gb = g1 + theta*(g0-g1);
        gdn[mu][nu]=gb; gdn[nu][mu]=gb;
        double dcur[3], dold[3];
        p27_deriv(tx,ty,tz,gstate.pgeodesic[m].pgd[mu][nu], &dcur[0], &dcur[1], &dcur[2]);
        p27_deriv(tx,ty,tz,gstate.pgeodesic[m].pgd1[mu][nu], &dold[0], &dold[1], &dold[2]);
        /* time derivative: forward difference of the two levels */
        dgd[0][mu][nu]=(g0-g1)/dt;  dgd[0][nu][mu]=dgd[0][mu][nu];
        /* spatial derivatives: blended normalized-coord derivatives / dxp */
        dgd[1][mu][nu]=(dold[0]+theta*(dcur[0]-dold[0]))/dxp0;  dgd[1][nu][mu]=dgd[1][mu][nu];
        dgd[2][mu][nu]=(dold[1]+theta*(dcur[1]-dold[1]))/dxp1;  dgd[2][nu][mu]=dgd[2][mu][nu];
        dgd[3][mu][nu]=(dold[2]+theta*(dcur[2]-dold[2]))/dxp2;  dgd[3][nu][mu]=dgd[3][mu][nu];
      }
      /* inverted metric g^{mu nu} from the blended Lorentzian 4-metric */
      if (!build_gup_from_gdn(gdn, NULL, beta_up, gamma_dn, gamma_up, gup))
      {
        /* blended metric not a valid Lorentzian metric -> fall back to the
           current level's Christoffel (theta=1) rather than returning bad
           data. */
        for (int a=0;a<4;a++) for (int b=0;b<4;b++) for (int c=0;c<4;c++) cf2[a][b][c]=0.0;
        for (int a=0;a<4;a++) for (int b=0;b<4;b++) for (int c=b;c<4;c++) {
          CCTK_REAL cc = poly27_eval(y,m,gstate.pgeodesic[m].pcf2[a][b][c],gstate.pgeodesic[m].pcf2c[a][b][c]);
          cf2[a][b][c] = cc;
        }
        for (int a=0;a<4;a++) for (int b=1;b<4;b++) for (int c=0;c<b;c++)
          cf2[a][b][c] = cf2[a][c][b];
        return 1;
      }
      cf2_pointwise(dgd, gup, cf2);
      return 1;
    }
  }
  if (gstate.sinterp_mode == 1)
  {
    double g31[10], dg31[10][3];
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) return 0;
    if (!gstate.stime_interp)
    {
      compose_4metric(g31,dg31,gdn,dgd,gup);
      if (gview.metric_static)
      {
        for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) dgd[0][j1][j2]=0.0;
      }
      else
      {
        double g31a[10], g31b[10], gdn1[4][4], gdn2[4][4];
        if (aei_interp_point(1,y[1],y[2],y[3],g31a,NULL) != 0) return 0;
        if (aei_interp_point(2,y[1],y[2],y[3],g31b,NULL) != 0) return 0;
        compose_4metric(g31a,NULL,gdn1,NULL,NULL);
        compose_4metric(g31b,NULL,gdn2,NULL,NULL);
        for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++)
          dgd[0][j1][j2]=(3.0*gdn[j1][j2]-4.0*gdn1[j1][j2]+gdn2[j1][j2])*(0.5/gview.dt);
      }
      cf2_pointwise(dgd,gup,cf2);
      return 1;
    }
    {
      double g31a[10], dg31a[10][3], gdn0[4][4], gdn1[4][4];
      /* the two stored ADMBase time levels span [t-dt, t] with t the level
         data time; theta is the particle's absolute phase in that window
         (geodesic_time_theta), clamped to [0,1] */
      double theta = geodesic_time_theta(m, y[0]);
      if (aei_interp_point(1,y[1],y[2],y[3],g31a,dg31a) != 0) return 0;
      /* smoke hook: perturb the OLD level's samples (par smoke_timedep_eps);
         0 = strict no-op */
      if (gstate.smoke_eps != 0.0)
      {
        const double s1 = 1.0 - gstate.smoke_eps;
        for (j1=0;j1<10;j1++)
        {
          g31a[j1] *= s1;
          for (j2=0;j2<3;j2++) dg31a[j1][j2] *= s1;
        }
      }
      compose_4metric(g31,NULL,gdn0,NULL,NULL);
      compose_4metric(g31a,NULL,gdn1,NULL,NULL);
      for (j1=0;j1<10;j1++)
      {
        g31[j1]=g31a[j1]+theta*(g31[j1]-g31a[j1]);
        for (j2=0;j2<3;j2++) dg31[j1][j2]=dg31a[j1][j2]+theta*(dg31[j1][j2]-dg31a[j1][j2]);
      }
      compose_4metric(g31,dg31,gdn,dgd,gup);
      for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) dgd[0][j1][j2]=(gdn0[j1][j2]-gdn1[j1][j2])/gview.dt;
      cf2_pointwise(dgd,gup,cf2);
      return 1;
    }
  }
  { /* carpet (time_interp = linear is rejected in ParamCheck).
     Spatial metric + spatial derivatives at the particle position.
     The spatial sampler is the order-4 Lagrange interpolator (the same
     AEILocalInterp-based routine aei uses) -- the previous hand-written
     quadratic (2nd-order, 3 nodes per dim) sampler was too inaccurate
     in the strong-field region and caused the particle-1 superluminal
     drop; the order-4 sampler fixes it.  The time derivative is the
     carpet-style 2nd-order 3-point difference of the FULL 4-metric at the
     particle position: d/dt = (3*g0 - 4*g1 + g2)/(2*dt). */
    double g31[10], dg31[10][3];
    double g31a[10], g31b[10];
    double gnn1[4][4], gnn2[4][4];
    if (gstate.stime_interp) { gstate.pgeodesic[m].outbd=1; return 0; }
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) return 0;
    compose_4metric(g31,dg31,gdn,dgd,gup);
    if (gview.metric_static)
    {
      for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) dgd[0][j1][j2]=0.0;
      cf2_pointwise(dgd,gup,cf2);
      return 1;
    }
    if (aei_interp_point(1,y[1],y[2],y[3],g31a,NULL) != 0) return 0;
    if (aei_interp_point(2,y[1],y[2],y[3],g31b,NULL) != 0) return 0;
    compose_4metric(g31a,NULL,gnn1,NULL,NULL);
    compose_4metric(g31b,NULL,gnn2,NULL,NULL);
    for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++)
      dgd[0][j1][j2]=(3.0*gdn[j1][j2]-4.0*gnn1[j1][j2]+gnn2[j1][j2])*(0.5/gview.dt);
    cf2_pointwise(dgd,gup,cf2);
    return 1;
  }
}

/* ============================================================================================
   solve_u0_future_directed:

   Given the spatial velocity components u_spatial[0..2] (= u^x, u^y, u^z,
   contravariant) and the local 4-metric gdn[4][4], solve
   g_{mu nu} u^mu u^nu = -1 for u^0 and return the physically correct,
   future-directed branch.

     ra*(u0)^2 + rb*u0 + rc = 0
     ra = g_00,  rb = 2*g_{0i}*u^i,  rc = 1 + gamma_ij*u^i*u^j  (rc > 0 always)

   Outside the ergosphere (g_00 < 0, ra < 0): the two roots have opposite
   signs (product rc/ra < 0); the unique positive root is future-directed
   and unambiguous.

   Inside the ergosphere (g_00 > 0, ra > 0): the two roots have the same
   sign (product rc/ra > 0).  The same spatial velocity u^i can then
   genuinely correspond to two 4-velocity solutions that both satisfy the
   normalization and both have u^0 > 0 -- a real geometric effect of frame
   dragging, not a numerical artifact.  The degeneracy must be broken by
   trajectory continuity: pick the root closest to u0_predictor (the u^0
   produced by this step's ODE solver before renormalization, or the
   previously accepted value).

   Returns 1 on success (*u0_out filled); returns 0 on failure (no valid
   future-directed solution -- the caller should flag the particle, e.g.
   gstate.pgeodesic[m].norm_bad = 1, and let the loss policy (drop / flag /
   respawn) handle it with a warning rather than filling in an arbitrary
   magic number).
   ============================================================================================ */
static int solve_u0_future_directed(const double gdn[4][4],
                                     const double u_spatial[3],
                                     double u0_predictor,
                                     double *u0_out)
{
  const double ra = gdn[0][0];
  const double rb = 2.0*( gdn[0][1]*u_spatial[0] + gdn[0][2]*u_spatial[1]
                         + gdn[0][3]*u_spatial[2] );

  double rc = 1.0;
  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
      rc += gdn[i+1][j+1]*u_spatial[i]*u_spatial[j];

  /* [DBG] GEODESIC_DBG_SOLVE: dump renorm inputs.
     Static counter so it does not flood on 100 simultaneous failures. */
  if (getenv("GEODESIC_DBG_SOLVE"))
  {
    static int s_sdbg = 0;
    if ((++s_sdbg) <= 200)
      fprintf(stderr, "[SOLVE_DBG] ra=%.6e rb=%.6e rc=%.6e | "
               "g01=%.6e g02=%.6e g03=%.6e | "
               "g11=%.6e g22=%.6e g33=%.6e | "
               "us=[%.6e %.6e %.6e] pred=%.6e\n",
              ra, rb, rc,
              gdn[0][1], gdn[0][2], gdn[0][3],
              gdn[1][1], gdn[2][2], gdn[3][3],
              u_spatial[0], u_spatial[1], u_spatial[2], u0_predictor);
  }

  /* ra degenerates to 0 (virtually impossible for a real metric; kept as
     a numerical guard): fall back to the linear equation */
  if (fabs(ra) < 1.0e-14)
  {
    if (fabs(rb) < 1.0e-300) return 0;
    const double root = -rc/rb;
    if (!isfinite(root) || root <= 0.0) return 0;
    *u0_out = root;
    return 1;
  }

  const double disc = rb*rb - 4.0*ra*rc;
  if (getenv("GEODESIC_DBG_SOLVE"))
  {
    static int s_sdbg2 = 0;
    if ((++s_sdbg2) <= 200)
      fprintf(stderr, "[SOLVE_DBG2] ra=%.6e disc=%.6e isfinite_disc=%d neg=%d\n",
              ra, disc, isfinite(disc), disc < 0.0);
  }
  if (!isfinite(disc) || disc < 0.0) return 0;   /* no real root: let the caller handle it */

  const double sq = sqrt(disc);
  const double r1 = (-rb - sq)/(2.0*ra);
  const double r2 = (-rb + sq)/(2.0*ra);

  const int ok1 = (isfinite(r1) && r1 > 0.0);
  const int ok2 = (isfinite(r2) && r2 > 0.0);

  if (ok1 && !ok2) { *u0_out = r1; return 1; }
  if (ok2 && !ok1) { *u0_out = r2; return 1; }

  if (ok1 && ok2)
  {
    /* Ergosphere degeneracy: break the tie using continuity (predictor) */
    if (!isfinite(u0_predictor) || u0_predictor <= 0.0)
      *u0_out = (r1 < r2) ? r1 : r2;              /* no usable predictor: take the smaller root */
    else
      *u0_out = (fabs(r1-u0_predictor) <= fabs(r2-u0_predictor)) ? r1 : r2;
    return 1;
  }

  /* [DBG] GEODESIC_DBG_SOLVE: neither root future-directed */
  if (getenv("GEODESIC_DBG_SOLVE"))
  {
    static int s_sdbg3 = 0;
    if ((++s_sdbg3) <= 200)
      fprintf(stderr, "[SOLVE_DBG3] FAIL: neither root>0 r1=%.6e r2=%.6e ra=%.6e rc=%.6e pred=%.6e\n",
              r1, r2, ra, rc, u0_predictor);
  }
  return 0;   /* neither root is future-directed: genuine failure */
}

/* ============================================================================================
   func_ode: RHS for ODE integration in proper time tau
   State y[0..7] = (t, x, y, z, u^t, u^x, u^y, u^z)
   Returns f = dy/dtau.
   ============================================================================================ */
int func_ode (double t, const double y[], double f[], void *params)
{
  ode_data * odedata = (ode_data *)params;
  int j1, j2, j3, j4, m;

  CCTK_REAL cf2[4][4][4];
  double tcf1[4][4][4];
  double tx, ty, tz;
  double tgd[4][4];
  double tg4up[4][4];
  double tdgd[4][4][4];

  m = odedata->m;
  /* Phase A: all particles are pinned to map 0; Phase B will assign it
     to the block the particle currently sits in. */
  const int pm = gstate.pgeodesic[m].cur_map;

  if (gstate.srverbose)
  {
    struct timeval tv;
    struct tm *current_time;
    gettimeofday(&tv, NULL);
    current_time = localtime(&tv.tv_sec);
    CCTK_VInfo(CCTK_THORNSTRING,
               "%02d:%02d:%02d.%03d m:%5i tau:%18.15G  Y:%18.15G %20.15G %20.15G %20.15G %20.15G %20.15G %20.15G %20.15G",
               current_time->tm_hour, current_time->tm_min, current_time->tm_sec,
               (int) (tv.tv_usec/1000),
               m, t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  }

  /* --------- out-of-bounds / level-change / refit check (both branches) ---------
     Once a particle leaves the SAFE region of level 0 it is marked
     outbd and handled by the loss policy after the step (same behavior
     in both branches).  Inside level 0 a finer level is always
     preferred automatically (pick_level). */
  if ((y[1] != 0.0 && (!isnormal(y[1]))) || (y[2] != 0.0 && (!isnormal(y[2]))) || (y[3] != 0.0 && (!isnormal(y[3]))))
  {
    gstate.pgeodesic[m].outbd = 1;
  }else
  {
    /* "out of bounds" = outside the SAFE region of the COARSEST level.
       Skipped in multi-process runs (multi_exact / multi_amr): each
       rank's local patch is only a portion of the global grid, so the
       local safe region would falsely drop particles that sit well
       inside the global grid (the old global-component runs never
       dropped them).  In multi_amr the ownership is handled by the
       interior boxes instead (mid-step check below). */
    /* Phase B (multi-block) crossing logic is armed only when there is
       genuinely more than one block.  A single-block run (per-hole AMR,
       plain unigrid) has nmaps <= 1 and must never enter
       locate_and_cross(); a particle outside the block's coarse safe
       region there is simply out of the domain.  nmaps is set by the
       one-shot init (0 until then); the > 1 test is safe in all cases. */
    if (!gstate.multi_exact && !gstate.multi_amr &&
        gstate.nmaps > 1 &&
        CCTK_IsFunctionAliased("MultiPatch_GetMap") &&
        !(in_level_dir(y, &gstate.LD[pm][0], 0) &&
          in_level_dir(y, &gstate.LD[pm][0], 1) &&
          in_level_dir(y, &gstate.LD[pm][0], 2)))
    {
      /* Phase B: particle left the current block's coarse safe region.
         It may have crossed into a neighbouring block of the multipath.
         locate_and_cross() re-expresses the position in the new block's
         local coords and transforms the spatial 4-velocity so the
         physical motion is continuous.  On success we flag crossed and
         abort this ODE step (GSL_EBADFUNC); the integrator restarts
         from the transformed state.  On failure the particle truly
         left the domain and is marked out of bounds. */
      int rc = locate_and_cross(m, (double *)y, odedata->cctkGH);
      if (rc == 1)
      {
        gstate.pgeodesic[m].crossed = 1;
        for (int k = 0; k < 8; k++)
          gstate.pgeodesic[m].crossed_y[k] = y[k];
        return GSL_EBADFUNC;
      }
      gstate.pgeodesic[m].outbd = 1;
    }else
    {
      const int lvl = gstate.multi_amr ? gstate.owner_lvl[m]
                         : (gstate.multi_exact ? 0 : pick_level(y, pm));
      if (lvl < 0)
      {
        gstate.pgeodesic[m].outbd = 1;
      }else
      {
        /* mid-step reachability (multi_amr, interpolated branch only):
           the 3x3x3 stencil with its 6th-order FD padding must stay
           inside this rank's local arrays, so the particle may move at
           most half a cell outside its owner patch during the step.
           The sexact branch has no grid access and is exempt. */
        if (gstate.multi_amr && !gstate.sexact)
        {
          const LevelData *Lo = &gstate.LD[pm][lvl];
          for (int d = 0; d < 3; d++)
            if (y[d+1] < Lo->i_lbnd[d] - 0.5*Lo->dx[d] || y[d+1] > Lo->i_ubnd[d] + 0.5*Lo->dx[d])
              gstate.pgeodesic[m].outbd = 1;
        }
        if (!gstate.pgeodesic[m].outbd &&
            (lvl != gstate.pgeodesic[m].level ||
             (y[1] < (gstate.pgeodesic[m].below[1]) || (y[1] > (gstate.pgeodesic[m].below[1] + 2*gstate.pgeodesic[m].dxp[0]))) ||
             (y[2] < (gstate.pgeodesic[m].below[2]) || (y[2] > (gstate.pgeodesic[m].below[2] + 2*gstate.pgeodesic[m].dxp[1]))) ||
             (y[3] < (gstate.pgeodesic[m].below[3]) || (y[3] > (gstate.pgeodesic[m].below[3] + 2*gstate.pgeodesic[m].dxp[2])))))
        {
          cf2_fitting(m, y);
        }
      }
    }
  }

  if (gstate.pgeodesic[m].outbd)
  {
    /* temporary debug: report which check flagged the particle (gated by
       GEODESIC_DEBUG_STEP, particle 0 only) */
    {
      static int dbg_ode = -1, n_dbg_ode = 0;
      if (dbg_ode < 0) dbg_ode = (getenv("GEODESIC_DEBUG_STEP") != NULL);
      if (dbg_ode && m == 0 && n_dbg_ode < 60)
      {
        n_dbg_ode++;
        fprintf(stderr, "ODE_DEBUG call=%d OUTBD t=%.17g y=(%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g)\n",
                n_dbg_ode, t, y[0], y[1], y[2], y[3], y[4], y[5], y[6]);
      }
    }
    return GSL_EBADFUNC;
  }

  /* AMR: ensure this thread's grid view (delta_space, data pointers, ghost cGH)
     matches the particle's level before any interpolation or FD stencil
     access.  switch_to_level is self-guarded (cheap no-op when already on
     the correct level+generation). */
  switch_to_level(pm, gstate.pgeodesic[m].level);

  /* ---------------- build Christoffel (either exact or interpolated) ---------------- */
  if (gstate.sexact)
  {
    tx = y[1];
    ty = y[2];
    tz = y[3];
    for(j1=0; j1<4; j1++)
      for(j2=j1; j2<4; j2++)
        tgd[j1][j2] = tg4dn(0, tx, ty, tz, j1, j2); /* tg4dn(double t, double x, double y, double z, int j1, int j2) */

    for(j1=1; j1<4; j1++)
      for(j2=0; j2<j1; j2++)
        tgd[j1][j2] = tgd[j2][j1];

    /* invert g_{mu nu} -> g^{mu nu} via the 3+1 (lapse/shift)
       decomposition: build_gup_from_gdn checks the spatial-metric
       determinant and alpha^2 and reports failure, instead of skipping
       singular pivots and returning a garbage inverse. */
    double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
    if (!build_gup_from_gdn(tgd, NULL, beta_up, gamma_dn, gamma_up, tg4up))
    {
      gstate.pgeodesic[m].outbd = 1;
      return GSL_EBADFUNC;
    }

    for(j1=0; j1<4; j1++)
      for(j2=0; j2<4; j2++)
        for(j3=j2; j3<4; j3++)
          tdgd[j1][j2][j3] = gdn_derivatives(j1, tx, ty, tz, j2, j3); /* gdn_derivatives(int d, double x, double y, double z, int j1, int j2); */

    for(j1=0; j1<4; j1++)
      for(j2=1; j2<4; j2++)
        for(j3=0; j3<j2; j3++)
          tdgd[j1][j2][j3] = tdgd[j1][j3][j2];

    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=0; j3<4; j3++)
        {
          tcf1[j1][j2][j3] = 0.0;
          cf2[j1][j2][j3] = 0.0;
        }

    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=j2; j3<4; j3++)
          tcf1[j1][j2][j3] += 0.5*(tdgd[j3][j2][j1] + tdgd[j2][j1][j3] - tdgd[j1][j2][j3]);

    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=0; j3<j2; j3++)
          tcf1[j1][j2][j3] = tcf1[j1][j3][j2];

    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=j2; j3<4; j3++)
          for (j4=0; j4<4; j4++)
            cf2[j1][j2][j3] += tg4up[j1][j4] * tcf1[j4][j2][j3];

  }else
  /* non-exact: interpolate cf2 */
  {
    if (gstate.sinterp_mode == 0)
    {
      /* local27: the robust (blend-then-inverse) path is taken only when
         adaptive_interp = on AND the particle's sticky use_robust flag is
         set (the strong-curvature criterion fired).  Otherwise the cheap
         path: off -> historical direct-fit Christoffel (cf2_func);
         linear -> blend of the two time-level Christoffel fits.  This
         makes the cheap path the default (fast, historical) behaviour and
         the robust path a per-particle promotion, per Geodesic::strong_curv_crit. */
      if (gstate.sadaptive && gstate.pgeodesic[m].use_robust)
      {
        /* robust: blend-then-inverse (fit the metric, blend time levels,
           invert, differentiate).  For the off path this reads pgd/pgd1,
           which are now both fitted by cf2_fitting (pgd1 fix).  For the
           linear path it blends the two level fits to theta. */
        if (!christoffel_at_point(m, y, odedata->bg, cf2))
        {
          gstate.pgeodesic[m].outbd = 1;
          return GSL_EBADFUNC;
        }
      }
      else if (!gstate.stime_interp)
      {
        /* off + cheap: historical 27-point Lagrange polynomial of the
           current-level fitted Christoffel */
        for (j1=0; j1<4; j1++)
          for (j2=0; j2<4; j2++)
            for (j3=j2; j3<4; j3++)
              cf2[j1][j2][j3] = cf2_func(y, m, j1, j2, j3);
      }
      else
      {
        /* linear + cheap: blend the two time-level Christoffel fits to the
           particle's fractional coordinate time.  Matches the "old (fast)
           path" in christoffel_at_point. */
        double theta = geodesic_time_theta(m, y[0]);
        if (gstate.pgeodesic[m].old_level_bad) theta=1.0;
        for (j1=0;j1<4;j1++)
          for (j2=0;j2<4;j2++)
            for (j3=j2;j3<4;j3++)
            {
              CCTK_REAL c0=poly27_eval(y,m,gstate.pgeodesic[m].pcf2[j1][j2][j3],gstate.pgeodesic[m].pcf2c[j1][j2][j3]);
              CCTK_REAL c1=poly27_eval(y,m,gstate.pgeodesic[m].pcf2_1[j1][j2][j3],gstate.pgeodesic[m].pcf2_1c[j1][j2][j3]);
              cf2[j1][j2][j3]=c1+theta*(c0-c1);
            }
        for (j1=0;j1<4;j1++)
          for (j2=1;j2<4;j2++)
            for (j3=0;j3<j2;j3++)
              cf2[j1][j2][j3]=cf2[j1][j3][j2];
      }
    }
    else if (!christoffel_at_point(m, y, odedata->bg, cf2))
    {
      gstate.pgeodesic[m].outbd = 1;
      return GSL_EBADFUNC;
    }
  }

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        cf2[j1][j2][j3] = cf2[j1][j3][j2];

  /* ---------------- dy/dtau for positions ---------------- */
  f[0] = y[4];
  f[1] = y[5];
  f[2] = y[6];
  f[3] = y[7];

  /* init accelerations */
  f[4] = 0.0;
  f[5] = 0.0;
  f[6] = 0.0;
  f[7] = 0.0;

  /* ---------------- geodesic acceleration term: -Gamma^mu_{ab} u^a u^b ---------------- */
  /* if (!gstate.sexact) */
  {
    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=0; j3<4; j3++)
          f[j1+4] -= cf2[j1][j2][j3]*y[j2+4]*y[j3+4];
  }

  /* track du^t/dtau for the tau_end prediction in geodesics_integrate,
     while we are still inside the current Cactus step */
  if (y[0] < odedata->bg + gstate.sstep)
  {
    odedata->a = f[4];
  }

  return GSL_SUCCESS;
}

/* Create the parent directories of a file path, best effort.
   Used for the particle dump file, which may be given as
   "some/dir/geodesic_particles.txt".  Already-existing components
   are skipped; any other failure (permissions, ...) is ignored --
   the fopen below will report the problem. */
static void geodesic_mkdir_parents(const char *path)
{
  char buf[4096];
  size_t len;
  size_t i;

  len = strlen(path);
  if (len == 0 || len >= sizeof(buf))
    return;
  memcpy(buf, path, len + 1);

  for (i = 1; i < len; i++)
  {
    if (buf[i] == '/')
    {
      buf[i] = '\0';
      if (mkdir(buf, 0755) != 0 && errno != EEXIST)
        return;            /* give up quietly; fopen reports the error */
      buf[i] = '/';
    }
  }
}

#ifdef CCTK_MPI
/* Multi-process AMR: exchange the full shared particle state with
   MPI_Allgather so every rank ends with the identical state.
   Payload per particle (GEODESIC_MPI_NPAY doubles):
   tt tx ty tz ttv txv tyv tzv ttw dead renorm_warn norm_bad
   freeze_warn rl_last outbd use_robust.  Full exchange (200 x 16 =
   23.7 KB/rank) keeps the code simple; every rank holds the complete
   state.  use_robust is a per-particle sticky flag that must survive
   ownership transfers between ranks.  The particle arrays are
   DISTRIB=CONSTANT grid variables, addressed through the patch-local
   pointers that DECLARE_CCTK_ARGUMENTS injects (cctkGH). */
static void geodesic_mmpi_sync(cGH *cctkGH, const int npr, const int me, const int npar)
{
  DECLARE_CCTK_ARGUMENTS
  const size_t n = (size_t)npar * GEODESIC_MPI_NPAY;
  if (gstate.mmpi_buf == NULL || gstate.mmpi_cap < (CCTK_INT)(npr * n))
  {
    free(gstate.mmpi_buf);
    gstate.mmpi_buf = (CCTK_REAL *) calloc((size_t)npr * n, sizeof(CCTK_REAL));
    if (!gstate.mmpi_buf)
      CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: MPI state buffer allocation failed.");
    gstate.mmpi_cap = (CCTK_INT)(npr * n);
  }
  double *full = (double *) gstate.mmpi_buf;
  double *s = full + (size_t)me * n;
  for (int mi = 0; mi < npar; mi++)
  {
    double *p = s + (size_t)mi * GEODESIC_MPI_NPAY;
    p[0] = particle_tt[mi];  p[1] = particle_tx[mi];  p[2] = particle_ty[mi];  p[3] = particle_tz[mi];
    p[4] = particle_ttv[mi]; p[5] = particle_txv[mi]; p[6] = particle_tyv[mi]; p[7] = particle_tzv[mi];
    p[8] = particle_ttw[mi]; p[9] = (double)gstate.pgeodesic[mi].dead;
    p[10] = (double)gstate.pgeodesic[mi].renorm_warn; p[11] = (double)gstate.pgeodesic[mi].norm_bad;
    p[12] = (double)gstate.pgeodesic[mi].freeze_warn; p[13] = gstate.pgeodesic[mi].rl_last;
    p[14] = (double)gstate.pgeodesic[mi].outbd;
    p[15] = (double)gstate.pgeodesic[mi].use_robust;
  }
  MPI_Allgather(s, (int)n, MPI_DOUBLE, full, (int)n, MPI_DOUBLE, MPI_COMM_WORLD);
  /* Write back per-particle, taking each particle's copy from the rank
     that INTEGRATED it (its owner).  A rank only advances the particles
     it owns; the other ranks hold the last-synced (stale) copy of those
     particles, so a bulk copy from any single rank would silently
     discard the integrations of every non-owning rank.  All ranks hold
     the same owner map (deterministic recompute on identical state), so
     they all select the same source rank per particle. */
  for (int mi = 0; mi < npar; mi++)
  {
    int r = me;
    if (gstate.owner_of != NULL && gstate.owner_of[mi] >= 0 && gstate.owner_of[mi] < npr)
      r = gstate.owner_of[mi];
    const double *p = full + (size_t)r * n + (size_t)mi * GEODESIC_MPI_NPAY;
    particle_tt[mi]  = p[0]; particle_tx[mi]  = p[1]; particle_ty[mi]  = p[2]; particle_tz[mi]  = p[3];
    particle_ttv[mi] = p[4]; particle_txv[mi] = p[5]; particle_tyv[mi] = p[6]; particle_tzv[mi] = p[7];
    particle_ttw[mi] = p[8]; gstate.pgeodesic[mi].dead = (CCTK_INT)p[9];
    gstate.pgeodesic[mi].renorm_warn = (CCTK_INT)p[10];
    gstate.pgeodesic[mi].norm_bad    = (CCTK_INT)p[11];
    gstate.pgeodesic[mi].freeze_warn = (CCTK_INT)p[12];
    gstate.pgeodesic[mi].rl_last     = p[13];
    gstate.pgeodesic[mi].outbd       = (CCTK_INT)p[14];
    gstate.pgeodesic[mi].use_robust  = (CCTK_INT)p[15];
  }
}
#endif

/* ------------------------------------------------------------------------------------------------
   geodesic_integrate_particle: integrate one particle from its
   current state up to coordinate time t_end (the Cactus step end
   for a single-pass run; a chunk boundary for the multi_amr chunked
   run).  Body of the per-particle section of the particle loop of
   geodesics_integrate, extracted verbatim (the caller skips dead /
   non-owned particles; the target time is passed as a parameter, so
   the func_ode tracking window bg + sstep still spans the whole
   Cactus step).  The non-multi_amr path calls it once per particle
   per step with t_end = bg + sstep, i.e. identical behavior to the
   original loop.

   t_step0 is the STEP-START coordinate time (the time of the older
   grid time level): the time_interp = linear blend in func_ode /
   christoffel_at_point anchors theta = (y[0] - bg)/sstep at bg, so
   bg must stay at the step start for every chunk of the multi_amr
   chunked integration.  Anchoring it at the particle's chunk-start
   time instead shifts the metric blend window by the elapsed chunks
   (the metric lags behind the true time), which corrupted the linear
   time interpolation in multi-process runs (1p is unaffected: one
   chunk, bg == step start).
------------------------------------------------------------------------------------------------- */
static void geodesic_integrate_particle(cGH *cctkGH, const int mi, const double t_end, const double t_step0)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  const int pm = gstate.pgeodesic[mi].cur_map;

    gsl_odeiv2_system sys;
    ode_data odedata;
    double ts, hstart, epsabs, epsrel;
    /* ODE tolerances */
    hstart = 1.0e-8;
    epsabs = 1.0e-8;
    epsrel = 0.0;

    /* the caller skips dead / non-owned particles; safety net only */
    if (gstate.pgeodesic[mi].dead)
      return;

    /* out-of-bounds quick check (SAFE region of the COARSEST level;
       LD[0] is shared read-only data, safe in the parallel loop).
       The boundary part is skipped in multi-process runs
       (multi_exact / multi_amr) for the same reason as in func_ode:
       each rank's local patch is only a portion of the global grid. */
    if ( (particle_tx[mi] != 0.0 && (!isnormal(particle_tx[mi]))) ||
         (particle_ty[mi] != 0.0 && (!isnormal(particle_ty[mi]))) ||
         (particle_tz[mi] != 0.0 && (!isnormal(particle_tz[mi]))) )
    {
      gstate.pgeodesic[mi].outbd = 1;
    }
    else if (!gstate.multi_exact && !gstate.multi_amr)
    {
      /* margin-appropriate coarse-level (level 0) safe region, using the
         same margin as in_level_dir / pick_level.  For the pointwise
         aei/carpet branch (sinterp_mode 1/2) the margin is small so a
         valid particle near a patch edge is not wrongly rejected
         (the old 7-cell margin rejected equatorial particles in the
         7-cell shadow zone of an off-centre AMR block). */
      const LevelData *L0 = &gstate.LD[pm][0];
      const int margin = safe_margin_cells();
      const double y[3] = { particle_tx[mi], particle_ty[mi], particle_tz[mi] };
      for (int d = 0; d < 3; d++)
      {
        const double lo = L0->origin[d] + (CCTK_REAL)margin * L0->dx[d];
        const double hi = L0->origin[d] + (CCTK_REAL)(L0->lsh[d]-1-margin) * L0->dx[d];
        if (y[d] < lo || y[d] > hi) { gstate.pgeodesic[mi].outbd = 1; break; }
      }
    }
    {
      /* debug: dump the pre-ODE out-of-bounds decision for mi=0 (all when GEODESIC_DBG_OUTBD_ALL) */
      static int dbg_outbd = -1;
      if (dbg_outbd < 0) dbg_outbd = (getenv("GEODESIC_DBG_OUTBD") != NULL) ? 1 : 0;
      if (dbg_outbd && (mi == 0 || getenv("GEODESIC_DBG_OUTBD_ALL")))
      {
        const LevelData *L0 = &gstate.LD[pm][0];
        int inx = (particle_tx[mi] >= L0->r_lbnd[0] && particle_tx[mi] <= L0->r_ubnd[0]
               && particle_ty[mi] >= L0->r_lbnd[1] && particle_ty[mi] <= L0->r_ubnd[1]
               && particle_tz[mi] >= L0->r_lbnd[2] && particle_tz[mi] <= L0->r_ubnd[2]);
        fprintf(stderr, "[OUTBD_DBG] it=%d mi=%d pm=%d pos=(%.4f,%.4f,%.4f) "
                  "lsh=(%d,%d,%d) lbnd=(%d,%d,%d) ubnd=(%d,%d,%d) "
                  "r_lbnd=(%.4f,%.4f,%.4f) r_ubnd=(%.4f,%.4f,%.4f) "
                  "origin=(%.4f,%.4f,%.4f) dx=(%.4f,%.4f,%.4f) inx=%d outbd=%d\n",
                  (int)cctk_iteration, mi, pm,
                  particle_tx[mi], particle_ty[mi], particle_tz[mi],
                  L0->lsh[0], L0->lsh[1], L0->lsh[2],
                  L0->lbnd[0], L0->lbnd[1], L0->lbnd[2],
                  L0->ubnd[0], L0->ubnd[1], L0->ubnd[2],
                  L0->r_lbnd[0], L0->r_lbnd[1], L0->r_lbnd[2],
                  L0->r_ubnd[0], L0->r_ubnd[1], L0->r_ubnd[2],
                  L0->origin[0], L0->origin[1], L0->origin[2],
                  L0->dx[0], L0->dx[1], L0->dx[2], inx, (int)gstate.pgeodesic[mi].outbd);
      }
    }

    /* ownership consistency (multi_amr): at a chunk start the owner map
       (recomputed from the synced positions at step start and at every
       chunk boundary) must agree with the position.  A particle outside
       its owner box means a stale ownership -- hand it to the loss
       policy.  In normal operation this never fires: the re-owning is
       computed from the same synced positions. */
    if (gstate.multi_amr)
    {
      const int ol = gstate.owner_lvl[mi];
      if (ol < 0 ||
          particle_tx[mi] < gstate.LD[pm][ol].i_lbnd[0] || particle_tx[mi] >= gstate.LD[pm][ol].i_ubnd[0] ||
          particle_ty[mi] < gstate.LD[pm][ol].i_lbnd[1] || particle_ty[mi] >= gstate.LD[pm][ol].i_ubnd[1] ||
          particle_tz[mi] < gstate.LD[pm][ol].i_lbnd[2] || particle_tz[mi] >= gstate.LD[pm][ol].i_ubnd[2])
        gstate.pgeodesic[mi].outbd = 1;
    }

    /* ---------------- integrate one particle ---------------- */
    memset((void *)&odedata, 0, sizeof(odedata));
    odedata.m  = mi;
    odedata.bg = t_step0;
    odedata.st = &gstate;
    odedata.cctkGH = cctkGH;   /* Phase B: multipatch transform geometry */

    sys = (gsl_odeiv2_system) {func_ode, NULL, 8, &odedata};

    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_rkf45;
    gsl_odeiv2_step *gstep = gsl_odeiv2_step_alloc(T, 8);
    gsl_odeiv2_control *control = gsl_odeiv2_control_y_new(epsabs, epsrel);
    gsl_odeiv2_evolve *evolve = gsl_odeiv2_evolve_alloc(8);

    if (!(gstep && control && evolve))
      CCTK_WARN(CCTK_WARN_ABORT, "Error: gsl_odeiv2 alloc failed.");

    CCTK_REAL ys[8];
    ys[0] = particle_tt[mi];
    ys[1] = particle_tx[mi];
    ys[2] = particle_ty[mi];
    ys[3] = particle_tz[mi];
    ys[4] = particle_ttv[mi];
    ys[5] = particle_txv[mi];
    ys[6] = particle_tyv[mi];
    ys[7] = particle_tzv[mi];

    {
      double gdn[4][4];
      for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
          gdn[mu][nu] = 0.0;

      /* 4-metric at the particle position (sexact / local27 / aei / carpet,
         see metric_at_point); an interpolation failure leaves gdn zeroed
         and flags the particle out-of-bounds, which the norm check below
         reports naturally */
      metric_at_point(mi, ys, ys[0], gdn);

      /* compute rl = u_mu u^mu */
      double u[4] = { ys[4], ys[5], ys[6], ys[7] };
      double u_cov[4] = {0,0,0,0};
      for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
          u_cov[mu] += gdn[mu][nu]*u[nu];

      double rl = 0.0;
      for (int mu=0; mu<4; ++mu) rl += u_cov[mu]*u[mu];

      /* ---------------- strong-curvature criterion (kimi2.txt) ----------------
         Decide whether this particle must be promoted to the robust
         (blend-then-inverse) local27 path.  Only the local27 sampler is
         affected (sinterp_mode 0) and only when adaptive_interp is on.
         The flag is STICKY: once set it stays 1 for the particle's life
         (simpler and safer than hysteresis).  Criteria
         (par Geodesic::strong_curv_crit):
           "u_dot_u" (default): trigger when the raw (pre-renorm)
             u·u exceeds Geodesic::u_dot_u_threshold (default -0.01).
             u·u should be -1; drift toward 0/positive means the cheap
             interpolation is degrading the norm -> unreliable here.
           "alpha": trigger when the lapse alpha = 1/sqrt(-g^00) is below
             Geodesic::alpha_threshold (default 0.5), i.e. deep in the
             potential well near a horizon / puncture.
         (kimi2.txt recommends the lapse as the physically-motivated choice
         and the u·u residual as a parameter-free fallback; both are
         available, u·u is the default per the user's request.) */
      {
        ParticleGeodesic *pg = &gstate.pgeodesic[mi];
        if (gstate.sinterp_mode == 0 && gstate.sadaptive && !pg->use_robust)
        {
          int flip = 0;
          if (gstate.strong_mode == 1)  /* alpha mode */
          {
            /* lapse = 1/sqrt(-g^{00}) from the inverse metric */
            double gup[4][4], bup[3], g3[3][3], gup3[3][3];
            if (build_gup_from_gdn(gdn, NULL, bup, g3, gup3, gup)
                && gup[0][0] < 0.0)
            {
              double lapse = 1.0 / sqrt(-gup[0][0]);
              if (isfinite(lapse) && lapse < gstate.alpha_thr) flip = 1;
            }
          }
          else  /* "u_dot_u" (default): the user's literal criterion */
          {
            if (isfinite(rl) && rl > gstate.u_dot_u_thr) flip = 1;
          }
          /* velocity-component magnitude check (added discriminant): the
             4-velocity is normalized to u·u = -1, yet a particle near light
             speed has |u^0|,|u^i| >> 1.  A positive par threshold flags an
             abnormal (unphysical) component and promotes to the robust path.
             0 = that component's check disabled (default).  u here is the
             normalized 4-velocity carried from the previous step's end-of-step
             renorm. */
          {
            if (gstate.u0_thr > 0.0 && fabs(u[0]) > gstate.u0_thr) flip = 1;
            if (gstate.ui_thr > 0.0 &&
                (fabs(u[1]) > gstate.ui_thr || fabs(u[2]) > gstate.ui_thr ||
                 fabs(u[3]) > gstate.ui_thr))
              flip = 1;
          }
          if (flip)
          {
            pg->use_robust = 1;
            if (getenv("GEODESIC_DBG_FLIP"))
              fprintf(stderr, "[FLIP] m=%d t=%.6f pos=(%.6f %.6f %.6f) rl=%.9g "
                              "u=(%.9g %.9g %.9g %.9g) u0_thr=%.9g ui_thr=%.9g\n",
                      mi, ys[0], ys[1], ys[2], ys[3], rl,
                      u[0], u[1], u[2], u[3], gstate.u0_thr, gstate.ui_thr);
            /* recompute the metric in the (now-robust) frame so the
               normalization below happens consistently */
            for (int mu=0;mu<4;mu++)
              for (int nu=0;nu<4;nu++)
                gdn[mu][nu] = 0.0;
            metric_at_point(mi, ys, ys[0], gdn);
            /* raw rl in the new frame (full contraction uc_mu u^mu) */
            double uc[4] = {0,0,0,0};
            for (int mu=0;mu<4;mu++)
              for (int nu=0;nu<4;nu++)
                uc[mu] += gdn[mu][nu]*u[nu];
            rl = 0.0;
            for (int mu=0;mu<4;mu++)
              rl += uc[mu]*u[mu];
          }
        }
      }

      if (!isfinite(rl) || rl >= 0.0)
      {
        /* [DBG] GEODESIC_DBG_DROP: dump the bad particle + its metric */
        if (getenv("GEODESIC_DBG_DROP"))
        {
          static int s_dropdbg = 0;
          if (s_dropdbg < 40)
          {
            fprintf(stderr, "[DROP_DBG] m=%d pos=(%.6f %.6f %.6f) u=(%.8e %.8e %.8e %.8e) rl=%.15e use_robust=%d\n",
                    mi, ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], rl,
                    (int)gstate.pgeodesic[mi].use_robust);
            for (int mu=0; mu<4; mu++)
              for (int nu=0; nu<4; nu++)
                fprintf(stderr, "  g[%d][%d]=%.15e ", mu, nu, gdn[mu][nu]);
            fprintf(stderr, "\n");
            s_dropdbg++;
          }
        }
        gstate.pgeodesic[mi].renorm_warn = 1;   /* aggregated after the loop */
      }
      else
      {
        const double u_spatial_init[3] = { ys[5], ys[6], ys[7] };
        const double u0_predictor = ys[4];     /* u^0 already in the particle state: continuity anchor */
        double u0_solved;

        if (solve_u0_future_directed(gdn, u_spatial_init, u0_predictor, &u0_solved))
        {
          ys[4] = u0_solved;
          particle_ttv[mi] = ys[4];
        }
        else
        {
          gstate.pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
          gstate.pgeodesic[mi].outbd = 1;   /* let the loss policy below handle this particle */
        }
      }
      u[0] = ys[4]; u[1] = ys[5]; u[2] = ys[6]; u[3] = ys[7];
      u_cov[0] = 0; u_cov[1] = 0; u_cov[2] = 0; u_cov[3] = 0;
      for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
          u_cov[mu] += gdn[mu][nu]*u[nu];
      rl = 0.0;
      for (int mu=0; mu<4; ++mu) rl += u_cov[mu]*u[mu];

      gstate.pgeodesic[mi].rl_last = rl;   /* reported after the loop */

      /* step-start velocity-component check: the same discriminant as the
         end-of-step check below, but evaluated on the just-renormalized
         step-start 4-velocity u[].  The step has not started yet, so the
         promotion needs no re-do -- the robust frame is simply used from
         this step on (sticky). */
      if (gstate.sinterp_mode == 0 && gstate.sadaptive
          && !gstate.pgeodesic[mi].use_robust)
      {
        int vel_abn = 0;
        if (gstate.u0_thr > 0.0 && fabs(u[0]) > gstate.u0_thr)
          vel_abn = 1;
        if (gstate.ui_thr > 0.0 &&
            (fabs(u[1]) > gstate.ui_thr || fabs(u[2]) > gstate.ui_thr ||
             fabs(u[3]) > gstate.ui_thr))
          vel_abn = 1;
        if (vel_abn)
        {
          gstate.pgeodesic[mi].use_robust = 1;
          if (getenv("GEODESIC_DBG_FLIP"))
            fprintf(stderr, "[FLIP] m=%d t=%.6f pos=(%.6f %.6f %.6f) "
                            "u=(%.9g %.9g %.9g %.9g) rl=%.9g (step-start vel-comp)\n",
                    mi, ys[0], ys[1], ys[2], ys[3], u[0], u[1], u[2], u[3], rl);
        }
      }
    }

    /* start-of-step snapshot (taken after the step-start renorm): if the
       end-of-step renorm discovers that the cheap Christoffel path broke
       down inside this Cactus step (raw u.u >= 0 / non-finite), the
       particle is promoted to the robust path and the step is redone from
       this snapshot instead of dropping the particle */
    CCTK_REAL ystart[8];
    memcpy(ystart, ys, sizeof(ystart));
    const double ts_start = particle_ttw[mi];

    ts = particle_ttw[mi];

    /* temporary debug (gated by GEODESIC_DEBUG_STEP, particle 0 only) */
    {
      static int dbg_step = -1, n_dbg_step = 0;
      if (dbg_step < 0) dbg_step = (getenv("GEODESIC_DEBUG_STEP") != NULL);
      if (dbg_step && mi == 0 && n_dbg_step < 80)
      {
        n_dbg_step++;
        fprintf(stderr,
          "STEP_DEBUG enter ts=%.17g ys=(%.17g,%.17g,%.17g,%.17g) u=(%.17g,%.17g,%.17g,%.17g) "
          "sstep=%.17g t_end=%.17g bg=%.17g lvl=%d outbd=%d rl_last=%.17g\n",
          ts, ys[0], ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7],
          (double)gstate.sstep, t_end, (double)odedata.bg,
          (int)gstate.pgeodesic[mi].level, (int)gstate.pgeodesic[mi].outbd,
          gstate.pgeodesic[mi].rl_last);
      }
    }

    /* integrate in tau until coordinate time reaches target */

  retry_integration:   /* end-of-step promotion redoes the step from ystart */
    ;
    const double t_target = t_end;
    CCTK_REAL h = hstart;
    int retc = GSL_SUCCESS, max_iter = 0;

    while (ys[0] < t_target && max_iter < 100002)
    {
      max_iter++;
      /* save the previous state for back-interpolation after an overshoot */
      CCTK_REAL yprev[8];
      const CCTK_REAL ts_prev = ts;
      memcpy(yprev, ys, sizeof(ys));

      /* Predict the tau_end that lands t as close to t_target as possible */
      double tau_end = ts + gstate.sstep; /* fallback cap; normally overwritten below */

      if (ys[0] + h*ys[4] > t_target)  /* coarse check with the current step size: will we overshoot? */
      {
        const double dt = t_target - ys[0];

        if (fabs(odedata.a) > epsabs)
        {
          /* solve for dtau from t(tau) ~ t0 + u^t dtau + 0.5 a dtau^2 */
          double disc = ys[4]*ys[4] + 2.0*odedata.a*dt;

          if (disc >= 0.0)
          {
            disc = sqrt(disc);
            const double d1 = (-ys[4] + disc)/odedata.a;
            const double d2 = (-ys[4] - disc)/odedata.a;

            /* pick the more sensible root: prefer positive roots;
               fall back to the linear estimate if neither is positive */
            double dtau = 0.0;
            if (d1 > 0.0 && d2 > 0.0) dtau = (d1 < d2 ? d1 : d2);
            else if (d1 > 0.0)        dtau = d1;
            else if (d2 > 0.0)        dtau = d2;
            else                      dtau = dt/ys[4];

            tau_end = ts + dtau;
          }
          else
          {
            /* negative discriminant: use the linear approximation */
            tau_end = ts + dt/ys[4];
          }
        }
        else
        {
          /* a too small: linear approximation */
          tau_end = ts + dt/ys[4];
        }
      }

      /* take one integration step capped at tau_end (key: never use a far-away cap) */
      retc = gsl_odeiv2_evolve_apply(evolve, control, gstep, &sys, &ts, tau_end, &h, ys);

      if (retc != GSL_SUCCESS)
      {
        /* Phase B: the only non-genuine ODE failure we emit on purpose
           is GSL_EBADFUNC from a detected block crossing (func_ode set
           crossed=1 and stored the transformed state in crossed_y[]).
           In that case the step is abandoned and the ODE is restarted
           from the transformed state in the NEW block, at the time at
           which the crossing was detected.  Any other failure is a real
           numerical out-of-bounds. */
        if (gstate.pgeodesic[mi].crossed)
        {
          gstate.pgeodesic[mi].crossed = 0;
          memcpy(ys, gstate.pgeodesic[mi].crossed_y, 8 * sizeof(double));
          ts = ys[0];          /* time at the crossing */
          h  = hstart;         /* re-converge the step size */
          continue;           /* loop recomputes tau_end from the new ts */
        }
        gstate.pgeodesic[mi].outbd = 1;
        ys[0] = t_target;
        break;
      }

      /* temporary debug (gated by GEODESIC_DEBUG_STEP, particle 0 only) */
      {
        static int dbg_tau = -1, n_dbg_tau = 0;
        if (dbg_tau < 0) dbg_tau = (getenv("GEODESIC_DEBUG_STEP") != NULL);
        if (dbg_tau && mi == 0 && n_dbg_tau < 80)
        {
          n_dbg_tau++;
          fprintf(stderr,
            "STEP_DEBUG iter=%d retc=%d ts=%.17g ys0=%.17g pos=(%.17g,%.17g,%.17g) outbd=%d\n",
            n_dbg_tau, (int)retc, ts, ys[0], ys[1], ys[2], ys[3],
            (int)gstate.pgeodesic[mi].outbd);
        }
      }

      const double r2 = ys[1]*ys[1] + ys[2]*ys[2] + ys[3]*ys[3];
      if (r2 < excised_radius * excised_radius)
      {
        gstate.pgeodesic[mi].outbd = 1;
      }

      if (gstate.pgeodesic[mi].outbd)
      {
        /* CCTK_VInfo(CCTK_THORNSTRING, "particle %d cross the boundary.", mi); */
        ys[0] = t_target;
        break;
      }

      /* if we still overshoot t_target, do one linear back-interpolation
         between the previous state (yprev) and the new state to align
         the particle exactly with t_target */
      if (ys[0] > t_target)
      {
        const double denom = (ys[0] - yprev[0]);
        if (fabs(denom) > 0.0)
        {
          const double s = (t_target - yprev[0]) / denom;
          ts = ts_prev + (ts - ts_prev)*s;
          for (int k=0;k<8;k++)
            ys[k] = yprev[k] + (ys[k] - yprev[k])*s;
        }
        ys[0] = t_target;
      }

      /* =========================================================================
         Renormalize 4-velocity u^mu at end of this Cactus step so that:
           u_mu u^mu = -1   and   u^t > 0
         ========================================================================= */
      {
        double gdn[4][4];
        for (int mu=0; mu<4; ++mu)
          for (int nu=0; nu<4; ++nu)
            gdn[mu][nu] = 0.0;

        /* 4-metric at the (mid-step) particle position; bg is the
           start of this particle's Cactus step (see metric_at_point) */
        metric_at_point(mi, ys, odedata.bg, gdn);

        /* compute rl = u_mu u^mu */
        double u[4] = { ys[4], ys[5], ys[6], ys[7] };
        double u_cov[4] = {0.0, 0.0, 0.0, 0.0};

        for (int mu=0; mu<4; ++mu)
          for (int nu=0; nu<4; ++nu)
            u_cov[mu] += gdn[mu][nu] * u[nu];

        double rl = 0.0;
        for (int mu=0; mu<4; ++mu)
          rl += u_cov[mu] * u[mu];

        if (!isfinite(rl) || rl >= 0.0)
        {
          /* end-of-step failure on the cheap local27 path: the step-start
             strong-curvature criterion cannot see a breakdown that develops
             inside the GSL integration (u.u is renormalized to -1 at every
             step start, so the step-start raw u.u always looks healthy),
             hence promote here and redo the step on the robust path; a
             failure on the robust path falls through to the loss policy */
          if (gstate.sinterp_mode == 0 && gstate.sadaptive
              && !gstate.pgeodesic[mi].use_robust)
          {
            gstate.pgeodesic[mi].use_robust = 1;
            if (getenv("GEODESIC_DBG_FLIP"))
              fprintf(stderr, "[FLIP] m=%d t=%.6f pos=(%.6f %.6f %.6f) rl=%.9g (end-of-step retry)\n",
                      mi, ys[0], ys[1], ys[2], ys[3], rl);
            memcpy(ys, ystart, sizeof(ys));
            ts = ts_start;
            odedata.a = 0.0;
            gsl_odeiv2_evolve_reset(evolve);
            gsl_odeiv2_step_reset(gstep);
            goto retry_integration;
          }
          /* [DBG] GEODESIC_DBG_DROP: dump the bad particle + its metric */
          if (getenv("GEODESIC_DBG_DROP"))
          {
            static int s_dropdbg = 0;
            if (s_dropdbg < 40)
            {
              fprintf(stderr, "[DROP_DBG] m=%d pos=(%.6f %.6f %.6f) u=(%.8e %.8e %.8e %.8e) rl=%.15e\n",
                      mi, ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], rl);
              for (int mu=0; mu<4; mu++)
                for (int nu=0; nu<4; nu++)
                  fprintf(stderr, "  g[%d][%d]=%.15e ", mu, nu, gdn[mu][nu]);
              fprintf(stderr, "\n");
              s_dropdbg++;
            }
          }
          gstate.pgeodesic[mi].outbd = 1;
          gstate.pgeodesic[mi].renorm_warn = 1;   /* aggregated after the loop */
        }
        else
        {
          /* gradual-drift trigger: the raw end-of-step u.u drifts away
             from -1 while the cheap path degrades, long before it goes
             non-negative; promote early (takes effect from the next Cactus
             step, the current one renormalizes fine) */
          if (gstate.sinterp_mode == 0 && gstate.sadaptive
              && !gstate.pgeodesic[mi].use_robust
              && rl > gstate.u_dot_u_thr)
          {
            gstate.pgeodesic[mi].use_robust = 1;
            if (getenv("GEODESIC_DBG_FLIP"))
              fprintf(stderr, "[FLIP] m=%d t=%.6f pos=(%.6f %.6f %.6f) rl=%.9g (end-of-step drift)\n",
                      mi, ys[0], ys[1], ys[2], ys[3], rl);
          }
          const double u_spatial_step[3] = { ys[5], ys[6], ys[7] };
          const double u0_predictor = ys[4];    /* u^0 just produced by the ODE solver, before renormalization: continuity anchor */
          double u0_solved;

          if (solve_u0_future_directed(gdn, u_spatial_step, u0_predictor, &u0_solved))
          {
            ys[4] = u0_solved;
          }
          else
          {
            gstate.pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
            gstate.pgeodesic[mi].outbd = 1;
          }
        }

        /* recompute rl after renormalization */
        u[0] = ys[4]; u[1] = ys[5]; u[2] = ys[6]; u[3] = ys[7];
        u_cov[0] = 0.0; u_cov[1] = 0.0; u_cov[2] = 0.0; u_cov[3] = 0.0;

        for (int mu=0; mu<4; ++mu)
          for (int nu=0; nu<4; ++nu)
            u_cov[mu] += gdn[mu][nu] * u[nu];

        rl = 0.0;
        for (int mu=0; mu<4; ++mu)
          rl += u_cov[mu] * u[mu];

        gstate.pgeodesic[mi].rl_last = rl;   /* reported after the loop */

        /* velocity-component magnitude check (added discriminant): even though
           u·u is renormalized to -1, an abnormal (near-light-speed) velocity
           component indicates the cheap path is degrading inside the step.  If
           any enabled component threshold is exceeded, redo this Cactus step
           from the step-start snapshot on the robust path (same mechanism as
           the rl>=0 re-do) and stick to the robust path.  The !use_robust
           guard stops a re-fire loop: after the redo the robust path is active
           and a repeat fall-through goes to the loss policy. */
        {
          int vel_abn = 0;
          if (gstate.u0_thr > 0.0 && fabs(u[0]) > gstate.u0_thr)
            vel_abn = 1;
          if (gstate.ui_thr > 0.0 &&
              (fabs(u[1]) > gstate.ui_thr || fabs(u[2]) > gstate.ui_thr ||
               fabs(u[3]) > gstate.ui_thr))
            vel_abn = 1;
          if (gstate.sinterp_mode == 0 && gstate.sadaptive
              && !gstate.pgeodesic[mi].use_robust && vel_abn)
          {
            gstate.pgeodesic[mi].use_robust = 1;
            if (getenv("GEODESIC_DBG_FLIP"))
              fprintf(stderr, "[FLIP] m=%d t=%.6f pos=(%.6f %.6f %.6f) "
                              "u=(%.9g %.9g %.9g %.9g) rl=%.9g (end-of-step vel-comp)\n",
                      mi, ys[0], ys[1], ys[2], ys[3], u[0], u[1], u[2], u[3], rl);
            memcpy(ys, ystart, sizeof(ys));
            ts = ts_start;
            odedata.a = 0.0;
            gsl_odeiv2_evolve_reset(evolve);
            gsl_odeiv2_step_reset(gstep);
            goto retry_integration;
          }
        }
      }

      if (fabs(ys[0] - t_target) < 1e-13 * fmax(1.0, fabs(t_target)))
        break;
    }
    if (max_iter > 100000)
    {
      gstate.pgeodesic[mi].outbd = 1;
      gstate.pgeodesic[mi].freeze_warn = 1;   /* aggregated after the loop */
    }

    /* write back particle state */
    particle_ttw[mi] = ts;
    particle_tt[mi]  = ys[0];
    particle_tx[mi]  = ys[1];
    particle_ty[mi]  = ys[2];
    particle_tz[mi]  = ys[3];
    particle_ttv[mi] = ys[4];
    particle_txv[mi] = ys[5];
    particle_tyv[mi] = ys[6];
    particle_tzv[mi] = ys[7];

    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(gstep);
}

#ifdef CCTK_MPI
/* multi_amr: (re-)compute the interior-box owner map from the
   gathered per-rank per-level boxes (Phase 1b): a particle is owned
   by the FIRST rank (in rank order) whose interior box of the
   FINEST level containing it (half-open: p >= lb && p < ub, fresh
   boxes only) holds it; a particle in no box (possible only if a
   level's data is stale) is pinned to rank 0, level -1, and
   flagged outbd -- the loss policy then handles it, so a particle
   is never silently dropped.  Every rank computes the identical map
   from the identical state.  Called at the step start (Phase 1) and
   at every chunk boundary of the chunked integration (re-owning);
   the boxes are fixed during the step, so the Phase 1 gathered
   buffer is reused. */
static void geodesic_recompute_owner_map(cGH *cctkGH, const double *bd, const int npr, const int nlev)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS
  int mi;
  for (mi = 0; mi < particle_n_total; mi++)
  {
    int owner = -1, olvl = -1;
    if (!gstate.pgeodesic[mi].dead)
      for (int l = nlev - 1; l >= 0 && owner < 0; l--)
        for (int r = 0; r < npr && owner < 0; r++)
        {
          const double *b = bd + (size_t)r * nlev * 8 + (size_t)l * 8;
          if (b[6] == 0.0) continue;
          if (particle_tx[mi] >= b[0] && particle_tx[mi] < b[3] &&
              particle_ty[mi] >= b[1] && particle_ty[mi] < b[4] &&
              particle_tz[mi] >= b[2] && particle_tz[mi] < b[5])
          { owner = r; olvl = l; }
        }
    if (owner < 0)
    {
      owner = 0;
      gstate.pgeodesic[mi].outbd = 1;
    }
    gstate.owner_of[mi]  = owner;
    gstate.owner_lvl[mi] = olvl;
  }
}
#endif

/* ------------------------------------------------------------------------------------------------
     geodesics_integrate: acquire the single-grid data, integrate all
     particles, and apply the loss policy to out-of-bounds / bad-norm
     particles.
------------------------------------------------------------------------------------------------- */

void geodesics_integrate (CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  int i, mi, loss_policy;
  double *bd = NULL;   /* multi_amr: gathered interior boxes (Phase 1); kept until the chunk loop frees it */
  const int lvl = (int)lrint(log2((double)cctk_levfac[0]));

  /* The particle-state diagnostic dump -- exactly one block per
     cctk_iteration, in time order, particles in index order (so a plotting
     program sees, for every particle, exactly one record per point,
     chronologically) -- is written AFTER the ODE loop below, on the
     finest-level call (the only call that integrates).  Each block therefore
     holds the POST-integration state (the particle's position at the end of
     the step).  last_dump_iter guarantees one block per iteration even
     though several levels report each iteration; rank 0 only; gated by
     particle_dump_every (N -> one block per N iterations); cadence is
     independent of CarpetRegrid2::regrid_every.  See the block after the
     multi_amr S2 sync. */

  /* must be set before the acquire block below (which branches on it) */
  gstate.sexact = Exact;
  /* Multi-process Exact runs: the global particle state used to live on
     rank 0 (OPTIONS: global); the AMR rewrite made the components
     local.  Keep the same observable behavior: only rank 0 integrates
     the particles and writes the dump, and the level-selection checks
     are restricted to level 0 (each rank only owns its own local grid
     patch, so a cross-level safe-region test would be meaningless). */
  gstate.multi_exact = (CCTK_nProcs(cctkGH) > 1) && Exact;

  /* Multi-process AMR (multi_amr): every rank integrates the particles
     whose position lies inside its own local patch (interior-box
     ownership); the shared particle state is synchronized with
     MPI_Allgather so every rank holds the identical full state.
     Only a 2x2 (y,z) decomposition of at most 4 ranks is supported. */
  gstate.multi_amr = (CCTK_nProcs(cctkGH) > 1) && (geodesic_amr_levels > 1);
  if (gstate.multi_amr && CCTK_nProcs(cctkGH) > 4)
    CCTK_ERROR("Geodesic (multi-process AMR): at most 4 MPI ranks are supported.");

  /* ================================================================================
     AMR acquire phase.

     Carpet traverses CCTK_POSTSTEP in level mode: this routine is called
     once per refinement level (per component).  With a single process
     and one centered patch per level there is exactly one call per
     level per iteration, so after all levels have been acquired the
     particles are integrated once (on the finest-level call).  On all
     earlier calls we just acquire and return.  A single-level run is
     one call per iteration and integrates immediately.
     ================================================================================ */
  {
    const int level = (int)lrint(log2((double)cctk_levfac[0]));
    /* Guard: in a single-patch (unigrid/AMR) run the MultiPatch thorn is not
       active, so MultiPatch_GetMap's alias is unregistered; treat map as 0.
       In a genuine multipatch run it returns the current map index. */
    const int map = CCTK_IsFunctionAliased("MultiPatch_GetMap")
                    ? MultiPatch_GetMap(cctkGH)
                    : 0;
    LevelData *L;
    /* GEODESIC_DBG_CALL: trace every acquire call (map, level) */
    if (getenv("GEODESIC_DBG_CALL"))
      fprintf(stderr, "GDCALL it=%d map=%d level=%d nlev_seen=%d l0_seen=%d "
                       "last_integ=%d nproc=%d MeProc=%d multi_amr=%d\n",
              (int)cctk_iteration, map, level, gstate.nlev_seen,
              gstate.l0_seen_iter, gstate.last_integ_iter,
              CCTK_nProcs(cctkGH), CCTK_MyProc(cctkGH), (int)gstate.multi_amr);
    if (level >= GEODESIC_MAX_LEVELS)
      return;   /* more levels than LD[] holds (GEODESIC_MAX_LEVELS): ignore */

    if (level < 0)
      CCTK_WARN(CCTK_WARN_ABORT,
                "Geodesic: negative refinement level is invalid.");
    /* Levels beyond geodesic_amr_levels are also acquired and tracked in
       nlev_seen so the trigger can fire on the finest level actually present
       (subcycling / regrid).  geodesic_amr_levels still only controls the
       particle's level choice (samr_levels), unchanged. */

    /* With time subcycling, Carpet calls CCTK_POSTSTEP once per level per
       substep; the trigger logic at the end of this block integrates
       only at the moment all levels hold data at the same time. */

    L = &gstate.LD[map][level];
    for (i=0; i<3; i++)
    {
      L->lsh[i]         = cctk_lsh[i];
      L->lbnd[i]        = cctk_lbnd[i];
      L->ubnd[i]        = cctk_ubnd[i];
      L->nghostzones[i] = cctk_nghostzones[i];
      L->levfac[i]      = cctk_levfac[i];

      /* In Carpet's level-mode POSTSTEP traversal, cctk_origin_space and
         cctk_delta_space keep the coarse-grid values on every level;
         only lbnd/ubnd/lsh/levfac vary per level.  Per-level geometry
         reconstruction:
           dx_lev = dx_coarse / levfac
           x(0)   = physical coordinate of array index 0
         x(0) is reconstructed with the same convention used by the other
         thorns that locate grid cells (see SphericalSurface_SetupRes,
         "physical(i) = CCTK_ORIGIN_SPACE(d) + (cctk_lbnd[d] + i) *
          CCTK_DELTA_SPACE(d)"), so array index 0 is
           x(0) = CCTK_ORIGIN_SPACE(i) + cctk_lbnd[i] * CCTK_DELTA_SPACE(i)
         with CCTK_DELTA_SPACE = cctk_delta_space/levfac = dx_lev and
         CCTK_ORIGIN_SPACE = cctk_origin_space + dx_lev*levoff/levoffdenom.
         This is level-independent (level 0 has lbnd=0, levoff=0, so
         x(0)=cctk_origin_space) and patch-independent (centred 1p, off-
         centre 1p, and per-rank multi-process quadrant patches alike).  The
         alignment check below is informational only: an un-aligned 1p patch
         is handled correctly by the lbnd-based origin, so we warn (not
         abort) so 1p AMR runs can be diagnosed. */
      L->dx[i]     = cctk_delta_space[i] / cctk_levfac[i];
      L->inv_dx[i] = 1.0 / L->dx[i];
      /* origin = physical position of the patch's array index 0.

         Mirrors the convention used by the other thorns that locate grid
         cells in the GW150914 run.  SphericalSurface_SetupRes computes the
         physical position of local grid index j as
            physical(j) = CCTK_ORIGIN_SPACE(d) + (cctk_lbnd[d] + j)
                          * CCTK_DELTA_SPACE(d)
         so array index 0 (which is what L->origin must be) is at
            L->origin[d] = CCTK_ORIGIN_SPACE(d) + cctk_lbnd[d] * CCTK_DELTA_SPACE(d)
         with
            CCTK_DELTA_SPACE(d) = cctk_delta_space[d]/cctk_levfac[d] = L->dx[d],
            CCTK_ORIGIN_SPACE(d) = cctk_origin_space[d] + L->dx[d]
                                   * cctk_levoff[d]/cctk_levoffdenom[d].
         This is level-independent (level 0 has lbnd=0, levoff=0 ->
         cctk_origin_space) and patch-independent: it is the index-0 position
         of whichever patch Carpet hands this rank, whether a full centred
         patch (1p), an off-centre patch, or a per-rank quadrant of a
         multi-process AMR run.  It uses only geometry already in the cGH,
         so no amr_center/amr_radius or lbnd/ngz reconstruction is needed. */
      L->origin[i] = CCTK_ORIGIN_SPACE(i) + (CCTK_REAL)cctk_lbnd[i] * L->dx[i];

      {
        /* temporary debug: dump the raw cGH geometry and the derived
           per-level geometry once per axis (gated by GEODESIC_DEBUG_GEOM) */
        static int dbg_geom = -1;
        if (dbg_geom < 0) dbg_geom = (getenv("GEODESIC_DEBUG_GEOM") != NULL);
        if (dbg_geom && (level == 0) &&
            ((int)cctk_iteration == 0 || (int)cctk_iteration == 128))
        {
          const double xfirst = L->origin[i];
          const double xlast  = L->origin[i] + (cctk_lsh[i]-1)*L->dx[i];
          fprintf(stderr,
            "GEO_DEBUG lvl=%d ax=%d it=%d cctk_origin=%.17g cctk_delta=%.17g "
            "lbnd=%d ubnd=%d lsh=%d ngz=%d levfac=%d cctk_dt=%.17g "
            "-> L_origin=%.17g L_dx=%.17g xfirst=%.17g xlast=%.17g xfirst+xlast=%.6g\n",
            level, i, (int)cctk_iteration,
            (double)cctk_origin_space[i], (double)cctk_delta_space[i],
            (int)cctk_lbnd[i], (int)cctk_ubnd[i], (int)cctk_lsh[i],
            (int)cctk_nghostzones[i], (int)cctk_levfac[i], (double)cctk_delta_time,
            xfirst, L->dx[i], xfirst, xlast, xfirst + xlast);
        }
      }

      /* Alignment check (single-process fine levels): the local array
         holds the regrid box plus ghost zones on both sides, so the
         interior box spans nint = lsh - 2*ngz cells.  For a patch that
         aligns with the coarse grid AND is centred on the origin, nint must
         be a multiple of levfac (divisibility) and nint/levfac even (centred).
         Off-centre AMR patches (e.g. GW150914 double-centre) and
         mis-aligned regrid boxes fail one or both tests.  The patch origin
         is computed from the real lbnd (see above), so an un-aligned 1p
         patch is handled correctly regardless.  These diagnostics are
         expected for per-hole AMR and are emitted only when
         Geodesic::amr_align_warn="yes" (off by default), so they do not
         flood the log on every acquire.  Level 0 is exempt (a plain
         unigrid domain need not be centered). */
      if (amr_align_warn)
      {
        if (level > 0 && CCTK_nProcs(cctkGH) == 1)
        {
          const int nint = (int)cctk_lsh[i] - 2 * (int)cctk_nghostzones[i];
          if (nint <= 0 || nint % cctk_levfac[i] != 0)
            CCTK_WARN(CCTK_WARN_ALERT,
                      "Geodesic: fine-level interior box (lsh - 2*ngz) is not "
                      "a multiple of levfac; the patch does not align with the "
                      "coarse grid.  Origin taken from lbnd (un-aligned 1p "
                      "patch, handled).");
          if (nint > 0 && (nint / cctk_levfac[i]) % 2 != 0)
            CCTK_WARN(CCTK_WARN_ALERT,
                      "Geodesic: fine-level interior box is not symmetric about "
                      "the origin (nint/levfac odd).  Origin taken from lbnd "
                      "(off-centre 1p patch, handled).");
        }
      }

      /* 7-cell safe margin: the 3x3x3 stencil plus 6th-order FD
         padding must stay inside the level (used by the particle
         safe-region check in both the exact and the interpolated
         branch). */
      L->r_lbnd[i]  = L->origin[i] + 7*L->dx[i];
      L->r_ubnd[i]  = L->origin[i] + (cctk_lsh[i]-8)*L->dx[i];

      /* interior box of this rank's local patch (multi-process AMR
         ownership): the physical region, half-open on the upper end,
         so the interior boxes of all ranks at a level tile the
         refinement region exactly */
      L->i_lbnd[i] = L->origin[i] + (cctk_nghostzones[i] - 0.5) * L->dx[i];
      L->i_ubnd[i] = L->origin[i] + (cctk_lsh[i] - cctk_nghostzones[i] - 0.5) * L->dx[i];
    }

    L->dt   = cctk_delta_time;
    L->time = cctk_time;

    /* ---------------- pointers to ADMBase (this level) ----------------
       Fetched only for the interpolated branch (Exact = no): the exact
       branch uses the built-in analytic Kerr-Schild metric and never
       reads the grid metric.  The variables are still declared as
       READS in schedule.ccl, so ADMBase must be present in the run
       either way (see README, "Required thorns"). */
    if (!gstate.sexact)
    {
      if (cctk_nghostzones[0] < 4 || cctk_nghostzones[1] < 4 || cctk_nghostzones[2] < 4)
        CCTK_ERROR("The ghost_size less 4 (need at least 4 for 6th-order FD).");

      {
        /* all 10 metric fields at all three time levels:
           [0] = now, [1] = t - dt, [2] = t - 2dt (driver convention) */
        static const char *field[10] = { "ADMBASE::gxx", "ADMBASE::gxy", "ADMBASE::gxz",
                                         "ADMBASE::gyy", "ADMBASE::gyz", "ADMBASE::gzz",
                                         "ADMBASE::alp", "ADMBASE::betax", "ADMBASE::betay",
                                         "ADMBASE::betaz" };
        CCTK_REAL **slot[10] = { L->gxx, L->gxy, L->gxz, L->gyy, L->gyz, L->gzz,
                                 L->alp, L->betax, L->betay, L->betaz };
        int fidx, tl;
        int have0 = 1, havepast = 1;
        for (fidx = 0; fidx < 10; fidx++)
          for (tl = 0; tl < 3; tl++)
          {
            int vi = CCTK_VarIndex(field[fidx]);
            slot[fidx][tl] = (CCTK_REAL *)(CCTK_VarDataPtrI(cctkGH, tl, vi));
            if (slot[fidx][tl] == NULL)
            {
              if (tl == 0) have0 = 0;
              else         havepast = 0;
            }
          }
        L->timelevels_ok = havepast;
        if (!have0)
          CCTK_ERROR("Geodesic (Exact=no): ADMBase metric/lapse/shift not stored; check the run configuration.");
        if (!havepast)
        {
          L->metric_static = 1;
          if (!gstate.pastlevels_warned)
          {
            gstate.pastlevels_warned = 1;
            CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
              "Geodesic (Exact=no): ADMBase past time levels (1,2) are not readable; "
              "assuming a static metric (d_t g = 0). For time-varying grid metrics set "
              "Carpet::init_fill_timelevels = \"yes\" and keep the ADMBase metric fields "
              "at timelevels >= 3.");
          }
        }
        else
        {
          /* The oldest stored time level (index 2) still holds
             allocation garbage until two time advances have occurred
             (these pars do not set Carpet::init_fill_timelevels, so
             Carpet leaves timelevels 1-2 untouched).  metric_is_static
             compares level 0 against level 2, and the d=0 branch of
             ndg4dn_tl reads all three levels, so before maturity the
             result would be garbage (NaN / 1.968e+243) and poison the
             Christoffel array on the very first step.  Until then fall
             back to d_t g = 0, as the code did before the
             3-time-level rework. */
          if (cctk_iteration < 2)
            L->metric_static = 1;
          else
            L->metric_static = metric_is_static(L);
        }
      }
    }
    else
    {
      L->timelevels_ok = 0;
      L->metric_static = 1;
    }

    /* temporary debug: time-level sanity (gated by GEODESIC_DEBUG_FIT) */
    {
      static int dbg_fit = -1;
      if (dbg_fit < 0) dbg_fit = (getenv("GEODESIC_DEBUG_FIT") != NULL);
      if (dbg_fit && !gstate.sexact)
        fprintf(stderr, "FIT_DEBUG acquire lvl=%d it=%d static=%d ok=%d "
                "gxx[0][0]=%.17g gxx[1][0]=%.17g gxx[2][0]=%.17g "
                "gxx[0][5000]=%.17g gxx[1][5000]=%.17g gxx[2][5000]=%.17g\n",
                level, (int)cctk_iteration, (int)L->metric_static, (int)L->timelevels_ok,
                (double)L->gxx[0][0], (double)L->gxx[1][0], (double)L->gxx[2][0],
                (double)L->gxx[0][5000], (double)L->gxx[1][5000], (double)L->gxx[2][5000]);
    }

    /* minimal ghost cGH exposing THIS level's geometry.  Only the fields
       set here are valid (everything else is zero); this lets legacy
       code such as g4dn() in derivative.h keep using
       CCTK_GFINDEX3D(gview.GH,...) against the current level's view.  It
       must NOT be used for CCTK_VarDataPtrI or any other driver API. */
    memset(&L->gh, 0, sizeof(cGH));
    L->gh.cctk_lsh          = L->lsh;
    L->gh.cctk_ash          = L->lsh;    /* CCTK_GFINDEX3D uses ash, not lsh */
    L->gh.cctk_lbnd         = L->lbnd;
    L->gh.cctk_ubnd         = L->ubnd;
    L->gh.cctk_nghostzones  = L->nghostzones;
    L->gh.cctk_levfac       = L->levfac;
    L->gh.cctk_origin_space = L->origin;
    L->gh.cctk_delta_space  = L->dx;
    L->gh.cctk_delta_time   = cctk_delta_time;
    L->gh.cctk_time         = cctk_time;
    L->gh.cctk_iteration    = cctk_iteration;

    L->stamp = cctk_iteration;
    L->valid = 1;
    gstate.ld_generation++;

    /* ---------------------------------------------------------------
       Trigger: integrate exactly once per call-batch, on the batch's
       finest-level call.

       A "batch" is one cctk_iteration's worth of POSTSTEP acquires: under
       time subcycling the finest level's POSTSTEP marks the batch (it is
       due every trf-fine substep) and coarser levels join the batch when
       their own substep is due, in coarse-to-fine order.  The trigger fires
       on the call whose level is the tracked finest (nlev_seen-1) -- the
       last call of the batch -- so each batch integrates once, advancing a
       particle by the finest level's own dt (sstep = elapsed cctk_time, set
       later).  That call's acquire has just run, so its data is fresh.

       Batch tracking (batch_iter / batch_maxlvl, init -1): a new
       cctk_iteration marks a batch boundary; at each boundary we apply the
       regrid decay -- if the previous batch never reached nlev_seen-1, that
       level was removed by a regrid and nlev_seen would otherwise wait
       forever for it; decay nlev_seen to the previous batch's actual
       highest level + 1.  nlev_seen also grows on any level first seen
       (level + 1 > nlev_seen).

       Iteration 0: the it==0 guard below waits for the finest level's first
       call so the one-shot u^mu init sees fresh LevelData on every level
       (the coarsest is acquired first).  Single-level runs
       (geodesic_amr_levels = 1) fire on their only call.
       --------------------------------------------------------------- */
    /* Batch boundary: a new cctk_iteration ends the previous batch (apply
       the regrid decay) and starts this one. */
    if (gstate.batch_iter != cctk_iteration)
    {
      if (gstate.batch_iter >= 0 && gstate.batch_maxlvl >= 0 &&
          gstate.batch_maxlvl + 1 < gstate.nlev_seen)
        gstate.nlev_seen = gstate.batch_maxlvl + 1;
      gstate.batch_iter   = cctk_iteration;
      gstate.batch_maxlvl = -1;
    }
    /* Track this batch's highest level and the global highest seen. */
    if (level > gstate.batch_maxlvl)
      gstate.batch_maxlvl = level;
    if (level + 1 > gstate.nlev_seen)
      gstate.nlev_seen = level + 1;
    if (level == 0)
      gstate.l0_seen_iter = cctk_iteration;

    if (getenv("GEODESIC_DBG_CALL"))
      fprintf(stderr, "GDTIR it=%d level=%d nlev_seen=%d batch_maxlvl=%d "
                       "last_integ=%d amr_levels=%d ->",
              (int)cctk_iteration, level, gstate.nlev_seen,
              gstate.batch_maxlvl, gstate.last_integ_iter,
              geodesic_amr_levels);
    if (level != gstate.nlev_seen-1)
      return;                               /* integrate only on the finest-level call */
    if (getenv("GEODESIC_DBG_CALL"))
      fprintf(stderr, " FIRED it=%d level=%d -> integ\n",
              (int)cctk_iteration, level);
    if (cctk_iteration == 0 && level < geodesic_amr_levels - 1)
      return;                               /* iteration 0: wait for the finest level  */
    if (gstate.last_integ_iter == cctk_iteration)
      return;                               /* already integrated in this batch        */
    gstate.last_integ_iter = cctk_iteration;
  }

  /* ================================================================================
     Multi-process policy (multi_exact only): only rank 0 integrates the
     particles and writes the dump file.  The acquire phase above still
     runs on every rank (each rank caches its own local grid view;
     harmless).  This reproduces the old global-component behavior, where
     the global component (and with it the particle arrays) lived on
     rank 0.  In multi_amr runs every rank integrates its owned subset
     (see Phase 1 below), so the gate does not apply.  With a single
     process MyProc() == 0 and this is a no-op.
     ================================================================================ */
  if (CCTK_MyProc(cctkGH) != 0 && !gstate.multi_amr)
    return;

  /* ==================== from here on: integrate particles ONCE ==================== */
  gstate.samr_levels = geodesic_amr_levels;

  /* ==================== integrate particles ==================== */

  /* ---------------- particle cache ----------------
     Allocated once for particle_n_total slots and reused at every
     Cactus step; the per-step fields (flag, outbd, norm_bad, the
     renorm/freeze diagnostics) are reset in the loop below.  The old
     per-step calloc/free churned hundreds of MB of allocation traffic
     per iteration at 10^4 particles. */
  if (gstate.pgeodesic == NULL || gstate.pgeodesic_capacity < particle_n_total)
  {
    free(gstate.pgeodesic);
    gstate.pgeodesic = (ParticleGeodesic * ) calloc(particle_n_total, sizeof(ParticleGeodesic));
    if (!gstate.pgeodesic)
      CCTK_WARN(CCTK_WARN_ABORT, "Error: calloc(pgeodesic) failed.");
    gstate.pgeodesic_capacity = particle_n_total;

    /* Phase A: all particles are pinned to the center block (map 0);
       Phase B (multi-block support) reassigns cur_map at each integration. */
    for (i = 0; i < particle_n_total; i++)
      gstate.pgeodesic[i].cur_map = 0;
  }

#ifdef CCTK_MPI
  /* ---------------- multi_amr Phase 1: interior-box owner map ----------------
     All ranks execute this (collective).  (a) Every rank publishes the
     per-level interior box of its local patch (freshness flag + static
     metric flag included).  (b) From the gathered boxes every rank
     computes the IDENTICAL owner map: a particle is owned by the rank
     whose interior box of the FINEST containing level holds it (rank
     order breaks box ties; half-open boxes make the boxes a partition).
     A particle in no box (possible only if a level's data is stale) is
     pinned to rank 0, level -1, and flagged outbd -- the loss policy
     then handles it, so a particle is never silently dropped. */
  if (gstate.multi_amr)
  {
    const int npr = CCTK_nProcs(cctkGH);
    const int me  = CCTK_MyProc(cctkGH);
    const int nlev = geodesic_amr_levels;
    if (gstate.owner_of == NULL || gstate.owner_cap < particle_n_total)
    {
      free(gstate.owner_of);
      free(gstate.owner_lvl);
      gstate.owner_of  = (int *) malloc(particle_n_total * sizeof(int));
      gstate.owner_lvl = (int *) malloc(particle_n_total * sizeof(int));
      if (!gstate.owner_of || !gstate.owner_lvl)
        CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: owner-map allocation failed.");
      gstate.owner_cap = particle_n_total;
    }

    /* Phase 1a: Allgather of per-rank per-level boxes
       (nlev * 8 doubles/rank: box_lbnd[3], box_ubnd[3], valid, metric_static).
       The owner map must use the 7-cell safe region, NOT the interior box,
       for the FINE levels: the interior box extends into the ghost region
       (where the fine GF has no data) and beyond the safe margin, so a
       particle in that band would be handed to a fine level whose 27-point
       stencil reads uninitialized cells -> NaN.  The safe region is a strict
       subset of the refinement region in which the stencil is guaranteed to
       fit.  A particle in the gap between rank safe regions (or the
       fine-level gap band) falls through to a coarser level, which is the
       correct, valid-data choice.  For the COARSEST level the safe region
       shrinks the rank quadrant and leaves an inter-rank gap that would
       make a particle unassignable, so we use the full interior box there
       (the 27-point stencil is clamped to the valid cell range in
       cf2_fitting()).  b[6] carries L->valid (acquired at least once), not
       a "fresh this iteration" flag: with subcycled AMR the trigger fires on
       the finest-level call, where only the fine levels have been acquired
       this iteration and the coarse levels are (harmlessly) stale.  A valid
       coarse level is still spatially valid; when its metric is static the
       value is time-independent, and AEI/carpet time-interpolates otherwise,
       so a stale-but-valid box is a correct owner. */
    /* assign the function-scope bd (kept until the chunk loop frees it);
       a block-local declaration here would shadow it, leaving the chunk
       loop's re-owning to dereference NULL */
    bd = (double *) malloc((size_t)npr * nlev * 8 * sizeof(double));
    if (!bd)
      CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: box Allgather buffer allocation failed.");
    {
      double *s0 = bd + (size_t)me * nlev * 8;
      for (int l = 0; l < nlev; l++)
      {
        LevelData *L = &gstate.LD[0][l];
        double *s = s0 + (size_t)l * 8;
        /* fall back to the interior box if the safe region is degenerate
           (a patch too thin in any direction has r_lbnd > r_ubnd and the
           stencil cannot fit anywhere in it anyway). */
        for (int i = 0; i < 3; i++) {
          if (l == 0) {
            /* Coarsest level: use the interior box (which tiles the domain,
               no gaps between ranks).  The 7-cell safe region shrinks the
               coarsest-level region and creates inter-rank gaps in which a
               particle would be unassigned (owner_lvl=-1).  At the coarsest
               level there is no finer level to fall through to, and the
               27-point stencil is clamped to the valid cell range in
               cf2_fitting(), so using the full interior box is safe. */
            s[i]      = L->i_lbnd[i]; s[i+3] = L->i_ubnd[i];
          } else {
            if (L->r_lbnd[i] < L->r_ubnd[i]) {
              s[i]      = L->r_lbnd[i]; s[i+3] = L->r_ubnd[i];
            } else {
              s[i]      = L->i_lbnd[i]; s[i+3] = L->i_ubnd[i];
            }
          }
        }
        s[6] = L->valid ? 1.0 : 0.0;
        s[7] = L->metric_static;
      }
      MPI_Allgather(s0, nlev*8, MPI_DOUBLE, bd, nlev*8, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /* Phase 1b: owner map (every rank computes the identical result;
       see geodesic_recompute_owner_map).  bd is kept until the chunk
       loop of the integration below frees it: the interior boxes are
       fixed during the step, so the chunk-boundary re-owning reuses
       the same gathered boxes. */
    geodesic_recompute_owner_map(cctkGH, bd, npr, nlev);
  }
#endif

  /* ---------------- set run flags / parameters ---------------- */
  gstate.sgverbose = gverbose;
  gstate.srverbose= rverbose;
  gstate.ksm = M;
  gstate.ksa = a;
  gstate.sinitial_radius = initial_radius;

  /* particle loss policy: 0 = drop, 1 = flag, 2 = respawn
     (param.ccl validates the keyword; drop is the default) */
  if (strcmp(particle_loss_policy, "drop") == 0)
    loss_policy = 0;
  else if (strcmp(particle_loss_policy, "flag") == 0)
    loss_policy = 1;
  else
    loss_policy = 2;

  /* metric_interp / time_interp (grid mode only; the Exact = yes branch
     ignores both).  The remaining linear constraints (Exact = no, not
     carpet, ADMBase metric_timelevels >= 2) are checked in ParamCheck. */
  if (strcmp(metric_interp, "aei") == 0)
    gstate.sinterp_mode = 1;
  else if (strcmp(metric_interp, "carpet") == 0)
    gstate.sinterp_mode = 2;
  else
    gstate.sinterp_mode = 0;
  gstate.stime_interp = (strcmp(time_interp, "linear") == 0) ? 1 : 0;
  gstate.sadaptive = (strcmp(adaptive_interp, "off") == 0) ? 0 : 1;
  /* strong-curvature criterion for the robust local27 path (0 = u_dot_u,
     default; 1 = alpha) */
  gstate.strong_mode = (strcmp(strong_curv_crit, "alpha") == 0) ? 1 : 0;
  gstate.u_dot_u_thr = u_dot_u_threshold;
  gstate.alpha_thr   = alpha_threshold;
  gstate.u0_thr      = u0_threshold;
  gstate.ui_thr      = ui_threshold;
  gstate.strong_christ_off = (strcmp(strong_christoffel, "l27") == 0) ? 1 : 0;
  gstate.log_metric_interp = (strcmp(log_metric_interp, "yes") == 0) ? 1 : 0;
  gstate.robust_metric_interp = (strcmp(robust_metric_interp, "fourd") == 0) ? 1
                              : (strcmp(robust_metric_interp, "fourdh") == 0) ? 2 : 0;
  gstate.smoke_eps = smoke_timedep_eps;
  gstate.cur_gh = cctkGH;

  /* Resolve the AEI interpolator handle for any non-exact grid run: the
     dedicated aei/carpet samplers (sinterp_mode 1/2) need it, and the
     local27 strong-curvature adaptive path (sinterp_mode 0, use_robust)
     also calls the same pointwise AEI order-4 interpolant.  The
     interpolators are registered at STARTUP, after PARAMCHECK. */
  if (!gstate.sexact && gstate.aei_handle < 0)
  {
    gstate.aei_handle   = CCTK_InterpHandle("Lagrange polynomial interpolation");
    gstate.aei_coordsys = CCTK_CoordSystemHandle("cart3d");
    if (gstate.aei_handle < 0 || gstate.aei_coordsys < 0)
      CCTK_WARN(CCTK_WARN_ABORT,
        "Geodesic (metric_interp = aei/carpet): cannot resolve the "
        "interpolator 'Lagrange polynomial interpolation' (AEILocalInterp) "
        "or the coordinate system 'cart3d'.");
    if (!adm_varidx_ok)
    {
      for (i=0; i<10; i++)
      {
        adm_varidx[i] = CCTK_VarIndex(adm_field[i]);
        if (adm_varidx[i] < 0)
          CCTK_WARN(CCTK_WARN_ABORT,
            "Geodesic (metric_interp = carpet): cannot resolve one of the "
            "ADMBase metric variables (gxx..gzz, lapse, shift).");
      }
      adm_varidx_ok = 1;
    }
  }

  /* default: advance by the time elapsed since the previous firing, i.e.
     the interval between consecutive finest-level calls.  Under subcycling
     that is the finest level's own dt; for a single-level run it equals
     the grid dt (bit-identical to the old level-0 value).  cctk_time is
     the newest data time of the finest level; last_integ_time is the
     previous firing's cctk_time.  The it==0 fallback (sstep<=0) uses the
     level-0 dt; the seed block returns before integration so the value is
     irrelevant there. */
  if (step == 0.0)
  {
    gstate.sstep = cctk_time - gstate.last_integ_time;
    gstate.last_integ_time = cctk_time;
    if (gstate.sstep <= 0.0)
      gstate.sstep = gstate.LD[0][0].dt;
  }
  else
    gstate.sstep = step;

  /* Time-varying grid metric: the Christoffel symbols are frozen at
     the grid's current time during the particle step, so when any
     level's metric is not static the particle step must not exceed
     the grid time step of the time-varying levels. */
  if (!gstate.sexact)
  {
    CCTK_REAL dt_min = gstate.sstep;
    CCTK_INT cap = 0;
    for (CCTK_INT lv = 0; lv < gstate.samr_levels; lv++)
    {
      if (!gstate.LD[0][lv].valid || gstate.LD[0][lv].metric_static)
        continue;
      if (gstate.LD[0][lv].dt < dt_min)
        dt_min = gstate.LD[0][lv].dt;
      cap = 1;
    }
    if (cap && gstate.sstep > dt_min)
    {
      gstate.sstep = dt_min;
      if (!gstate.step_cap_warned)
      {
        gstate.step_cap_warned = 1;
        CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
          "Geodesic: time-varying grid metric detected; particle step capped to the grid time step (%.17g).",
          dt_min);
      }
    }
    /* multi_amr: the cap above is computed from each rank's own view of
       the levels, but all ranks must agree on sstep (the shared state
       is advanced by sstep on the owning rank); take the global
       minimum.  sexact runs skip this whole block identically on all
       ranks, so no collective-communication divergence. */
#ifdef CCTK_MPI
    if (gstate.multi_amr)
    {
      double sloc = (double)gstate.sstep, sglob = 0.0;
      MPI_Allreduce(&sloc, &sglob, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      gstate.sstep = (CCTK_REAL)sglob;
    }
#endif
  }

  /* ---------------- reset per-step particle fields ----------------
     (dead is persistent: it survives across steps) */
  for (mi=0; mi<particle_n_total; mi++)
  {
    gstate.pgeodesic[mi].flag        = 0;
    gstate.pgeodesic[mi].outbd       = 0;
    gstate.pgeodesic[mi].norm_bad    = 0;
    gstate.pgeodesic[mi].renorm_warn = 0;
    gstate.pgeodesic[mi].freeze_warn = 0;
    gstate.pgeodesic[mi].rl_last     = 0.0;
    gstate.pgeodesic[mi].old_level_bad = 0;
  }

  /* ================================================================
   * Particle 4-velocity initialization
   *
   * Convention: particle_tv/xv/yv/zv in the parfile are the
   * contravariant 4-velocity components u^mu = (u^t, u^x, u^y, u^z)
   * of the particle in the simulation (Kerr-Schild) coordinates --
   * NOT a 3-velocity.
   *
   * Behavior (runs once, on the first trigger -- normally the first
   * CCTK time step; in multi_amr that is the finest-level call of
   * iteration 0, where all levels hold fresh data; see the trigger
   * guard above):
   *   - norm = g_mu_nu u^mu u^nu is evaluated at the initial position
   *     (exact Kerr-Schild metric when Exact=yes, 27-point grid
   *     interpolation otherwise).
   *   - If |norm + 1| < 1e-6 the input is already a normalized
   *     4-velocity and is used unchanged.  This makes the
   *     initialization idempotent: feeding the values logged here
   *     back into the parfile changes nothing.
   *   - Otherwise u^0 is re-solved from the normalization condition
   *     (u^i held fixed, future-directed branch) and the log shows
   *     the before/after values.
   *   - If no future-directed solution exists, a warning is issued
   *     and the parfile values are kept as-is.
   * ================================================================ */
  /* Recovery: when a checkpoint is restored, the particle positions +
     4-velocities (the particle_* arrays, checkpointable per
     interface.ccl) are already the restored, normalized state.  The
     one-shot init block must not run: it re-normalizes against the
     parfile initial state and, worse, queries LevelData that (for
     per-hole AMR) is only populated on the fine levels Carpet acquires
     this iteration -- level 0 is never acquired, so pick_level /
     in_level_dir see zero geometry and every particle is dropped.
     In a fresh run the first trigger fires at cctk_iteration == 0
     (no-op); on a recovered run cctk_iteration is the checkpoint
     iteration (> 0) on the first trigger. */
  if (cctk_iteration > 0 && !gstate.init_done)
    gstate.init_done = 1;
  if (!gstate.init_done)
  {
    if (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0)
      CCTK_VInfo(CCTK_THORNSTRING, "Kerr M:%.4g a:%.4g Particle total:%d", gstate.ksm, gstate.ksa, particle_n_total);
    if (CCTK_IsFunctionAliased("MultiPatch_GetMaps"))
      gstate.nmaps = MultiPatch_GetMaps(cctkGH);
    else
      gstate.nmaps = 1;
    for (int mi = 0; mi < particle_n_total; mi++)
    {
      /* multi_amr: only the owning rank runs the init (re-)solve for
         this particle; the S1 Allgather below distributes the result */
      if (gstate.multi_amr && gstate.owner_of[mi] != CCTK_MyProc(cctkGH))
        continue;
      double ys[8];
      ys[0] = particle_tt[mi];
      ys[1] = particle_tx[mi];
      ys[2] = particle_ty[mi];
      ys[3] = particle_tz[mi];
      /* 4-velocity taken directly from the parfile (u^mu convention) */
      ys[4] = particle_ttv[mi];
      ys[5] = particle_txv[mi];
      ys[6] = particle_tyv[mi];
      ys[7] = particle_tzv[mi];

      /* make sure the interpolation coefficients are computed */
      gstate.pgeodesic[mi].flag = 0;
      cf2_fitting(mi, ys);

      /* Metric at the initial position.
         sexact:  exact Kerr-Schild analytic metric (evaluated directly;
                  cf2_fitting performs no fitting in the sexact branch).
         !sexact: metric_at_point (local27 / aei / carpet; an interpolation
                  failure leaves gdn_i zeroed and flags the particle
                  out-of-bounds, which the norm check below reports).
         bg = ys[0] - sstep, not ys[0]: the grid's two time levels span
         [t - dt, t] and the particle sits exactly at the NEW level t, so
         the linear time blend (theta = (y[0] - bg)/sstep) must be 1 to
         land on the new level.  Passing ys[0] as bg would give theta = 0
         and evaluate the time-dependent metric at t - dt, one grid step
         too early (a one-off ~O(dt) u^t error at init for the
         time_interp = linear paths; the time_interp = off path ignores
         bg and is unaffected). */
      double gdn_i[4][4];
      for (int mu = 0; mu < 4; mu++)
        for (int nu = 0; nu < 4; nu++)
          gdn_i[mu][nu] = 0.0;
      if (gstate.sexact)
      {
        for (int mu = 0; mu < 4; mu++)
          for (int nu = 0; nu < 4; nu++)
            gdn_i[mu][nu] = tg4dn(0, particle_tx[mi], particle_ty[mi], particle_tz[mi], mu, nu);
      }
      else
      {
        metric_at_point(mi, ys, ys[0] - gstate.sstep, gdn_i);
      }

      /* norm = g_mu_nu u^mu u^nu of the parfile 4-velocity */
      double u[4] = { ys[4], ys[5], ys[6], ys[7] };
      double norm = 0.0;
      for (int mu = 0; mu < 4; mu++)
        for (int nu = 0; nu < 4; nu++)
          norm += gdn_i[mu][nu] * u[mu] * u[nu];

      if (isfinite(norm) && fabs(norm + 1.0) < 1.0e-6)
      {
        /* Already a normalized 4-velocity: keep the parfile values as-is. */
        CCTK_VInfo(CCTK_THORNSTRING,
                   "Particle %d init: u^mu=[%.15g,%.15g,%.15g,%.15g] "
                   "norm=%.9g (normalized, no correction) "
                   "at(%.4g,%.4g,%.4g)",
                   mi, u[0], u[1], u[2], u[3], norm, ys[1], ys[2], ys[3]);
      }
      else
      {
        /* Not (within tolerance) normalized: re-solve u^0 with u^i
           fixed (future-directed branch) and log the correction. */
        double u_spatial[3] = { u[1], u[2], u[3] };
        double u0_solved;
        if (solve_u0_future_directed(gdn_i, u_spatial, u[0], &u0_solved))
        {
          CCTK_VInfo(CCTK_THORNSTRING,
                     "Particle %d init: u^t corrected %.15g -> %.15g "
                     "(u^i unchanged; norm before correction = %.9g) "
                     "at(%.4g,%.4g,%.4g)",
                     mi, u[0], u0_solved, norm, ys[1], ys[2], ys[3]);
          ys[4] = u0_solved;
          particle_ttv[mi] = u0_solved;
        }
        else
        {
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Particle %d: no future-directed u^0 solution at init "
                     "(norm=%.4g, u^i=[%g,%g,%g]); keeping parfile values",
                     mi, norm, u[1], u[2], u[3]);
        }
      }
    }
    gstate.init_done = 1;
  }

#ifdef CCTK_MPI
  /* ---------------- multi_amr S1: state sync after the init block ----------------
     The init block above (it == 0) re-solved u^0 on the owning ranks
     and modified particle_ttv in place; all ranks need the identical
     state before the integration (and before the rank-0 dump seed
     below).  At it >= 1 it is a cheap consistency point: the state was
     already identical after the previous step's S3. */
  if (gstate.multi_amr)
    geodesic_mmpi_sync(cctkGH, CCTK_nProcs(cctkGH), CCTK_MyProc(cctkGH), particle_n_total);
#endif

  /* ==================================================================
     Diagnostic dump file (see the dump block after the particle
     loop): when particle_dump_every > 0, open the file named by
     particle_dump_file at the first iteration and record the initial
     state (t=0) BEFORE any integration step, so the file's first data
     row is the parfile input (after the init check above).  Parent
     directories are created as needed.
     ================================================================== */
  if (particle_dump_every > 0 && cctk_iteration == 0 &&
      (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
  {
    geodesic_mkdir_parents(particle_dump_file);
    gstate.dfp = fopen(particle_dump_file, "w");
    if (gstate.dfp)
    {
      fprintf(gstate.dfp, "# t x y z ut ux uy uz tau\n");
      for (mi = 0; mi < particle_n_total; mi++)
        fprintf(gstate.dfp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                (double)particle_tt[mi], (double)particle_tx[mi],
                (double)particle_ty[mi], (double)particle_tz[mi],
                (double)particle_ttv[mi], (double)particle_txv[mi],
                (double)particle_tyv[mi], (double)particle_tzv[mi],
                (double)particle_ttw[mi]);
      fflush(gstate.dfp);
    }
    else
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: cannot open %s for writing", particle_dump_file);
  }

  /* iteration 0: the grid is still in its initial state (slot 0 holds
     the t=0 metric, the past slots hold negative-time data), so there is
     no [t-dt, t] window to integrate over yet.  The init block above
     (u^t normalization against the t=0 metric), the S1 sync, and the
     t=0 dump seed have already run; the first integration happens at
     iteration 1, where slot 1 / slot 0 span [0, sstep] exactly.  All
     ranks reach this point identically, so the collective calls below
     are untouched. */
  if (cctk_iteration == 0)
  {
#ifdef CCTK_MPI
    if (gstate.multi_amr) { free(bd); bd = NULL; }
#endif
    return;
  }

  /* ---------------- ODE system ---------------- */

  /* ==================================================================
     Particle integration.

     multi_amr: chunked integration with collective re-owning at the
     chunk boundaries.  The Phase 1 owner map is computed once per
     Cactus step, but a pericentre passage moves a particle ~44
     level-0 cells per step -- far beyond the 4-cell halo -- so a
     single-pass integration would run the 6th-order stencil out of
     the owner rank's local arrays.  The step is therefore split into
     chunks.  Within a chunk every live particle stays strictly
     inside its owner's interior box (the chunk is at most a quarter
     of the distance-to-edge over the chunk-start speed, so the
     particle cannot reach the edge even if its speed grows a few
     times), which keeps the 3x3x3 stencil plus the 6th-order FD
     padding in bounds on the owner's arrays.  At each chunk
     boundary the full state is exchanged (S2-style sync) and the
     owner map is recomputed from the synced positions (the interior
     boxes are fixed during the step, so the Phase 1 gathered boxes
     are reused); a particle that crossed into a neighbor's box is
     picked up by the neighbor for the next chunk.  All ranks
     compute the identical chunk end from the identical state, so
     no extra communication is needed to agree on the chunking.  The
     step-boundary invariants are preserved: after the last chunk
     every live particle has t = bg + sstep, and the S2 sync below
     sees the final state.  The func_ode half-cell mid-step check
     stays as a backstop: a particle that nevertheless leaves the
     box is flagged outbd and handed to the loss policy.

     Non-multi_amr: one pass, each particle integrated once to the
     step end (the original behavior).
     ================================================================== */
  if (gstate.multi_amr)
  {
#ifdef CCTK_MPI
    {
      const int npr = CCTK_nProcs(cctkGH);
      const int me  = CCTK_MyProc(cctkGH);
      const int nlev = geodesic_amr_levels;
      const double margin = 0.25;    /* chunk = margin x distance-to-edge / speed */

      /* all live particles share the same coordinate time at the
         step start (step-boundary invariant); t0 is that value */
      double t0 = 1.0e300;
      for (mi = 0; mi < particle_n_total; mi++)
        if (!gstate.pgeodesic[mi].dead)
          t0 = fmin(t0, particle_tt[mi]);
      if (t0 >= 0.5e300)
        t0 = 0.0;                    /* no live particles: nothing to do */

      const double t_target = t0 + gstate.sstep;
      const double eps_step = 1e-13 * fmax(1.0, fabs(t_target));
      double t_now = t0;
      int nchunks = 0;

      while (t_now < t_target - eps_step)
      {
        /* ---- chunk end (computed identically on all ranks) ---- */
        double t_chunk = t_target;
        int chunk_live = 0;
        int best_mi = -1;    /* diagnostic: the particle that constrains t_chunk */
        int nnear = 0;       /* diagnostic: live particles with t_hit < 1e-3 */
        for (mi = 0; mi < particle_n_total; mi++)
        {
          if (gstate.pgeodesic[mi].dead || gstate.pgeodesic[mi].outbd)
            continue;
          chunk_live = 1;
          const int r = gstate.owner_of[mi];
          const int l = gstate.owner_lvl[mi];
          if (r < 0 || l < 0)
          {
            gstate.pgeodesic[mi].outbd = 1;   /* unowned: the loss policy handles it */
            continue;
          }
          /* the chunk end must be bit-identical on every rank (the
             chunk-boundary sync below is a collective): measure the
             distance to the edge against the OWNER's gathered box
             (Phase 1a), not this rank's local view of the level -- a
             non-owner rank's box for the same level differs, which
             desynced the chunk count and broke the collective pairing */
          const double *bo = bd + (size_t)r * nlev * 8 + (size_t)l * 8;
          const double ut = particle_ttv[mi];
          if (!(ut > 0.0) || !isfinite(ut))
          {
            gstate.pgeodesic[mi].outbd = 1;
            continue;
          }
          /* coordinate velocities dx^i/dt = u^i/u^t */
          const double vx = particle_txv[mi] / ut;
          const double vy = particle_tyv[mi] / ut;
          const double vz = particle_tzv[mi] / ut;
          if (!((vx*vx + vy*vy + vz*vz) > 0.0) ||
              !(isfinite(vx) && isfinite(vy) && isfinite(vz)))
          {
            gstate.pgeodesic[mi].outbd = 1;
            continue;
          }
          const double px = particle_tx[mi], py = particle_ty[mi], pz = particle_tz[mi];
          /* direction-aware time-to-face: per axis, the time to reach the
             face the particle moves TOWARD.  The min-distance form
             (distance to the NEAREST face over vmax) broke for particles
             riding a shared box face: the equatorial shell sits on the
             z=0 rank boundary, so dm_z ~ 1e-6 with v_z ~ 1e-7 produced
             infinitesimal chunks forever (dm shrinks 25%/chunk, never
             crossing).  With the directional form: a particle on a face
             moving away from it is unconstrained by that face; one
             moving toward the face it sits on has t_hit = 0 and is
             skipped by the threshold below (it rides through the chunk;
             the tpoint clip in cf2_fitting keeps its reads safe and the
             boundary re-owning moves it to the adjacent box) */
          double t_hit = 1.0e300;
          /* face-riding guard: a particle within FACE_EPS of the face it
             moves TOWARD is treated as sitting ON that face and does not
             constrain the chunk.  A rider's normal velocity is numerical
             noise (~1e-10) that keeps flipping sign, so d/v stays
             ~1e-5..1e-6 step after step and the 25%-margin chunks shrink
             the distance by 25% each without ever letting it cross --
             the chunk count ran into the defensive cap and every
             particle was handed to the loss policy (4p aei, t~1.64).
             Letting the rider ride is safe: its displacement across the
             face per step is ~1e-11 (far inside the 0.5-cell mid-step
             backstop and the >=4-cell ghost zone), and the
             chunk-boundary re-owning puts it on whichever side it is.
             A genuinely approaching particle (d >= FACE_EPS) still
             constrains as before and is handed to the neighbor by the
             boundary re-owning once it gets within FACE_EPS. */
          const double FACE_EPS = 1.0e-6;
          if (vx > 0.0)    { const double d = bo[3] - px; if (d >= FACE_EPS) { const double th = d / vx;  if (th < t_hit) t_hit = th; } }
          else if (vx < 0.0) { const double d = px - bo[0]; if (d >= FACE_EPS) { const double th = d / -vx; if (th < t_hit) t_hit = th; } }
          if (vy > 0.0)    { const double d = bo[4] - py; if (d >= FACE_EPS) { const double th = d / vy;  if (th < t_hit) t_hit = th; } }
          else if (vy < 0.0) { const double d = py - bo[1]; if (d >= FACE_EPS) { const double th = d / -vy; if (th < t_hit) t_hit = th; } }
          if (vz > 0.0)    { const double d = bo[5] - pz; if (d >= FACE_EPS) { const double th = d / vz;  if (th < t_hit) t_hit = th; } }
          else if (vz < 0.0) { const double d = pz - bo[2]; if (d >= FACE_EPS) { const double th = d / -vz; if (th < t_hit) t_hit = th; } }
          if (t_hit < 1.0e299)
          {
            if (t_hit < 1.0e-3)
              nnear++;
            const double tc = t_now + margin * t_hit;
            if (tc - t_now >= 1e-11 * fmax(1.0, fabs(t_target - t_now)) && tc < t_chunk)
            {
              t_chunk = tc;
              best_mi = mi;
            }
          }
        }

        if (!chunk_live)
          break;    /* all remaining particles are dead/outbd */

        if (++nchunks > 100000)
        {
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Geodesic (multi_amr): chunk count exceeded the defensive cap (100000) in one step; all live particles are handed to the loss policy");
          for (mi = 0; mi < particle_n_total; mi++)
            if (!gstate.pgeodesic[mi].dead && !gstate.pgeodesic[mi].outbd)
              gstate.pgeodesic[mi].outbd = 1;
          break;
        }

        /* diagnostic: detect a spin in the chunk loop (nchunks far above
           the normal few per step) and log the constraining particle on
           rank 0 at a handful of iteration marks */
        if (me == 0 && best_mi >= 0 &&
            (nchunks == 1000 || nchunks == 5000 || nchunks == 10000 ||
             (nchunks >= 20000 && (nchunks % 20000) == 0)))
        {
          const int bm = best_mi;
          const int br = gstate.owner_of[bm], bl = gstate.owner_lvl[bm];
          const double *bbo = bd + (size_t)br * nlev * 8 + (size_t)bl * 8;
          const double but = particle_ttv[bm];
          const double bvx = particle_txv[bm]/but,
                       bvy = particle_tyv[bm]/but,
                       bvz = particle_tzv[bm]/but;
          double bx = 1.0e300, by = 1.0e300, bz = 1.0e300;
          if (bvx > 0.0) bx = (bbo[3] - particle_tx[bm]) / bvx;
          else if (bvx < 0.0) bx = (particle_tx[bm] - bbo[0]) / -bvx;
          if (bvy > 0.0) by = (bbo[4] - particle_ty[bm]) / bvy;
          else if (bvy < 0.0) by = (particle_ty[bm] - bbo[1]) / -bvy;
          if (bvz > 0.0) bz = (bbo[5] - particle_tz[bm]) / bvz;
          else if (bvz < 0.0) bz = (particle_tz[bm] - bbo[2]) / -bvz;
          CCTK_VInfo(CCTK_THORNSTRING,
                     "chunkspin iter=%d t_now=%.17g t_chunk=%.17g nnear=%d argmin_mi=%d pos=(%.17g,%.17g,%.17g) v=(%.17g,%.17g,%.17g) owner=(%d,%d) box=([%.17g,%.17g)x[%.17g,%.17g)x[%.17g,%.17g)) th=(%.17g,%.17g,%.17g)",
                     nchunks, t_now, t_chunk, nnear, bm,
                     particle_tx[bm], particle_ty[bm], particle_tz[bm],
                     particle_txv[bm], particle_tyv[bm], particle_tzv[bm],
                     br, bl, bbo[0], bbo[3], bbo[1], bbo[4], bbo[2], bbo[5],
                     bx, by, bz);
        }

        /* ---- integrate the chunk (each rank: its owned particles) ----
           carpet is thread-safe here: the pointwise interpolation reads
           only the thread-local gview copy (carpet_interp_point), and the
           implicit OpenMP barrier below keeps every owned particle done
           before the chunk-boundary collective sync */
#ifndef CCTK_DEBUG
#pragma omp parallel for schedule(dynamic,1) shared(gstate)
#endif
        for (mi = 0; mi < particle_n_total; mi++)
        {
          if (gstate.pgeodesic[mi].dead || gstate.pgeodesic[mi].outbd)
            continue;
          if (gstate.owner_of[mi] != me)
            continue;
          geodesic_integrate_particle(cctkGH, mi, t_chunk, t0);
        }

        /* ---- chunk boundary: sync + re-own ---- */
        geodesic_mmpi_sync(cctkGH, npr, me, particle_n_total);
        if (t_chunk >= t_target - eps_step)
          break;
        geodesic_recompute_owner_map(cctkGH, bd, npr, nlev);
        t_now = t_chunk;
      }
    }
    free(bd);
    bd = NULL;
#endif
  }
  else
  {
    /* single pass: each particle integrated once to the step end
       (carpet included: carpet_interp_point reads only the thread-local
       gview copy, so it is thread-safe; the implicit OpenMP barrier
       precedes the post-loop S2 sync) */
#ifndef CCTK_DEBUG
#pragma omp parallel for schedule(dynamic,1) shared(gstate)
#endif
    for (mi = 0; mi < particle_n_total; mi++)
    {
      if (gstate.pgeodesic[mi].dead)
        continue;
      geodesic_integrate_particle(cctkGH, mi, particle_tt[mi] + gstate.sstep, particle_tt[mi]);
    }
  }

#ifdef CCTK_MPI
  /* ---------------- multi_amr S2: state sync after the ODE loop ----------------
     Each rank integrated its owned particles in place; exchange the full
     state so that the warning aggregation, the dump below, and the loss
     policy (synced again at S3) see identical data on all ranks. */
  if (gstate.multi_amr)
    geodesic_mmpi_sync(cctkGH, CCTK_nProcs(cctkGH), CCTK_MyProc(cctkGH), particle_n_total);
#endif

  /* ---------------- particle-state dump (post-integration) ----------------
     Fires on the finest-level call for which the trigger fired this
     iteration (last_integ_iter == cctk_iteration), i.e. the single
     call per iteration that integrates.  At iteration 0 the one-shot t=0
     seed (above) already wrote a block and set last_dump_iter, so the
     post-integration check (last_dump_iter != cctk_iteration) skips the
     double-write; the first post-integration block is the first real
     integration that is a multiple of particle_dump_every.  After the S2
     sync all ranks see the identical final state, so the rank-0-only dump
     writes the coherent post-integration block. */
  /* temporary debug: dump gate visibility (GEODESIC_DBG_DUMP) */
  if (getenv("GEODESIC_DBG_DUMP"))
    fprintf(stderr, "[DUMP_GATE] it=%d lvl=%d nlev_seen=%d last_integ=%d "
                    "last_dump=%d dfp=%d MeProc=%d -> %d\n",
            (int)cctk_iteration, (int)lvl, (int)gstate.nlev_seen,
            (int)gstate.last_integ_iter, (int)gstate.last_dump_iter,
            (gstate.dfp != NULL ? 1 : 0), CCTK_MyProc(cctkGH),
            (int)(lvl == gstate.nlev_seen - 1 &&
                  gstate.last_integ_iter == cctk_iteration &&
                  particle_dump_every > 0 &&
                  cctk_iteration % particle_dump_every == 0 &&
                  gstate.last_dump_iter != cctk_iteration &&
                  (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0)));
  if (lvl == gstate.nlev_seen - 1 &&
      gstate.last_integ_iter == cctk_iteration &&
      particle_dump_every > 0 &&
      (cctk_iteration % particle_dump_every == 0) &&
      gstate.last_dump_iter != cctk_iteration &&
      (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
  {
    if (gstate.dfp == NULL)
    {
      geodesic_mkdir_parents(particle_dump_file);
      gstate.dfp = fopen(particle_dump_file, "w");
      if (gstate.dfp)
        fprintf(gstate.dfp, "# t x y z ut ux uy uz tau\n");
    }
    if (gstate.dfp)
    {
      for (mi = 0; mi < particle_n_total; mi++)
        fprintf(gstate.dfp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                (double)particle_tt[mi], (double)particle_tx[mi],
                (double)particle_ty[mi], (double)particle_tz[mi],
                (double)particle_ttv[mi], (double)particle_txv[mi],
                (double)particle_tyv[mi], (double)particle_tzv[mi],
                (double)particle_ttw[mi]);
      fflush(gstate.dfp);
    }
    gstate.last_dump_iter = cctk_iteration;
    if (getenv("GEODESIC_DBG_DUMP"))
      fprintf(stderr, "[DUMP_DONE] it=%d rows=%d\n", cctk_iteration, particle_n_total);
  }

  /* ==================================================================
     Post-loop warning aggregation.  CCTK_VWarn / CCTK_VInfo must not
     be called from many OpenMP threads at once (interleaved output,
     undefined behavior), so the per-particle flags set inside the
     loop above are collected here (single-threaded, after the
     implicit parallel barrier) and reported as a few summary lines.
     ================================================================== */
  {
    int n_renorm = 0, n_normbad = 0, n_freeze = 0;
    int n_outbd = 0;
    int first_renorm[5] = { -1, -1, -1, -1, -1 };
    int n_first = 0;
    double rl_min = 0.0, rl_max = 0.0;
    int rl_have = 0;
    const char *loss_txt =
      (loss_policy == 0) ? "dropped (state set to NaN)" :
      (loss_policy == 1) ? "flagged (frozen at their last state)" :
                           "respawned";
    for (mi = 0; mi < particle_n_total; mi++)
    {
      if (gstate.pgeodesic[mi].renorm_warn)
      {
        n_renorm++;
        if (n_first < 5) first_renorm[n_first] = mi;
        n_first++;
      }
      if (gstate.pgeodesic[mi].norm_bad)    n_normbad++;
      if (gstate.pgeodesic[mi].freeze_warn) n_freeze++;
      /* "pure" out-of-bounds losses (safe region, excised sphere, ODE
         failure) have no other diagnostic; report them so that a
         dropped/flagged particle is never silent */
      if (gstate.pgeodesic[mi].outbd && !gstate.pgeodesic[mi].norm_bad &&
          !gstate.pgeodesic[mi].renorm_warn && !gstate.pgeodesic[mi].freeze_warn)
        n_outbd++;
      if (gstate.pgeodesic[mi].rl_last != 0.0)
      {
        if (!rl_have)
        {
          rl_min = rl_max = gstate.pgeodesic[mi].rl_last;
          rl_have = 1;
        }
        else
        {
          if (gstate.pgeodesic[mi].rl_last < rl_min) rl_min = gstate.pgeodesic[mi].rl_last;
          if (gstate.pgeodesic[mi].rl_last > rl_max) rl_max = gstate.pgeodesic[mi].rl_last;
        }
      }
    }
    /* multi_amr: the flags are identical on all ranks after the S2 sync,
       so the summary lines are printed on rank 0 only */
    if (n_renorm && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) could not be renormalized (u.u >= 0 or non-finite) and will be %s; first: %d %d %d %d %d",
                 n_renorm, loss_txt, first_renorm[0], first_renorm[1], first_renorm[2], first_renorm[3], first_renorm[4]);
    if (n_normbad && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) had no future-directed u^0 solution (possibly superluminal spatial velocity) and will be %s",
                 n_normbad, loss_txt);
    if (n_freeze && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) hit the tau-loop iteration cap and will be %s",
                 n_freeze, loss_txt);
    if (n_outbd && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) left the safe region or failed the ODE step and will be %s",
                 n_outbd, loss_txt);
    /* rl (radial-light-signal / ray) range is reported once per coarse step,
       at the same level-0 POSTSTEP moments as the dump/integration.  This is
       independent of the AMR regrid schedule; rl_have is 1 only after a
       successful integration, so no line is printed without real data. */
    if (gstate.srverbose && rl_have && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VInfo(CCTK_THORNSTRING, "rl: min:%18.15G max:%18.15G", rl_min, rl_max);
  }

  /* ==================================================================
     Loss policy: particles that left the domain or failed
     normalization are handled according to particle_loss_policy:
       drop    -- state is set to NaN and the particle is no longer
                  tracked (t / tau are kept as "when it was lost")
       flag    -- state is frozen at its last valid values and the
                  particle is no longer tracked
       respawn -- (legacy Monte-Carlo tracer behavior) the particle is
                  re-placed at a random position inside the safe box
                  (rejecting the polar axis, the excision sphere, and
                  the region outside the initial shell) with a u^t
                  placeholder (u^i = 0); the per-iteration re-solve
                  normalizes u^mu to u_mu u^mu = -1 on the next step
     In multi_amr mode only the owner rank handles each particle
     (a respawn uses that rank's local safe box); the resulting state
     is exchanged by the S3 sync below.
     ================================================================== */
  for (mi=0; mi<particle_n_total; mi++)
  {
    if (gstate.multi_amr && gstate.owner_of[mi] != CCTK_MyProc(cctkGH))
      continue;
    if (gstate.pgeodesic[mi].norm_bad)
      CCTK_VInfo(CCTK_THORNSTRING,
             "Particle %d flagged norm_bad this step (see warnings above).", mi);

    if (gstate.pgeodesic[mi].outbd || gstate.pgeodesic[mi].norm_bad)
    {
      if (loss_policy == 2)
      {
        particle_ttw[mi] = 0.0;
        /* particle_tt[mi]  = 0.0; */

        double er;
        long int i1 = 0;
        do
        {
          particle_tx[mi]  = urand_range(&gstate.seed, gstate.xmin_safe, gstate.xmax_safe);
          particle_ty[mi]  = urand_range(&gstate.seed, gstate.ymin_safe, gstate.ymax_safe);
          particle_tz[mi]  = urand_range(&gstate.seed, gstate.zmin_safe, gstate.zmax_safe);
          er = particle_tx[mi]*particle_tx[mi] + particle_ty[mi]*particle_ty[mi] + particle_tz[mi]*particle_tz[mi];
          i1++;
        }while ((fabs(particle_tz[mi]) < particle_middle_sp || er < (excised_radius + 1.5)*(excised_radius + 1.5) || er > gstate.sinitial_radius*gstate.sinitial_radius) && i1 < 100000);

        if (i1 >= 100000)
        {
          CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: respawn rejection loop exhausted "
                    "(check excised_radius/initial_radius/rand box geometry).");
        }

        /* u^t placeholder (u^i = 0); the per-iteration re-solve normalizes
           u^mu to u_mu u^mu = -1 on the next step */
        particle_ttv[mi] = 1.0;
        particle_txv[mi] = 0.0;
        particle_tyv[mi] = 0.0;
        particle_tzv[mi] = 0.0;
        gstate.pgeodesic[mi].dead = 0;
      }
      else if (loss_policy == 0)
      {
        /* drop: wipe the state and stop tracking the particle */
        particle_tx[mi]  = NAN;
        particle_ty[mi]  = NAN;
        particle_tz[mi]  = NAN;
        particle_ttv[mi] = NAN;
        particle_txv[mi] = NAN;
        particle_tyv[mi] = NAN;
        particle_tzv[mi] = NAN;
        gstate.pgeodesic[mi].dead = 1;
      }
      else
      {
        /* flag: keep the last valid state, stop tracking */
        gstate.pgeodesic[mi].dead = 1;
      }
      gstate.pgeodesic[mi].outbd = 0;
      gstate.pgeodesic[mi].norm_bad = 0;
    }
  }

#ifdef CCTK_MPI
  /* ---------------- multi_amr S3: state sync after the loss policy ----------------
     The owner ranks may have modified their particles' states (respawn /
     drop); exchanging them again leaves every rank with the identical
     full state for the next Cactus step. */
  if (gstate.multi_amr)
    geodesic_mmpi_sync(cctkGH, CCTK_nProcs(cctkGH), CCTK_MyProc(cctkGH), particle_n_total);
#endif

  /* the per-particle cache (pgeodesic) is kept and reused at the next
     Cactus step; it is reclaimed by the OS at the end of the run */
}


