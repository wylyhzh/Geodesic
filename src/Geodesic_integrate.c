
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
                         .samr_levels = 1, .nlev_seen = 1,
                         .aei_handle = -1, .aei_coordsys = -1 };

/* ============================ AMR support (max GEODESIC_MAX_LEVELS levels) ====

   Single process, multi-thread.  geodesics_integrate is called once per
   refinement level at CCTK_POSTSTEP (Carpet level-mode traversal); each
   call only stores the level's data POINTERS + geometry into LD[level]
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
   switch_to_level() from gstate.LD[level].  All grid access in this
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
  CCTK_INT level, gen;
} GeodesicView;
static GEODESIC_TLS GeodesicView gview = { .level = -1, .gen = -1 };

/* linear index into the current thread's level arrays (i,j,k are 0-based) */
#define LD_INDEX(i,j,k) ((i) + gview.lsh[0]*((j) + gview.lsh[1]*(k)))

/* point this thread's grid view at level l (self-guarded, cheap) */
static void switch_to_level(int l)
{
  if (gview.level == l && gview.gen == gstate.ld_generation) return;
  gview.level = l;
  gview.gen   = gstate.ld_generation;

  LevelData *L = &gstate.LD[l];
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

/* finest level whose SAFE region (7-cell margin) contains y; -1 if none */
static int pick_level(const double y[])
{
  for (int l = gstate.samr_levels-1; l >= 0; l--)
  {
    if (!gstate.LD[l].valid) continue;
    const LevelData *L = &gstate.LD[l];
    if (y[1] >= L->r_lbnd[0] && y[1] <= L->r_ubnd[0] &&
        y[2] >= L->r_lbnd[1] && y[2] <= L->r_ubnd[1] &&
        y[3] >= L->r_lbnd[2] && y[3] <= L->r_ubnd[2])
      return l;
  }
  return -1;
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
                     : (gstate.multi_exact ? 0 : pick_level(y));
  if (lvl < 0)
  {
    gstate.pgeodesic[m].outbd = 1;
    return 0;
  }
  /* make sure this thread's grid view matches the particle's level
     (self-guarded; also covers the no-rebuild path when this thread
     previously worked on a different particle/level) */
  switch_to_level(lvl);

  /* compute base grid point index for the current particle position (on level lvl) */
  CCTK_INT tpoint[4];
  tpoint[0] = 0;
  for (i=1; i<4; i++)
  {
    tpoint[i] = lrint (floor ((y[i] - gstate.LD[lvl].origin[i-1]) * gstate.LD[lvl].inv_dx[i-1] + 0.5));
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
      if (tpoint[i] > gstate.LD[lvl].lsh[i-1] - 4)
        tpoint[i] = gstate.LD[lvl].lsh[i-1] - 4;
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
      gstate.pgeodesic[m].below[i] = gstate.LD[lvl].origin[i-1] + (tpoint[i] - 1) * gstate.LD[lvl].dx[i-1];
      gstate.pgeodesic[m].dxp[i-1] = gstate.LD[lvl].dx[i-1];
    }

    /* !sexact: sample metric + Christoffel symbols on the stencil.
       In the sexact branch both are exact Kerr-Schild analytic values,
       evaluated directly by tg4dn/gdn_derivatives inside func_ode, so
       no sampling is needed. */
    if (!gstate.sexact && gstate.sinterp_mode == 0)
    {
      if (!gstate.stime_interp)
      {
        /* original path: metric + Christoffel symbols on the stencil */
        for (j1=0; j1<3; j1++)
          for (j2=0; j2<3; j2++)
            for (j3=0; j3<3; j3++)
              cf2_grid(m, j1, j2, j3);
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
static int carpet_interp_point(double x, double y, double z,
                               double g31[10], double dg31[10][3])
{
  const double *fp[10] = {
    gview.gxx[0], gview.gxy[0], gview.gxz[0], gview.gyy[0],
    gview.gyz[0], gview.gzz[0], gview.alp[0],
    gview.betax[0], gview.betay[0], gview.betaz[0]
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

/* 4-metric at a particle position (renormalization; void).
   sexact: exact Kerr-Schild.  Grid modes: level selection via
   cf2_fitting (switch_to_level); interpolation failure flags the
   particle out-of-bounds and leaves gdn zeroed by the caller. */
static void metric_at_point(int m, const double y[8], double bg, double gdn[4][4])
{
  int mu, nu;
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
    if (!gstate.stime_interp)
    {
      for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++)
        gdn[mu][nu]=poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],gstate.pgeodesic[m].pgdc[mu][nu]);
    }
    else
    {
      /* the grid's two ADMBase time levels span [t-dt, t] with t the newest
         grid time; the particle window [bg, bg+sstep] starts at bg = t-dt
         (the old level's time) when sstep = dt, so theta in [0,1] maps the
         window exactly onto the two levels */
      double theta=(y[0]-bg)/gstate.sstep;
      if (theta<0.0) theta=0.0;
      if (theta>1.0) theta=1.0;
      for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++)
      {
        const double g0=poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],gstate.pgeodesic[m].pgdc[mu][nu]);
        const double g1=poly27_eval(y,m,gstate.pgeodesic[m].pgd1[mu][nu],gstate.pgeodesic[m].pgd1c[mu][nu]);
        gdn[mu][nu]=g1+theta*(g0-g1);
      }
    }
    for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
    return;
  }
  if (gstate.sinterp_mode == 1)
  {
    double g31[10], dg31[10][3];
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) { gstate.pgeodesic[m].outbd=1; return; }
    if (!gstate.stime_interp)
    {
      compose_4metric(g31,dg31,gdn,NULL,NULL);
      return;
    }
    {
      double g31a[10], dg31a[10][3];
      /* the grid's two ADMBase time levels span [t-dt, t] with t the newest
         grid time; the particle window [bg, bg+sstep] starts at bg = t-dt
         (the old level's time) when sstep = dt, so theta in [0,1] maps the
         window exactly onto the two levels */
      double theta=(y[0]-bg)/gstate.sstep;
      if (theta<0.0) theta=0.0;
      if (theta>1.0) theta=1.0;
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
    if (carpet_interp_point(y[1],y[2],y[3],g31,dg31) != 0) { gstate.pgeodesic[m].outbd=1; return; }
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
    /* local27+linear: evaluate the two fitted pointwise-Christoffel
       polynomials (t and t - dt) at the particle and blend linearly
       in time; for a stationary metric the two fits are identical and
       the blend reduces exactly to the off-path value */
    /* the grid's two ADMBase time levels span [t-dt, t] with t the newest
       grid time; the particle window [bg, bg+sstep] starts at bg = t-dt
       (the old level's time) when sstep = dt, so theta in [0,1] maps the
       window exactly onto the two levels */
    double theta=(y[0]-bg)/gstate.sstep;
    if (theta<0.0) theta=0.0;
    if (theta>1.0) theta=1.0;
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
      /* the grid's two ADMBase time levels span [t-dt, t] with t the newest
         grid time; the particle window [bg, bg+sstep] starts at bg = t-dt
         (the old level's time) when sstep = dt, so theta in [0,1] maps the
         window exactly onto the two levels */
      double theta=(y[0]-bg)/gstate.sstep;
      if (theta<0.0) theta=0.0;
      if (theta>1.0) theta=1.0;
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
  { /* carpet (time_interp = linear is rejected in ParamCheck) */
    double g31[10], dg31[10][3];
    if (gstate.stime_interp) { gstate.pgeodesic[m].outbd=1; return 0; }
    if (carpet_interp_point(y[1],y[2],y[3],g31,dg31) != 0) return 0;
    compose_4metric(g31,dg31,gdn,dgd,gup);
    if (gview.metric_static)
    {
      for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) dgd[0][j1][j2]=0.0;
    }
    else
    {
      CCTK_INT tp[3];
      gd_data gd0,gd1,gd2;
      int i;
      for (i=0;i<3;i++)
      {
        tp[i]=lrint(floor((y[1+i]-gview.origin_space[i])*gview.delta_inv[i]+0.5));
        if (tp[i]<1) tp[i]=1;
        if (tp[i]>gview.lsh[i]-2) tp[i]=gview.lsh[i]-2;
      }
      metric_at_cell_tl(0,tp[0],tp[1],tp[2],&gd0);
      metric_at_cell_tl(1,tp[0],tp[1],tp[2],&gd1);
      metric_at_cell_tl(2,tp[0],tp[1],tp[2],&gd2);
      for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++)
        dgd[0][j1][j2]=(3.0*gd0.gd[j1][j2]-4.0*gd1.gd[j1][j2]+gd2.gd[j1][j2])*(0.5/gview.dt);
    }
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
    if ((!gstate.multi_exact && !gstate.multi_amr &&
        ((y[1] < gstate.LD[0].r_lbnd[0] || y[1] > gstate.LD[0].r_ubnd[0]) ||
        (y[2] < gstate.LD[0].r_lbnd[1] || y[2] > gstate.LD[0].r_ubnd[1]) ||
        (y[3] < gstate.LD[0].r_lbnd[2] || y[3] > gstate.LD[0].r_ubnd[2]))))
    {
      gstate.pgeodesic[m].outbd = 1;
    }else
    {
      const int lvl = gstate.multi_amr ? gstate.owner_lvl[m]
                         : (gstate.multi_exact ? 0 : pick_level(y));
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
          const LevelData *Lo = &gstate.LD[lvl];
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
  switch_to_level(gstate.pgeodesic[m].level);

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
    if (gstate.sinterp_mode == 0 && !gstate.stime_interp)
    {
      /* original path: 27-point Lagrange polynomial of the fitted Christoffel */
      for (j1=0; j1<4; j1++)
        for (j2=0; j2<4; j2++)
          for (j3=j2; j3<4; j3++)
            cf2[j1][j2][j3] = cf2_func(y, m, j1, j2, j3);
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
   freeze_warn rl_last outbd.  Full exchange (200 x 15 = 23.2 KB/rank) keeps
   the code simple; every rank holds the complete state.  The particle
   arrays are DISTRIB=CONSTANT grid variables, addressed through the
   patch-local pointers that DECLARE_CCTK_ARGUMENTS injects (cctkGH). */
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
         (particle_tz[mi] != 0.0 && (!isnormal(particle_tz[mi]))) ||
         (!gstate.multi_exact && !gstate.multi_amr &&
          (particle_tx[mi] < gstate.LD[0].r_lbnd[0] || particle_tx[mi] > gstate.LD[0].r_ubnd[0] ||
          particle_ty[mi] < gstate.LD[0].r_lbnd[1] || particle_ty[mi] > gstate.LD[0].r_ubnd[1] ||
          particle_tz[mi] < gstate.LD[0].r_lbnd[2] || particle_tz[mi] > gstate.LD[0].r_ubnd[2])) )
    {
      gstate.pgeodesic[mi].outbd = 1;
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
          particle_tx[mi] < gstate.LD[ol].i_lbnd[0] || particle_tx[mi] >= gstate.LD[ol].i_ubnd[0] ||
          particle_ty[mi] < gstate.LD[ol].i_lbnd[1] || particle_ty[mi] >= gstate.LD[ol].i_ubnd[1] ||
          particle_tz[mi] < gstate.LD[ol].i_lbnd[2] || particle_tz[mi] >= gstate.LD[ol].i_ubnd[2])
        gstate.pgeodesic[mi].outbd = 1;
    }

    /* ---------------- integrate one particle ---------------- */
    memset((void *)&odedata, 0, sizeof(odedata));
    odedata.m  = mi;
    odedata.bg = t_step0;
    odedata.st = &gstate;

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

      if (!isfinite(rl) || rl >= 0.0)
      {
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
    }

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
        /* CCTK_VInfo(CCTK_THORNSTRING,
                 "Iteration %d Particle %d: gsl_odeiv2_evolve_apply failed (code=%d) X=%g Y=%g Z=%g dt=%g dx=%g dy=%g dz=%g\n", cctk_iteration, mi, retc, ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7]); */
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
          gstate.pgeodesic[mi].outbd = 1;
          gstate.pgeodesic[mi].renorm_warn = 1;   /* aggregated after the loop */
        }
        else
        {
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
    if (level >= geodesic_amr_levels)
      return;   /* more levels than configured: ignore the extras (avoid LD[] overflow) */

    LevelData *L;

    if (level < 0 || level >= geodesic_amr_levels || level >= GEODESIC_MAX_LEVELS)
      CCTK_WARN(CCTK_WARN_ABORT,
                "Geodesic: refinement level out of range. "
                "Check geodesic_amr_levels and GEODESIC_MAX_LEVELS.");

    /* With time subcycling, Carpet calls CCTK_POSTSTEP once per level per
       substep; the trigger logic at the end of this block integrates
       only at the moment all levels hold data at the same time. */

    L = &gstate.LD[level];
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
         x(0) is reconstructed differently per case:
           level 0: the driver value cctk_origin_space is correct by
             construction.
           single-process fine level: a static regrid centered on the
             origin (refinement position (0,0,0), cell centering) gives a
             local array (box plus equal ghost zones on both sides) that
             is symmetric about x = 0, so x(0) = -dx*(lsh-1)/2.  The
             alignment check below verifies that the interior box aligns
             with the coarse grid and is centered on the origin.
           multi-process fine level: per-rank quadrant patches are not
             centered; the old formula x(0) = origin_coarse +
             (lbnd + (ngz-1)*(levfac-1)) * dx_lev is kept for
             compatibility but is untested on this grid. */
      L->dx[i]     = cctk_delta_space[i] / cctk_levfac[i];
      L->inv_dx[i] = 1.0 / L->dx[i];
      if (level == 0 || CCTK_nProcs(cctkGH) > 1)
      {
        const double align = (cctk_nghostzones[i] - 1.0) * (cctk_levfac[i] - 1);
        L->origin[i] = (level == 0) ? cctk_origin_space[i]
                                    : cctk_origin_space[i] + (cctk_lbnd[i] + align) * L->dx[i];
      }
      else
        L->origin[i] = -0.5 * (cctk_lsh[i] - 1) * L->dx[i];

      {
        /* temporary debug: dump the raw cGH geometry and the derived
           per-level geometry once per axis (gated by GEODESIC_DEBUG_GEOM) */
        static int dbg_geom = -1, n_dbg_geom = 0;
        if (dbg_geom < 0) dbg_geom = (getenv("GEODESIC_DEBUG_GEOM") != NULL);
        if (dbg_geom && n_dbg_geom < 40)
        {
          n_dbg_geom++;
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
         interior box spans nint = lsh - 2*ngz cells.  The centered-patch
         origin formula above requires nint/levfac coarse cells (the box
         aligns with coarse cells: divisibility) and that count must be
         even (the box is centered on the origin).  Any other arrangement
         (different centering, asymmetric regrid, uneven ghost zones)
         aborts instead of silently producing misaligned geometry.  Level
         0 is exempt (a plain unigrid domain need not be centered);
         multi-process ranks are exempt (per-rank quadrant patches do not
         satisfy either condition). */
      if (level > 0 && CCTK_nProcs(cctkGH) == 1)
      {
        const int nint = (int)cctk_lsh[i] - 2 * (int)cctk_nghostzones[i];
        if (nint <= 0 || nint % cctk_levfac[i] != 0)
          CCTK_ERROR("Geodesic: fine-level interior box (lsh - 2*ngz) is not "
                     "a multiple of levfac; the patch does not align with the "
                     "coarse grid.  The centered-patch origin reconstruction "
                     "does not apply to this arrangement.");
        if ((nint / cctk_levfac[i]) % 2 != 0)
          CCTK_ERROR("Geodesic: fine-level interior box is not symmetric about "
                     "the origin (nint/levfac odd); the centered-patch origin "
                     "reconstruction does not apply to this arrangement.");
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

    L->dt = cctk_delta_time;

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
       Trigger: integrate exactly once per iteration, and only at the
       moment all levels hold fresh data at the same time.

       With time subcycling, cctk_iteration is counted in finest-level
       substeps.  Observed call pattern (finest level L, 2^(L-1)
       substeps per coarse step; for L = 3, every 4 iterations):
           it%4==0:  L0, L1, L2   <- coarse step finished, all levels
                                     synchronized at cctk_time,
                                     called L0 -> L1 -> L2
           it%4==1:  L2
           it%4==2:  L1, L2
           it%4==3:  L2
       The only correct moment to integrate is the finest-level call of
       the it%4==0 group -- the moment all levels hold fresh data at
       the same time t+dt.  Criterion: L0 already seen in this
       iteration + current call is on the finest level + not yet
       integrated in this iteration.  The subcycling factor is NOT
       assumed; this works for any number of levels, with subcycling
       off, and without AMR (a single-level run sees all three
       conditions hold on its only call).

       Auto-adaptation: the finest level is the max level seen so far
       (nlev_seen); the true level count is locked in within one
       iteration.

       Iteration 0 is the exception: without a guard the trigger above
       would fire on the LEVEL-0 call, before the finer levels' acquire
       has run -- their LevelData is not valid yet, the Phase 1 owner
       map would pin every particle to level 0, and the init re-solve
       would normalize u^mu against coarse level-0 data (measured:
       u^t off by up to 0.2 vs the single-process finest-level init).
       The guard below waits for the finest level's call, where all
       levels hold fresh data at the same cctk_time.  Single-level
       runs (geodesic_amr_levels = 1) are unaffected.
       --------------------------------------------------------------- */
    if (level + 1 > gstate.nlev_seen)
      gstate.nlev_seen = level + 1;
    if (level == 0)
      gstate.l0_seen_iter = cctk_iteration;

    if (level != gstate.nlev_seen-1)
      return;                               /* integrate only on the finest-level call */
    if (gstate.l0_seen_iter != cctk_iteration)
      return;                               /* mid-subcycle: coarse-level data is stale */
    if (cctk_iteration == 0 && level < geodesic_amr_levels - 1)
      return;                               /* iteration 0: wait for the finest level  */
    if (gstate.last_integ_iter == cctk_iteration)
      return;                               /* already integrated in this iteration    */
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

    /* Phase 1a: Allgather of per-rank per-level interior boxes
       (nlev * 8 doubles/rank: i_lbnd[3], i_ubnd[3], fresh, metric_static).
       fresh = (L->valid && L->stamp == cctk_iteration) -- at the trigger
       point all levels were acquired earlier in the same traversal. */
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
        LevelData *L = &gstate.LD[l];
        double *s = s0 + (size_t)l * 8;
        s[0] = L->i_lbnd[0]; s[1] = L->i_lbnd[1]; s[2] = L->i_lbnd[2];
        s[3] = L->i_ubnd[0]; s[4] = L->i_ubnd[1]; s[5] = L->i_ubnd[2];
        s[6] = (L->valid && L->stamp == cctk_iteration) ? 1.0 : 0.0;
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
  gstate.smoke_eps = smoke_timedep_eps;
  gstate.cur_gh = cctkGH;

  if (!gstate.sexact && gstate.sinterp_mode > 0 && gstate.aei_handle < 0)
  {
    /* the interpolator/coordinates-system handles and the ADMBase
       variable indices are resolved lazily at first integration (the
       interpolators are registered at STARTUP, after PARAMCHECK) */
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

  /* default: use the grid time step.  In the AMR call pattern
     cctk_delta_time is the coarse step size on every level, so take
     the level-0 value (identical for a single-level run). */
  if (step == 0.0) gstate.sstep = gstate.LD[0].dt;
  else             gstate.sstep = step;

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
      if (!gstate.LD[lv].valid || gstate.LD[lv].metric_static)
        continue;
      if (gstate.LD[lv].dt < dt_min)
        dt_min = gstate.LD[lv].dt;
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
  if (!gstate.init_done)
  {
    if (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0)
      CCTK_VInfo(CCTK_THORNSTRING, "Kerr M:%.4g a:%.4g Particle total:%d", gstate.ksm, gstate.ksa, particle_n_total);
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
    if (gstate.srverbose && rl_have && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VInfo(CCTK_THORNSTRING, "rl: min:%18.15G max:%18.15G", rl_min, rl_max);
  }

  /* ==================================================================
     Diagnostic dump: append the full particle state
     (t, x, y, z, u^t, u^x, u^y, u^z, tau) to the file named by
     particle_dump_file, once every particle_dump_every iterations
     (0 disables).  The file was opened and seeded with the t=0
     initial state before the first integration step (see above).
     Runs after the parallel particle loop, single-threaded (parallelism
     is OpenMP inside the process); purely diagnostic, no effect on the
     integration.  In multi_amr mode only rank 0 writes: the state is
     bit-identical on all ranks after the S2 sync, so the file content
     is independent of the process count.
     ================================================================== */
  if (particle_dump_every > 0 &&
      (cctk_iteration % particle_dump_every == 0) &&
      (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
  {
    if (gstate.dfp == NULL)
    {
      /* defensive fallback (e.g. the initial open failed earlier) */
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
