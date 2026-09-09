
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
     - single-grid data acquisition (grid-function pointers + geometry)
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
#include "Geodesic.h"

#define lN      27                      /* number of data points to fit */

/* NOTE:
   These are defined here as globals because your current build compiles Geodesic_integrate.c
   as the only translation unit including this header. If later multiple .c include this header,
   consider switching these to extern + one .c definition.
*/

/* NOTE: the grid-data globals (sgxx, salp, sbetax, sGH, delta_space, ...)
   are defined further below.  They point at the (single) unigrid and are
   shared, read-only, by all OpenMP threads. */
double ksm;
double ksa;
double sinitial_radius;
double xmin_safe;
double xmax_safe;
double ymin_safe;
double ymax_safe;
double zmin_safe;
double zmax_safe;

double sstep;

/* ============================ single-grid (unigrid) data ====================

   Single process, multi-thread, one grid.  geodesics_integrate is called
   once per iteration at CCTK_POSTSTEP; it caches the grid-function data
   POINTERS + geometry into the globals below (no copy).  The cached data
   is read-only during the particle integration, so all OpenMP threads
   share it.

   All grid access below goes through these globals.  Legacy code that uses
   CCTK_GFINDEX3D(sGH,...) (e.g. g4dn() in derivative.h) works against
   cctkGH itself.
   ============================================================================ */
static cGH * sGH;
static double * sgxx;
static double * sgxy;
static double * sgxz;
static double * sgyy;
static double * sgyz;
static double * sgzz;
static double * salp;
static double * sbetax;
static double * sbetay;
static double * sbetaz;
static CCTK_REAL delta_inv[3];
static CCTK_REAL origin_space[3];
static CCTK_REAL delta_space[3];
static CCTK_REAL r_lbnd[3];
static CCTK_REAL r_ubnd[3];
static CCTK_INT  lsh[3];

/* linear index into the grid arrays (i,j,k are 0-based) */
#define LD_INDEX(i,j,k) ((i) + lsh[0]*((j) + lsh[1]*(k)))

/* Derivatives / analytic Kerr-Schild helpers, polynomial matrix MtxA etc. */
#include "derivative.h"

ParticleGeodesic * pgeodesic;
static int pgeodesic_capacity = 0;   /* slots allocated in pgeodesic */


bool sexact;
bool sgverbose;
bool srverbose;
bool check_velocity; /* particle velocity */
unsigned int seed = 123456789u;
int dispno = -10;

/* ------------------ forward declarations (existing) ------------------ */
double ndg4dn(int m, int d, int j1, int j2, int ix, int iy, int iz);

double urand_range(unsigned int *seed, double a, double b);

void cf2_grid(int m, int i1, int i2, int i3);

int func_ode(double t, const double y[], double f[], void *params);

CCTK_REAL cf2_func(const double y[], int m, int j1, int j2, int j3);

int cf2_fitting(int m, const double y[]);

void geodesics_integrate(CCTK_ARGUMENTS);

/* ============================================================================================
   6th-order FD for metric derivatives (your original)
   ============================================================================================ */
double ndg4dn(int m, int d, int j1, int j2, int ix, int iy, int iz)
{
  (void)m;
  double od60, ret;

  ret = 0.0;
  if (d == 0)
  {
    ret = 0.0;
  } else if (d == 1)
  {
    od60 = 1 / (60 * delta_space[0]);
    ret = (g4dn(m, j1, j2, ix+3, iy, iz) -  9*g4dn(m, j1, j2, ix+2, iy, iz) +
        45*g4dn(m, j1, j2, ix+1, iy, iz) -    g4dn(m, j1, j2, ix-3, iy, iz) +
         9*g4dn(m, j1, j2, ix-2, iy, iz) - 45*g4dn(m, j1, j2, ix-1, iy, iz)) * od60;
  } else if (d == 2)
  {
    od60 = 1 / (60 * delta_space[1]);
    ret = (g4dn(m, j1, j2, ix, iy+3, iz) -  9*g4dn(m, j1, j2, ix, iy+2, iz) +
        45*g4dn(m, j1, j2, ix, iy+1, iz) -    g4dn(m, j1, j2, ix, iy-3, iz) +
         9*g4dn(m, j1, j2, ix, iy-2, iz) - 45*g4dn(m, j1, j2, ix, iy-1, iz)) * od60;
  } else if (d == 3)
  {
    od60 = 1 / (60 * delta_space[2]);
    ret = (g4dn(m, j1, j2, ix, iy, iz+3) -  9*g4dn(m, j1, j2, ix, iy, iz+2) +
        45*g4dn(m, j1, j2, ix, iy, iz+1) -    g4dn(m, j1, j2, ix, iy, iz-3) +
         9*g4dn(m, j1, j2, ix, iy, iz-2) - 45*g4dn(m, j1, j2, ix, iy, iz-1)) * od60;
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

  if (pgeodesic[m].pcf2c[j1][j2][j3])
  {
    ret = pgeodesic[m].pcf2[j1][j2][j3][0];
  }
  else
  {
    tx1 = (y[1] - pgeodesic[m].below[1])/delta_space[0];
    tx2 = tx1*tx1;
    ty1 = (y[2] - pgeodesic[m].below[2])/delta_space[1];
    ty2 = ty1*ty1;
    tz1 = (y[3] - pgeodesic[m].below[3])/delta_space[2];
    tz2 = tz1*tz1;

    ret = pgeodesic[m].pcf2[j1][j2][j3][ 0] +
          pgeodesic[m].pcf2[j1][j2][j3][ 1]*tx1 +
          pgeodesic[m].pcf2[j1][j2][j3][ 2]*ty1 +
          pgeodesic[m].pcf2[j1][j2][j3][ 3]*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][ 4]*tx2 +
          pgeodesic[m].pcf2[j1][j2][j3][ 5]*ty2 +
          pgeodesic[m].pcf2[j1][j2][j3][ 6]*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][ 7]*tx1*ty1 +
          pgeodesic[m].pcf2[j1][j2][j3][ 8]*tx1*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][ 9]*ty1*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][10]*tx2*ty1 +
          pgeodesic[m].pcf2[j1][j2][j3][11]*tx2*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][12]*tx1*ty2 +
          pgeodesic[m].pcf2[j1][j2][j3][13]*ty2*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][14]*tx1*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][15]*ty1*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][16]*tx1*ty1*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][17]*tx2*ty2 +
          pgeodesic[m].pcf2[j1][j2][j3][18]*tx2*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][19]*ty2*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][20]*tx2*ty1*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][21]*tx1*ty2*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][22]*tx1*ty1*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][23]*tx2*ty2*tz1 +
          pgeodesic[m].pcf2[j1][j2][j3][24]*tx2*ty1*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][25]*tx1*ty2*tz2 +
          pgeodesic[m].pcf2[j1][j2][j3][26]*tx2*ty2*tz2;
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

  const CCTK_REAL tx1 = (y[1] - pgeodesic[m].below[1]) / delta_space[0];
  const CCTK_REAL ty1 = (y[2] - pgeodesic[m].below[2]) / delta_space[1];
  const CCTK_REAL tz1 = (y[3] - pgeodesic[m].below[3]) / delta_space[2];

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
  i1 = i1 + pgeodesic[m].point[1] - 1;
  i2 = i2 + pgeodesic[m].point[2] - 1;
  i3 = i3 + pgeodesic[m].point[3] - 1;

  index = LD_INDEX(i1-1, i2-1, i3-1);

  /* ---------------- metric from ADMBase 3+1 ---------------- */
  gxxL = sgxx[index];
  gxyL = sgxy[index];
  gxzL = sgxz[index];
  gyyL = sgyy[index];
  gyzL = sgyz[index];
  gzzL = sgzz[index];

  det = -gxzL*gxzL*gyyL + 2*gxyL*gxzL*gyzL - gxxL*gyzL*gyzL - gxyL*gxyL*gzzL + gxxL*gyyL*gzzL;
  invdet = 1.0 / det;
  gupxxL = (-gyzL*gyzL + gyyL*gzzL)*invdet;
  gupxyL = ( gxzL*gyzL - gxyL*gzzL)*invdet;
  gupyyL = (-gxzL*gxzL + gxxL*gzzL)*invdet;
  gupxzL = (-gxzL*gyyL + gxyL*gyzL)*invdet;
  gupyzL = ( gxyL*gxzL - gxxL*gyzL)*invdet;
  gupzzL = (-gxyL*gxyL + gxxL*gyyL)*invdet;

  lapse    = salp[index];
  lapse_1  = 1.0/lapse;
  lapse_11 = lapse_1*lapse_1;
  shiftx   = sbetax[index];
  shifty   = sbetay[index];
  shiftz   = sbetaz[index];

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
          pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = -lapse*lapse + (shiftx*shift_x + shifty*shift_y + shiftz*shift_z);
        }
        else
        {
          if (j2 == 1)
            pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxxL*shiftx + gxyL*shifty + gxzL*shiftz;
          else if (j2 == 2)
            pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxyL*shiftx + gyyL*shifty + gyzL*shiftz;
          else
            pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxzL*shiftx + gyzL*shifty + gzzL*shiftz;
        }
      }
      else
      {
        if (j1 == 1)
        {
          if (j2 == 1)      pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxxL;
          else if (j2 == 2) pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxyL;
          else              pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gxzL;
        }
        else if (j1 == 2)
        {
          if (j2 == 2)      pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gyyL;
          else              pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gyzL;
        }
        else
        {
          pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = gzzL;
        }
      }
    }
  }

  for(j1=1; j1<4; j1++)
    for(j2=0; j2<j1; j2++)
      pgeodesic[m].gd[i4][i5][i6].gd[j1][j2] = pgeodesic[m].gd[i4][i5][i6].gd[j2][j1];

  /* ---------------- derivatives of metric (your ndg4dn) ---------------- */
  for(j1=0; j1<4; j1++)
    for(j2=0; j2<4; j2++)
      for(j3=j2; j3<4; j3++)
        dgd[j1][j2][j3] = ndg4dn(m, j1, j2, j3, i1, i2, i3);

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
        pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] = 0.0;
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
          pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] += g4up[j1][j4] * cf1[j4][j2][j3];

  for (j1=0; j1<4; j1++)
    for (j2=0; j2<4; j2++)
      for (j3=0; j3<j2; j3++)
        pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j2][j3] =
          pgeodesic[m].cf2g[i4][i5][i6].cf2[j1][j3][j2];
}

/*
  Calculate interpolation coefficients
  int m            Particle index
  const double y[] Particle current state vector (t,x,y,z, u^t,u^x,u^y,u^z)
*/
int cf2_fitting (int m, const double y[])
{
  CCTK_INT i, j1, j2, j3, j4, j5, j6, j7, j8;

  /* threshold for "constant field" detection */
  const double rel_eps = 1e-8;

  /* ---------------- domain check: the 3x3x3 stencil plus the 6th-order
     finite-difference padding must fit inside the grid ----------------
     tpoint[i] is the 1-based cell number of the stencil's upper corner;
     the stencil spans 0-based cells tpoint[i]-2 .. tpoint[i], and the
     6th-order FD (ndg4dn) needs +/-3 more cells around each stencil
     cell, so the whole window [tpoint[i]-5, tpoint[i]+3] must lie in
     the grid: 5 <= tpoint[i] <= lsh[i-1]-4 in every dimension
     (consistent with the 7-cell safe margin in geodesics_integrate). */
  CCTK_INT tpoint[4];
  tpoint[0] = 0;
  for (i=1; i<4; i++)
  {
    tpoint[i] = lrint (floor ((y[i] - origin_space[i-1]) * delta_inv[i-1] + 0.5));
    if (tpoint[i] < 5 || tpoint[i] > lsh[i-1] - 4)
    {
      pgeodesic[m].outbd = 1;
      return 0;
    }
  }

  /* ------------------------------------------------------------------
     need_rebuild == true  <=>  the particle moved into a new 3x3x3
     stencil cube, or this is the first call for it in the current
     Cactus step (flag == 0).  Only in this case do we resample
     cf2_grid and refit the 27-term polynomials.
     ------------------------------------------------------------------ */
  const int need_rebuild = (!pgeodesic[m].flag ||
                             tpoint[1] != pgeodesic[m].point[1] ||
                             tpoint[2] != pgeodesic[m].point[2] ||
                             tpoint[3] != pgeodesic[m].point[3]);

  if (need_rebuild)
  {
    pgeodesic[m].point[1] = tpoint[1];
    pgeodesic[m].point[2] = tpoint[2];
    pgeodesic[m].point[3] = tpoint[3];

    for (i=1; i<4; i++)
    {
      pgeodesic[m].below[i] = origin_space[i-1] + (tpoint[i] - 1) * delta_space[i-1];
      pgeodesic[m].dxp[i-1] = delta_space[i-1];
    }

    /* !sexact: sample metric + Christoffel symbols on the stencil.
       In the sexact branch both are exact Kerr-Schild analytic values,
       evaluated directly by tg4dn/gdn_derivatives inside func_ode, so
       no sampling is needed. */
    if (!sexact)
    {
      for (j1=0; j1<3; j1++)
        for (j2=0; j2<3; j2++)
          for (j3=0; j3<3; j3++)
            cf2_grid(m, j1, j2, j3);
    }
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
  if (!sexact)
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
          ly[j8] = pgeodesic[m].cf2g[j7][j6][j5].cf2[j1][j2][j3];

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
            pgeodesic[m].pcf2[j1][j2][j3][j4] = 0.0;
            for (j5=0; j5<lN; j5++)
              pgeodesic[m].pcf2[j1][j2][j3][j4] += MtxA[j4][j5]*ly[j5];
          }
          pgeodesic[m].pcf2c[j1][j2][j3] = 0;
        }
        else
        {
          for (j4=1; j4<lN; j4++)
            pgeodesic[m].pcf2[j1][j2][j3][j4] = 0.0;

          pgeodesic[m].pcf2[j1][j2][j3][0] = tmpy;
          pgeodesic[m].pcf2c[j1][j2][j3] = 1;
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
          pgeodesic[m].pcf2[j1][j2][j3][j4] = pgeodesic[m].pcf2[j1][j3][j2][j4];
        pgeodesic[m].pcf2c[j1][j2][j3] = pgeodesic[m].pcf2c[j1][j3][j2];
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
        ly[j8] = pgeodesic[m].gd[j7][j6][j5].gd[j1][j2];

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
          pgeodesic[m].pgd[j1][j2][j4] = 0.0;
          for (j5=0; j5<lN; j5++)
            pgeodesic[m].pgd[j1][j2][j4] += MtxA[j4][j5]*ly[j5];
        }
        pgeodesic[m].pgdc[j1][j2] = 0;
      }
      else
      {
        for (j4=1; j4<lN; j4++)
          pgeodesic[m].pgd[j1][j2][j4] = 0.0;

        pgeodesic[m].pgd[j1][j2][0] = tmpy;
        pgeodesic[m].pgdc[j1][j2] = 1;
      }
    }
  }

  /* metric symmetry */
  for (j1=1; j1<4; j1++)
  {
    for (j2=0; j2<j1; j2++)
    {
      for (j4=0; j4<lN; j4++)
        pgeodesic[m].pgd[j1][j2][j4] = pgeodesic[m].pgd[j2][j1][j4];
      pgeodesic[m].pgdc[j1][j2] = pgeodesic[m].pgdc[j2][j1];
    }
  }
  } /* end if (!sexact): Christoffel/metric fitting */

  pgeodesic[m].flag = 1;
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
   pgeodesic[m].norm_bad = 1, and let the loss policy (drop / flag /
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

  if (srverbose)
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

  /* --------- out-of-bounds / refit check (both branches) ---------
     Once a particle leaves the SAFE region (7-cell margin) it is marked
     outbd and handled by the loss policy after the step (same behavior
     in both branches). */
  if ((y[1] != 0.0 && (!isnormal(y[1]))) || (y[2] != 0.0 && (!isnormal(y[2]))) || (y[3] != 0.0 && (!isnormal(y[3]))))
  {
    pgeodesic[m].outbd = 1;
  }else
  {
    /* "out of bounds" = outside the SAFE region (7-cell margin) */
    if ((y[1] < r_lbnd[0] || y[1] > r_ubnd[0]) ||
        (y[2] < r_lbnd[1] || y[2] > r_ubnd[1]) ||
        (y[3] < r_lbnd[2] || y[3] > r_ubnd[2]))
    {
      pgeodesic[m].outbd = 1;
    }else if ((y[1] < (pgeodesic[m].below[1]) || (y[1] > (pgeodesic[m].below[1] + 2*pgeodesic[m].dxp[0]))) ||
              (y[2] < (pgeodesic[m].below[2]) || (y[2] > (pgeodesic[m].below[2] + 2*pgeodesic[m].dxp[1]))) ||
              (y[3] < (pgeodesic[m].below[3]) || (y[3] > (pgeodesic[m].below[3] + 2*pgeodesic[m].dxp[2]))))
    {
      cf2_fitting(m, y);
    }
  }

  if (pgeodesic[m].outbd)
  {
    return GSL_EBADFUNC;
  }

  /* ---------------- build Christoffel (either exact or interpolated) ---------------- */
  if (sexact)
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
      pgeodesic[m].outbd = 1;
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
    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=j2; j3<4; j3++)
          cf2[j1][j2][j3] = cf2_func(y, m, j1, j2, j3);
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
  /* if (!sexact) */
  {
    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=0; j3<4; j3++)
          f[j1+4] -= cf2[j1][j2][j3]*y[j2+4]*y[j3+4];
  }

  /* track du^t/dtau for the tau_end prediction in geodesics_integrate,
     while we are still inside the current Cactus step */
  if (y[0] < odedata->bg + sstep)
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

  /* must be set before the acquire block below (which branches on it) */
  sexact = Exact;

  /* ================================================================================
     Single-grid acquire phase (called once per iteration at CCTK_POSTSTEP).

     Cache the grid-function data POINTERS and geometry of the (single)
     grid into the globals above (no data copy).  The grid data is
     read-only during the particle integration, so all OpenMP threads
     share the cached pointers.
     ================================================================================ */
  {
    for (i=0; i<3; i++)
    {
      lsh[i]          = cctk_lsh[i];
      origin_space[i] = cctk_origin_space[i];
      delta_space[i]  = cctk_delta_space[i];
      delta_inv[i]    = 1.0 / cctk_delta_space[i];

      /* 7-cell safe margin: the 3x3x3 stencil plus 6th-order FD
         padding must stay inside the grid (used by the particle
         safe-region check in both the exact and the interpolated
         branch). */
      r_lbnd[i]  = cctk_origin_space[i] + 7*cctk_delta_space[i];
      r_ubnd[i]  = cctk_origin_space[i] + (cctk_lsh[i]-8)*cctk_delta_space[i];
    }

    /* ---------------- pointers to ADMBase ----------------
       Fetched only for the interpolated branch (Exact = no): the
       exact branch uses the built-in analytic Kerr-Schild metric
       and never reads the grid metric. The variables are still
       declared as READS in schedule.ccl, so ADMBase must be
       present in the run either way (see README, "Required
       thorns"). */
    if (!sexact)
    {
      if (cctk_nghostzones[0] < 4 || cctk_nghostzones[1] < 4 || cctk_nghostzones[2] < 4)
        CCTK_ERROR("The ghost_size less 4 (need at least 4 for 6th-order FD).");

      sGH    = cctkGH;
      sgxx   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gxx")));
      sgxy   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gxy")));
      sgxz   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gxz")));
      sgyy   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gyy")));
      sgyz   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gyz")));
      sgzz   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::gzz")));
      salp   = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::alp")));
      sbetax = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::betax")));
      sbetay = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::betay")));
      sbetaz = (double *)(CCTK_VarDataPtrI(cctkGH,0,CCTK_VarIndex("ADMBASE::betaz")));

      if (!sgxx || !sgxy || !sgxz || !sgyy || !sgyz || !sgzz ||
          !salp || !sbetax || !sbetay || !sbetaz)
        CCTK_ERROR("Geodesic (Exact=no): ADMBase metric/lapse/shift not stored; check the run configuration.");
    }
  }

  /* ==================== integrate particles ==================== */

  /* ---------------- particle cache ----------------
     Allocated once for particle_n_total slots and reused at every
     Cactus step; the per-step fields (flag, outbd, norm_bad, the
     renorm/freeze diagnostics) are reset in the loop below.  The old
     per-step calloc/free churned hundreds of MB of allocation traffic
     per iteration at 10^4 particles. */
  if (pgeodesic == NULL || pgeodesic_capacity < particle_n_total)
  {
    free(pgeodesic);
    pgeodesic = (ParticleGeodesic * ) calloc(particle_n_total, sizeof(ParticleGeodesic));
    if (!pgeodesic)
      CCTK_WARN(CCTK_WARN_ABORT, "Error: calloc(pgeodesic) failed.");
    pgeodesic_capacity = particle_n_total;
  }

  /* ---------------- set run flags / parameters ---------------- */
  sgverbose = gverbose;
  srverbose= rverbose;
  ksm = M;
  ksa = a;
  sinitial_radius = initial_radius;

  /* particle loss policy: 0 = drop, 1 = flag, 2 = respawn
     (param.ccl validates the keyword; drop is the default) */
  if (strcmp(particle_loss_policy, "drop") == 0)
    loss_policy = 0;
  else if (strcmp(particle_loss_policy, "flag") == 0)
    loss_policy = 1;
  else
    loss_policy = 2;

  /* default: use the grid time step */
  if (step == 0.0) sstep = cctk_delta_time;
  else             sstep = step;

  /* ---------------- reset per-step particle fields ----------------
     (dead is persistent: it survives across steps) */
  for (mi=0; mi<particle_n_total; mi++)
  {
    pgeodesic[mi].flag        = 0;
    pgeodesic[mi].outbd       = 0;
    pgeodesic[mi].norm_bad    = 0;
    pgeodesic[mi].renorm_warn = 0;
    pgeodesic[mi].freeze_warn = 0;
    pgeodesic[mi].rl_last     = 0.0;
  }

  /* ================================================================
   * Particle 4-velocity initialization
   *
   * Convention: particle_tv/xv/yv/zv in the parfile are the
   * contravariant 4-velocity components u^mu = (u^t, u^x, u^y, u^z)
   * of the particle in the simulation (Kerr-Schild) coordinates --
   * NOT a 3-velocity.
   *
   * Behavior (runs only on the first CCTK time step):
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
  if (cctk_iteration == 0)
  {
    CCTK_VInfo(CCTK_THORNSTRING, "Kerr M:%.4g a:%.4g Particle total:%d", ksm, ksa, particle_n_total);
    for (int mi = 0; mi < particle_n_total; mi++)
    {
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
      pgeodesic[mi].flag = 0;
      cf2_fitting(mi, ys);

      /* Metric at the initial position:
         sexact:  exact Kerr-Schild analytic metric (evaluated directly).
                  NOTE: cf2_fitting performs no fitting in the sexact
                  branch, so pgd/pgdc keep their calloc'd zero values --
                  do NOT build the metric with poly27_eval(pgd) there
                  (it would give an all-zero metric and a bogus norm).
         !sexact: 27-point interpolation. */
      double gdn_i[4][4];
      if (sexact)
      {
        for (int mu = 0; mu < 4; mu++)
          for (int nu = 0; nu < 4; nu++)
            gdn_i[mu][nu] = tg4dn(0, particle_tx[mi], particle_ty[mi], particle_tz[mi], mu, nu);
      }
      else
      {
        for (int mu = 0; mu < 4; mu++)
          for (int nu = mu; nu < 4; nu++)
            gdn_i[mu][nu] = poly27_eval(ys, mi,
                                        pgeodesic[mi].pgd[mu][nu],
                                        pgeodesic[mi].pgdc[mu][nu]);
        for (int mu = 1; mu < 4; mu++)
          for (int nu = 0; nu < mu; nu++)
            gdn_i[mu][nu] = gdn_i[nu][mu];
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
  }

  /* ==================================================================
     Diagnostic dump file (see the dump block after the particle
     loop): when particle_dump_every > 0, open the file named by
     particle_dump_file at the first iteration and record the initial
     state (t=0) BEFORE any integration step, so the file's first data
     row is the parfile input (after the init check above).  Parent
     directories are created as needed.
     ================================================================== */
  static FILE *dfp = NULL;
  if (particle_dump_every > 0 && cctk_iteration == 0)
  {
    geodesic_mkdir_parents(particle_dump_file);
    dfp = fopen(particle_dump_file, "w");
    if (dfp)
    {
      fprintf(dfp, "# t x y z ut ux uy uz tau\n");
      for (mi = 0; mi < particle_n_total; mi++)
        fprintf(dfp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                (double)particle_tt[mi], (double)particle_tx[mi],
                (double)particle_ty[mi], (double)particle_tz[mi],
                (double)particle_ttv[mi], (double)particle_txv[mi],
                (double)particle_tyv[mi], (double)particle_tzv[mi],
                (double)particle_ttw[mi]);
      fflush(dfp);
    }
    else
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: cannot open %s for writing", particle_dump_file);
  }

  /* ---------------- ODE system ---------------- */

  /* We integrate each particle independently */
#ifndef CCTK_DEBUG
#pragma omp parallel for schedule(dynamic,1) shared(pgeodesic)
#endif
  for (mi=0; mi<particle_n_total; mi++)
  {
    gsl_odeiv2_system sys;
    ode_data odedata;
    double ts, hstart, epsabs, epsrel;
    /* ODE tolerances */
    hstart = 1.0e-8;
    epsabs = 1.0e-8;
    epsrel = 0.0;

    /* particles dropped/flagged by the loss policy are no longer tracked */
    if (pgeodesic[mi].dead)
      continue;

    /* out-of-bounds quick check */
    if ( (particle_tx[mi] != 0.0 && (!isnormal(particle_tx[mi]))) ||
         (particle_ty[mi] != 0.0 && (!isnormal(particle_ty[mi]))) ||
         (particle_tz[mi] != 0.0 && (!isnormal(particle_tz[mi]))) ||
         particle_tx[mi] < r_lbnd[0] || particle_tx[mi] > r_ubnd[0] ||
         particle_ty[mi] < r_lbnd[1] || particle_ty[mi] > r_ubnd[1] ||
         particle_tz[mi] < r_lbnd[2] || particle_tz[mi] > r_ubnd[2] )
    {
      pgeodesic[mi].outbd = 1;
    }

    /* ---------------- integrate one particle ---------------- */
    memset((void *)&odedata, 0, sizeof(odedata));
    odedata.m  = mi;
    odedata.bg = particle_tt[mi];

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

      if (sexact)
      {
        /* analytic metric at particle position (Kerr-Schild exact branch) */
        const double tx = ys[1];
        const double ty = ys[2];
        const double tz = ys[3];
        for (int mu=0; mu<4; ++mu)
        {
          for (int nu=mu; nu<4; ++nu)
          {
            gdn[mu][nu] = tg4dn(0, tx, ty, tz, mu, nu);
          }
        }
        for (int mu=1; mu<4; ++mu)
          for (int nu=0; nu<mu; ++nu)
            gdn[mu][nu] = gdn[nu][mu];
      }
      else
      {
        /* ensure interpolation coefficients cover current position */
        cf2_fitting(mi, ys);
        for (int mu=0; mu<4; ++mu)
        {
          for (int nu=mu; nu<4; ++nu)
          {
            gdn[mu][nu] = poly27_eval(ys, mi,
                                     pgeodesic[mi].pgd[mu][nu],
                                     pgeodesic[mi].pgdc[mu][nu]);
          }
        }
        for (int mu=1; mu<4; ++mu)
          for (int nu=0; nu<mu; ++nu)
            gdn[mu][nu] = gdn[nu][mu];
      }

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
        pgeodesic[mi].renorm_warn = 1;   /* aggregated after the loop */
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
          pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
          pgeodesic[mi].outbd = 1;   /* let the loss policy below handle this particle */
        }
      }
      u[0] = ys[4]; u[1] = ys[5]; u[2] = ys[6]; u[3] = ys[7];
      u_cov[0] = 0; u_cov[1] = 0; u_cov[2] = 0; u_cov[3] = 0;
      for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
          u_cov[mu] += gdn[mu][nu]*u[nu];
      rl = 0.0;
      for (int mu=0; mu<4; ++mu) rl += u_cov[mu]*u[mu];

      pgeodesic[mi].rl_last = rl;   /* reported after the loop */
    }

    ts = particle_ttw[mi];

    /* integrate in tau until coordinate time reaches target */

    const double t_target = odedata.bg + sstep;
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
      double tau_end = ts + sstep; /* fallback cap; normally overwritten below */

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
        pgeodesic[mi].outbd = 1;
        ys[0] = t_target;
        break;
      }

      const double r2 = ys[1]*ys[1] + ys[2]*ys[2] + ys[3]*ys[3];
      if (r2 < excised_radius * excised_radius)
      {
        pgeodesic[mi].outbd = 1;
      }

      if (pgeodesic[mi].outbd)
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

        if (sexact)
        {
          /* analytic metric at particle position (Kerr-Schild exact branch) */
          const double tx = ys[1];
          const double ty = ys[2];
          const double tz = ys[3];
          for (int mu=0; mu<4; ++mu)
          {
            for (int nu=mu; nu<4; ++nu)
            {
              gdn[mu][nu] = tg4dn(0, tx, ty, tz, mu, nu);
            }
          }
          for (int mu=1; mu<4; ++mu)
            for (int nu=0; nu<mu; ++nu)
              gdn[mu][nu] = gdn[nu][mu];
        }
        else
        {
          /* make sure interpolation coefficients cover current position */
          cf2_fitting(mi, ys);

          for (int mu=0; mu<4; ++mu)
          {
            for (int nu=mu; nu<4; ++nu)
            {
              gdn[mu][nu] = poly27_eval(ys, mi,
                                     pgeodesic[mi].pgd[mu][nu],
                                     pgeodesic[mi].pgdc[mu][nu]);
            }
          }
          for (int mu=1; mu<4; ++mu)
            for (int nu=0; nu<mu; ++nu)
              gdn[mu][nu] = gdn[nu][mu];
        }

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
          pgeodesic[mi].outbd = 1;
          pgeodesic[mi].renorm_warn = 1;   /* aggregated after the loop */
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
            pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
            pgeodesic[mi].outbd = 1;
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

        pgeodesic[mi].rl_last = rl;   /* reported after the loop */
      }

      if (fabs(ys[0] - t_target) < 1e-13 * fmax(1.0, fabs(t_target)))
        break;
    }
    if (max_iter > 100000)
    {
      pgeodesic[mi].outbd = 1;
      pgeodesic[mi].freeze_warn = 1;   /* aggregated after the loop */
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
      if (pgeodesic[mi].renorm_warn)
      {
        n_renorm++;
        if (n_first < 5) first_renorm[n_first] = mi;
        n_first++;
      }
      if (pgeodesic[mi].norm_bad)    n_normbad++;
      if (pgeodesic[mi].freeze_warn) n_freeze++;
      /* "pure" out-of-bounds losses (safe region, excised sphere, ODE
         failure) have no other diagnostic; report them so that a
         dropped/flagged particle is never silent */
      if (pgeodesic[mi].outbd && !pgeodesic[mi].norm_bad &&
          !pgeodesic[mi].renorm_warn && !pgeodesic[mi].freeze_warn)
        n_outbd++;
      if (pgeodesic[mi].rl_last != 0.0)
      {
        if (!rl_have)
        {
          rl_min = rl_max = pgeodesic[mi].rl_last;
          rl_have = 1;
        }
        else
        {
          if (pgeodesic[mi].rl_last < rl_min) rl_min = pgeodesic[mi].rl_last;
          if (pgeodesic[mi].rl_last > rl_max) rl_max = pgeodesic[mi].rl_last;
        }
      }
    }
    if (n_renorm)
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) could not be renormalized (u.u >= 0 or non-finite) and will be %s; first: %d %d %d %d %d",
                 n_renorm, loss_txt, first_renorm[0], first_renorm[1], first_renorm[2], first_renorm[3], first_renorm[4]);
    if (n_normbad)
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) had no future-directed u^0 solution (possibly superluminal spatial velocity) and will be %s",
                 n_normbad, loss_txt);
    if (n_freeze)
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) hit the tau-loop iteration cap and will be %s",
                 n_freeze, loss_txt);
    if (n_outbd)
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Geodesic: %d particle(s) left the safe region or failed the ODE step and will be %s",
                 n_outbd, loss_txt);
    if (srverbose && rl_have)
      CCTK_VInfo(CCTK_THORNSTRING, "rl: min:%18.15G max:%18.15G", rl_min, rl_max);
  }

  /* ==================================================================
     Diagnostic dump: append the full particle state
     (t, x, y, z, u^t, u^x, u^y, u^z, tau) to the file named by
     particle_dump_file, once every particle_dump_every iterations
     (0 disables).  The file was opened and seeded with the t=0
     initial state before the first integration step (see above).
     Runs after the parallel particle loop, single-threaded (this
     build is single-process: the particle array lives in the one
     Cactus process, parallelism is OpenMP inside it); purely
     diagnostic, no effect on the integration.
     ================================================================== */
  if (particle_dump_every > 0 &&
      (cctk_iteration % particle_dump_every == 0))
  {
    if (dfp == NULL)
    {
      /* defensive fallback (e.g. the initial open failed earlier) */
      geodesic_mkdir_parents(particle_dump_file);
      dfp = fopen(particle_dump_file, "w");
      if (dfp)
        fprintf(dfp, "# t x y z ut ux uy uz tau\n");
    }
    if (dfp)
    {
      for (mi = 0; mi < particle_n_total; mi++)
        fprintf(dfp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                (double)particle_tt[mi], (double)particle_tx[mi],
                (double)particle_ty[mi], (double)particle_tz[mi],
                (double)particle_ttv[mi], (double)particle_txv[mi],
                (double)particle_tyv[mi], (double)particle_tzv[mi],
                (double)particle_ttw[mi]);
      fflush(dfp);
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
     ================================================================== */
  for (mi=0; mi<particle_n_total; mi++)
  {
    if (pgeodesic[mi].norm_bad)
      CCTK_VInfo(CCTK_THORNSTRING,
             "Particle %d flagged norm_bad this step (see warnings above).", mi);

    if (pgeodesic[mi].outbd || pgeodesic[mi].norm_bad)
    {
      if (loss_policy == 2)
      {
        particle_ttw[mi] = 0.0;
        /* particle_tt[mi]  = 0.0; */

        double er;
        long int i1 = 0;
        do
        {
          particle_tx[mi]  = urand_range(&seed, xmin_safe, xmax_safe);
          particle_ty[mi]  = urand_range(&seed, ymin_safe, ymax_safe);
          particle_tz[mi]  = urand_range(&seed, zmin_safe, zmax_safe);
          er = particle_tx[mi]*particle_tx[mi] + particle_ty[mi]*particle_ty[mi] + particle_tz[mi]*particle_tz[mi];
          i1++;
        }while ((fabs(particle_tz[mi]) < particle_middle_sp || er < (excised_radius + 1.5)*(excised_radius + 1.5) || er > sinitial_radius*sinitial_radius) && i1 < 100000);

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
        pgeodesic[mi].dead = 0;
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
        pgeodesic[mi].dead = 1;
      }
      else
      {
        /* flag: keep the last valid state, stop tracking */
        pgeodesic[mi].dead = 1;
      }
      pgeodesic[mi].outbd = 0;
      pgeodesic[mi].norm_bad = 0;
    }
  }

  /* the per-particle cache (pgeodesic) is kept and reused at the next
     Cactus step; it is reclaimed by the OS at the end of the run */
}


