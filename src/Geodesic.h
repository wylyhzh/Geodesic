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
  Geodesic.h   (pure geodesic integrator, single grid)

  Data structures used by Geodesic_integrate.c to compute:
    - geodesic motion (via Christoffel symbols)
    - 4-velocity normalization (u_mu u^mu = -1)

  Grid model (single process, multi-thread, one unigrid):

    * CCTK_POSTSTEP is called once per iteration.  geodesics_integrate
      caches the grid-function data pointers and grid geometry into
      plain globals (sGH, sgxx, delta_space, lsh, r_lbnd, ...).

    * The cached data is read-only during the particle integration, so
      all OpenMP threads share it.  Legacy code that uses
      CCTK_GFINDEX3D(sGH,...) (e.g. g4dn() in derivative.h) works
      against cctkGH itself.
*/

#include "cctk.h"

/* -------------------- Polynomial interpolation order (fixed) -------------------- */
#ifndef LAGRANGE_POINTS_3
#define LAGRANGE_POINTS_3 3
#endif

#ifndef POLY27
#define POLY27 27
#endif

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
  CCTK_INT  point[4]; /* base point grid (i,j,k) */
  CCTK_REAL below[4]; /* polynomial interpolation base point (physical coords) */
  CCTK_REAL dxp[3]; /* grid spacing at the particle's current stencil */

  /* ---- values on 3x3x3 stencil (grid-aligned) ---- */
  cf2_data    cf2g[3][3][3]; /* Christoffel symbols (2nd kind) at grid points */
  gd_data     gd[3][3][3]; /* 4-metric g_{mu nu} at grid points */

  /* ---- polynomial coefficients (27-term) for fast evaluation in func_ode ---- */
  CCTK_REAL pcf2[4][4][4][POLY27]; /* Christoffel polynomial interpolation coeffs */
  CCTK_INT  pcf2c[4][4][4]; /* Christoffel is constant? */

  CCTK_REAL pgd[4][4][POLY27]; /* 4-metric polynomial interpolation coeffs */
  CCTK_INT  pgdc[4][4]; /* 4-metric is constant? */

  /* ---- per-step diagnostics: written inside the OpenMP particle loop
     (each particle by exactly one thread), aggregated single-threaded
     after the loop in geodesics_integrate ---- */
  CCTK_INT  renorm_warn;   /* u.u could not be renormalized this step */
  CCTK_INT  freeze_warn;   /* tau-loop iteration cap hit */
  CCTK_REAL rl_last;       /* last computed u_mu u^mu (diagnostic) */

} ParticleGeodesic;

/* -------------------- Per-step ODE parameters for func_ode -------------------- */
typedef struct ODE_DATA {
  int m;          /* particle index */
  CCTK_REAL bg;   /* coordinate time at the start of this Cactus step */
  CCTK_REAL a;    /* du^t/dtau, tracked by func_ode for the tau_end prediction */
} ode_data;

#endif /* GEODESIC_H */
