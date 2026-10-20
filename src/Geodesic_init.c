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
   Geodesic_init.c  (pure geodesic integrator)
   - initializes particle state arrays from param.ccl arrays
   - fills particle_n ... particle_n_total with random initial conditions
   ============================================================================================ */

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "Geodesic.h"

/* MultiPatch (Coordinates thorn) prototypes, declared because Geodesic's
   include chain does not include Coordinates_Prototypes.h.  Calls are
   guarded by CCTK_IsFunctionAliased; see Geodesic_integrate.c for the full
   comment.  GCC>=12 errors on implicit function declaration, so declare them. */
extern CCTK_INT MultiPatch_GetMap (const CCTK_POINTER_TO_CONST cctkGH);

void Geodesics_init(CCTK_ARGUMENTS);

/* ============================================================================================
   Small random helpers
   ============================================================================================ */

double urand01(unsigned int *seed)
{
  return rand_r(seed) / (double)RAND_MAX;
}

double urand_range(unsigned int *seed, double a, double b)
{
  return a + (b - a) * urand01(seed);
}

/* Set by Geodesics_init_carpetx while it forwards the call, so the guard in
   Geodesics_init knows the call already comes from the GLOBAL-mode entry. */
static int cx_init_global_call = 0;

void Geodesics_init(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  /* Without a mode keyword the driver uses its default scheduling: Carpet
     calls once per grid component per level, CarpetX once per tile per
     level.  Under CarpetX the seeding is done by Geodesics_init_carpetx
     (GLOBAL mode, exactly one call per run) instead, so bail out here. */
  if (CCTK_IsThornActive("CarpetX") && !cx_init_global_call) return;

  int i;

  /* CCTK_INITIAL is traversed once per refinement level in local mode
     (Carpet).  Keep only the level-0 call so the arrays are seeded exactly
     once per run.  NOTE: the particle arrays are DISTRIB=CONSTANT, i.e.
     each rank owns an independent copy, so the seed MUST run on EVERY rank
     (not just map 0) or the non-master ranks are left with uninitialised
     (garbage) arrays that the mmpi sync then propagates as NaN/dead
     particles.  The old "map 0 only" MultiPatch guard was therefore wrong.
     On a plain unigrid run levfac[0]==1 so the level guard is a no-op.
     cctk_levfac is a level-mode-only variable: the CarpetX global-mode cGH
     never sets it, and under CarpetX there is only one synthetic level, so
     the guard is skipped there. */
  if (!CCTK_IsThornActive ("CarpetX") &&
      (int)lrint(log2((double)cctk_levfac[0])) != 0) return;

  /* Echo the settings that actually took effect so a particle dump can always
     be traced back to the parameter file that produced it. */
  {
    char cfg[512];
    snprintf(cfg, sizeof(cfg),
             "revision %s: effective config: excised_radius=%g initial_radius=%g"
             " Exact=%s metric_interp=%s particle_loss_policy=%s"
             " particle_n_total=%d particle_dump_file=%s",
             GEODESIC_REVISION,
             (double)excised_radius, (double)initial_radius,
             Exact ? "yes" : "no", metric_interp, particle_loss_policy,
             (int)particle_n_total, particle_dump_file);
    CCTK_INFO(cfg);
  }

  if (particle_n > particle_n_total)
    CCTK_WARN(CCTK_WARN_ABORT, "Error: particle_n > particle_n_total.");
  if (particle_rand_seed)
  {
    gstate.seed = particle_rand_seed;
  }

  /* ==========================================================================================
     Safe random ranges:
     - avoid the outer 7 cells because Geodesic_integrate uses high-order stencils
     ========================================================================================== */
  {
    const double margin = 7.0;

    /* CarpetX never calls Geodesic in level mode, so the cGH geometry is
       unset here (cctk_origin_space is NaN, cctk_lbnd/ubnd undefined); take
       the domain from the CarpetX parameters instead.  Vertex-centred:
       ncells+1 points, so the last index is ncells. */
    double org[3], dsp[3];
    int npt[3];
    if (CCTK_IsThornActive("CarpetX"))
    {
      double lo[3], hi[3];
      int nc[3], ngz, d;
      if (geodesic_carpetx_domain(lo, hi, nc, &ngz) != 0)
        CCTK_WARN(CCTK_WARN_ABORT,
                  "Geodesic (metric_interp = carpetx): invalid CarpetX domain "
                  "(ncells_* must be >= 2 and xmax > xmin and so on).");
      for (d=0; d<3; d++)
      {
        org[d] = lo[d];
        dsp[d] = (hi[d] - lo[d]) / (double)nc[d];
        npt[d] = nc[d];
      }
    }
    else
    {
      int d;
      for (d=0; d<3; d++)
      {
        org[d] = cctk_origin_space[d];
        dsp[d] = cctk_delta_space[d];
        npt[d] = (int)(cctk_ubnd[d] - cctk_lbnd[d]);
      }
    }

    gstate.xmin_safe = org[0] + margin * dsp[0];
    gstate.xmax_safe = org[0] + (npt[0] - margin) * dsp[0];
    if (gstate.xmin_safe < particle_rand_xmin && particle_rand_xmin < gstate.xmax_safe && particle_rand_xmin <= particle_rand_xmax) gstate.xmin_safe = particle_rand_xmin;
    if (gstate.xmax_safe > particle_rand_xmax && particle_rand_xmax > gstate.xmin_safe && particle_rand_xmin <= particle_rand_xmax) gstate.xmax_safe = particle_rand_xmax;

    gstate.ymin_safe = org[1] + margin * dsp[1];
    gstate.ymax_safe = org[1] + (npt[1] - margin) * dsp[1];
    if (gstate.ymin_safe < particle_rand_ymin && particle_rand_ymin < gstate.ymax_safe && particle_rand_ymin <= particle_rand_ymax) gstate.ymin_safe = particle_rand_ymin;
    if (gstate.ymax_safe > particle_rand_ymax && particle_rand_ymax > gstate.ymin_safe && particle_rand_ymin <= particle_rand_ymax) gstate.ymax_safe = particle_rand_ymax;

    gstate.zmin_safe = org[2] + margin * dsp[2];
    gstate.zmax_safe = org[2] + (npt[2] - margin) * dsp[2];
    if (gstate.zmin_safe < particle_rand_zmin && particle_rand_zmin < gstate.zmax_safe && particle_rand_zmin <= particle_rand_zmax) gstate.zmin_safe = particle_rand_zmin;
    if (gstate.zmax_safe > particle_rand_zmax && particle_rand_zmax > gstate.zmin_safe && particle_rand_zmin <= particle_rand_zmax) gstate.zmax_safe = particle_rand_zmax;

    /* fixed seed => reproducible runs
       if you want different random values every run, replace with:
         unsigned int seed = (unsigned int)time(NULL);
    */

    /* ----------------------------------------------------------------------------------------
       1) initialize the first particle_n particles from parfile arrays
       ---------------------------------------------------------------------------------------- */
    for (i = 0; i < particle_n; i++)
    {
      particle_ttw[i] = 0.0;
      particle_tt[i]  = 0.0;

      particle_tx[i]  = particle_x[i];
      particle_ty[i]  = particle_y[i];
      particle_tz[i]  = particle_z[i];

      particle_ttv[i] = particle_tv[i];
      particle_txv[i] = particle_xv[i];
      particle_tyv[i] = particle_yv[i];
      particle_tzv[i] = particle_zv[i];


    }

    /* ----------------------------------------------------------------------------------------
       2) initialize particle_n ... particle_n_total with random initial conditions:
          uniform in the safe box, rejecting near the polar axis, inside the
          excision sphere, and outside the initial shell; zero spatial velocity.
          u^mu is a placeholder (u^t = 2.5, u^i = 0): Geodesic_integrate's
          init block re-solves u^0 from u_mu u^mu = -1 with u^i fixed.
       ---------------------------------------------------------------------------------------- */
    for (i = particle_n; i < particle_n_total; i++)
    {
      particle_ttw[i] = 0.0;
      particle_tt[i]  = 0.0;

      double er;
      long int i1 = 0;
      do
      {
        particle_tx[i]  = urand_range(&gstate.seed, gstate.xmin_safe, gstate.xmax_safe);
        particle_ty[i]  = urand_range(&gstate.seed, gstate.ymin_safe, gstate.ymax_safe);
        particle_tz[i]  = urand_range(&gstate.seed, gstate.zmin_safe, gstate.zmax_safe);
        er = particle_tx[i]*particle_tx[i] + particle_ty[i]*particle_ty[i] + particle_tz[i]*particle_tz[i];
        i1++;
      }while ((fabs(particle_tz[i]) < particle_middle_sp || er < (excised_radius + 1.5)*(excised_radius + 1.5) || er > initial_radius*initial_radius) && i1 < 100000);

      if (i1 >= 100000)
      {
        CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: initial placement rejection loop exhausted "
                  "(check excised_radius/initial_radius/rand box geometry).");
      }
      /* u^t placeholder (u^i = 0); Geodesic_integrate's init block
         re-solves u^0 from u_mu u^mu = -1 with u^i fixed */
      particle_ttv[i] = 2.5;
      particle_txv[i] = 0.0;
      particle_tyv[i] = 0.0;
      particle_tzv[i] = 0.0;


    }
  }
}

/* GLOBAL-mode entry point for CarpetX (see schedule.ccl): exactly one call
   per run, with the driver's global-mode cGH.  No-op under Carpet, where the
   per-level Geodesics_init above does the work. */
void Geodesics_init_carpetx (CCTK_ARGUMENTS)
{
  if (!CCTK_IsThornActive ("CarpetX")) return;

  cx_init_global_call = 1;
  Geodesics_init (cctkGH);
  cx_init_global_call = 0;
}
