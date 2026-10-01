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

void Geodesics_init(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  int i;

  /* CCTK_INITIAL is traversed once per refinement level in local mode
     (Carpet).  Particle initialisation must run exactly once, so the
     guard below keeps only the level-0 call of map 0 (the center
     block owns all particles); on a plain unigrid run levfac is 1
     and the guard is a no-op.  Only map 0's level-0 call is the
     legitimate init call in a multipatch run. */
  if ((int)lrint(log2((double)cctk_levfac[0])) != 0) return;
  /* Guard: MultiPatch thorn is only active in multipatch runs.  In a plain
     single-patch (unigrid/AMR) run the alias is not registered, so we must
     not call MultiPatch_GetMap; treat the run as map 0. */
  if (CCTK_IsFunctionAliased("MultiPatch_GetMap") && MultiPatch_GetMap(cctkGH) != 0)
    return;

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

    gstate.xmin_safe = cctk_origin_space[0] + margin * cctk_delta_space[0];
    gstate.xmax_safe = cctk_origin_space[0] + (cctk_ubnd[0] - cctk_lbnd[0] - margin) * cctk_delta_space[0];
    if (gstate.xmin_safe < particle_rand_xmin && particle_rand_xmin < gstate.xmax_safe && particle_rand_xmin <= particle_rand_xmax) gstate.xmin_safe = particle_rand_xmin;
    if (gstate.xmax_safe > particle_rand_xmax && particle_rand_xmax > gstate.xmin_safe && particle_rand_xmin <= particle_rand_xmax) gstate.xmax_safe = particle_rand_xmax;

    gstate.ymin_safe = cctk_origin_space[1] + margin * cctk_delta_space[1];
    gstate.ymax_safe = cctk_origin_space[1] + (cctk_ubnd[1] - cctk_lbnd[1] - margin) * cctk_delta_space[1];
    if (gstate.ymin_safe < particle_rand_ymin && particle_rand_ymin < gstate.ymax_safe && particle_rand_ymin <= particle_rand_ymax) gstate.ymin_safe = particle_rand_ymin;
    if (gstate.ymax_safe > particle_rand_ymax && particle_rand_ymax > gstate.ymin_safe && particle_rand_ymin <= particle_rand_ymax) gstate.ymax_safe = particle_rand_ymax;

    gstate.zmin_safe = cctk_origin_space[2] + margin * cctk_delta_space[2];
    gstate.zmax_safe = cctk_origin_space[2] + (cctk_ubnd[2] - cctk_lbnd[2] - margin) * cctk_delta_space[2];
    if (gstate.zmin_safe < particle_rand_zmin && particle_rand_zmin < gstate.zmax_safe && particle_rand_zmin <= particle_rand_zmax) gstate.zmin_safe = particle_rand_zmin;
    if (gstate.zmax_safe > particle_rand_zmax && particle_rand_zmax > gstate.zmin_safe && particle_rand_zmin <= particle_rand_zmax) gstate.zmax_safe = particle_rand_zmax;

    /* AMR region centre / half-extent, used by the acquire phase to rebuild
       each fine-level patch origin (see Geodesic_integrate.c).  Defaults are
       (0,0,0) centre and zero radii; set the radii in the par for multi-rank
       AMR runs so the split directions are centred on the rank's portion. */
    gstate.amr_center[0] = amr_center_x;
    gstate.amr_center[1] = amr_center_y;
    gstate.amr_center[2] = amr_center_z;
    gstate.amr_radius[0] = amr_radius_x;
    gstate.amr_radius[1] = amr_radius_y;
    gstate.amr_radius[2] = amr_radius_z;

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

      /* inward (centripetal) radial 4-velocity kick so circular orbits
         visibly deviate; 0 disables (see particle_radial_kick in param.ccl) */
      if (particle_radial_kick > 0.0)
      {
        double r = sqrt(particle_tx[i]*particle_tx[i] +
                        particle_ty[i]*particle_ty[i] +
                        particle_tz[i]*particle_tz[i]);
        if (r > 1.0e-12)
        {
          double kr = -(particle_radial_kick) / r;
          particle_txv[i] += kr * particle_tx[i];
          particle_tyv[i] += kr * particle_ty[i];
          particle_tzv[i] += kr * particle_tz[i];
        }
      }
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

      /* inward (centripetal) radial 4-velocity kick so circular orbits
         visibly deviate; 0 disables (see particle_radial_kick in param.ccl) */
      if (particle_radial_kick > 0.0)
      {
        double r = sqrt(particle_tx[i]*particle_tx[i] +
                        particle_ty[i]*particle_ty[i] +
                        particle_tz[i]*particle_tz[i]);
        if (r > 1.0e-12)
        {
          double kr = -(particle_radial_kick) / r;
          particle_txv[i] += kr * particle_tx[i];
          particle_tyv[i] += kr * particle_ty[i];
          particle_tzv[i] += kr * particle_tz[i];
        }
      }
    }
  }
}
