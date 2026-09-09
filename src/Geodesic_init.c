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

extern unsigned int seed;
extern double xmin_safe;
extern double xmax_safe;
extern double ymin_safe;
extern double ymax_safe;
extern double zmin_safe;
extern double zmax_safe;

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
     guard below keeps only the level-0 call; on a plain unigrid run
     levfac is 1 and the guard is a no-op. */
  if ((int)lrint(log2((double)cctk_levfac[0])) != 0) return;

  if (particle_n > particle_n_total)
    CCTK_WARN(CCTK_WARN_ABORT, "Error: particle_n > particle_n_total.");

  if (particle_rand_seed)
  {
    seed = particle_rand_seed;
  }

  /* ==========================================================================================
     Safe random ranges:
     - avoid the outer 7 cells because Geodesic_integrate uses high-order stencils
     ========================================================================================== */
  {
    const double margin = 7.0;

    xmin_safe = cctk_origin_space[0] + margin * cctk_delta_space[0];
    xmax_safe = cctk_origin_space[0] + (cctk_ubnd[0] - cctk_lbnd[0] - margin) * cctk_delta_space[0];
    if (xmin_safe < particle_rand_xmin && particle_rand_xmin < xmax_safe && particle_rand_xmin <= particle_rand_xmax) xmin_safe = particle_rand_xmin;
    if (xmax_safe > particle_rand_xmax && particle_rand_xmax > xmin_safe && particle_rand_xmin <= particle_rand_xmax) xmax_safe = particle_rand_xmax;

    ymin_safe = cctk_origin_space[1] + margin * cctk_delta_space[1];
    ymax_safe = cctk_origin_space[1] + (cctk_ubnd[1] - cctk_lbnd[1] - margin) * cctk_delta_space[1];
    if (ymin_safe < particle_rand_ymin && particle_rand_ymin < ymax_safe && particle_rand_ymin <= particle_rand_ymax) ymin_safe = particle_rand_ymin;
    if (ymax_safe > particle_rand_ymax && particle_rand_ymax > ymin_safe && particle_rand_ymin <= particle_rand_ymax) ymax_safe = particle_rand_ymax;

    zmin_safe = cctk_origin_space[2] + margin * cctk_delta_space[2];
    zmax_safe = cctk_origin_space[2] + (cctk_ubnd[2] - cctk_lbnd[2] - margin) * cctk_delta_space[2];
    if (zmin_safe < particle_rand_zmin && particle_rand_zmin < zmax_safe && particle_rand_zmin <= particle_rand_zmax) zmin_safe = particle_rand_zmin;
    if (zmax_safe > particle_rand_zmax && particle_rand_zmax > zmin_safe && particle_rand_zmin <= particle_rand_zmax) zmax_safe = particle_rand_zmax;

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
        particle_tx[i]  = urand_range(&seed, xmin_safe, xmax_safe);
        particle_ty[i]  = urand_range(&seed, ymin_safe, ymax_safe);
        particle_tz[i]  = urand_range(&seed, zmin_safe, zmax_safe);
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
