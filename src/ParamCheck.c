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

/*@@
   @file      ParamCheck.c
   @date      July 2026
   @author    Yuhua Li
   @desc
              Check parameters for Geodesic
   @enddesc
 @@*/


#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include <stdbool.h>

extern bool check_velocity;

static const char *rcsid = "$Header$";

CCTK_FILEVERSION(Geodesic_ParamCheck_c)


void Geodesic_ParamCheck(CCTK_ARGUMENTS);

void Geodesic_ParamCheck(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  /* NOTE: Exact = yes uses the built-in analytic Kerr-Schild metric
     (parameters M and a of this thorn), so there is no dependency on
     the ADMBase::initial_data parameter here. */
  if (particle_n > particle_n_total)
  {
     CCTK_PARAMWARN("particle_n is larger then particle_n_total.");
  }

   check_velocity = true;
}

