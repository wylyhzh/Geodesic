
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

/*
  derivative.h -- metric/Christoffel helpers for the Geodesic thorn:
    - MtxA: 27x27 coefficient matrix of the 27-point Lagrange interpolant
    - gdn_derivatives / tg4dn: exact Kerr-Schild metric and derivatives
      (metric parameters come from the global thorn state gstate)
    - g4dn: grid metric access at a given time level, through the
      per-thread level view gview
  Included only by Geodesic_integrate.c; it references the gstate and
  gview objects defined there.
*/

static CCTK_REAL gdn_derivatives(CCTK_INT d, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_INT j1, CCTK_INT j2);
static CCTK_REAL tg4dn(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_INT j1, CCTK_INT j2);
static CCTK_REAL g4dn(CCTK_INT tl, CCTK_INT j1, CCTK_INT j2, CCTK_INT ix, CCTK_INT iy, CCTK_INT iz);

/* MtxA: 27x27 coefficient matrix of the 27-point Lagrange interpolant.
   Nodes {-1,0,1} per axis (centered stencil; middle node = center cell center,
   matching the cell-center sample positions). row = monomial index (same order
   as cf2_func/poly27_eval), col = iz*9+iy*3+ix (x fastest). */
static const CCTK_REAL MtxA[27][27] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/2.0, 0.0,
                         1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/2.0, -1.0,
                         1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, -1.0,
                         0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, 0.0, -1.0/4.0, 0.0, 0.0,
                         0.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 1.0/4.0, 0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0, 0.0},
                         {0.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 0.0, 0.0,
                         0.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 0.0, 1.0/4.0, 1.0/2.0, 0.0,
                         -1.0/2.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, -1.0/4.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 1.0/4.0, 0.0},
                         {0.0, 0.0, 0.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/2.0, 0.0,
                         -1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0, 0.0},
                         {0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, 0.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, -1.0/2.0, 0.0, 0.0, -1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, 0.0},
                         {-1.0/8.0, 0.0, 1.0/8.0, 0.0, 0.0, 0.0, 1.0/8.0, 0.0, -1.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 1.0/8.0, 0.0, -1.0/8.0, 0.0, 0.0, 0.0, -1.0/8.0, 0.0, 1.0/8.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, -1.0/2.0, 1.0,
                         -1.0/2.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/2.0, 1.0,
                         -1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0, 0.0},
                         {0.0, 1.0/4.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 1.0/4.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, -1.0/2.0, 0.0, 0.0, 1.0/4.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 1.0/4.0, 0.0},
                         {1.0/8.0, -1.0/4.0, 1.0/8.0, 0.0, 0.0, 0.0, -1.0/8.0, 1.0/4.0, -1.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, -1.0/8.0, 1.0/4.0, -1.0/8.0, 0.0, 0.0, 0.0, 1.0/8.0, -1.0/4.0, 1.0/8.0},
                         {1.0/8.0, 0.0, -1.0/8.0, -1.0/4.0, 0.0, 1.0/4.0, 1.0/8.0, 0.0, -1.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, -1.0/8.0, 0.0, 1.0/8.0, 1.0/4.0, 0.0, -1.0/4.0, -1.0/8.0, 0.0, 1.0/8.0},
                         {1.0/8.0, 0.0, -1.0/8.0, 0.0, 0.0, 0.0, -1.0/8.0, 0.0, 1.0/8.0, -1.0/4.0, 0.0, 1.0/4.0, 0.0, 0.0,
                         0.0, 1.0/4.0, 0.0, -1.0/4.0, 1.0/8.0, 0.0, -1.0/8.0, 0.0, 0.0, 0.0, -1.0/8.0, 0.0, 1.0/8.0},
                         {-1.0/8.0, 1.0/4.0, -1.0/8.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, -1.0/8.0, 1.0/4.0, -1.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 1.0/8.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/8.0},
                         {-1.0/8.0, 1.0/4.0, -1.0/8.0, 0.0, 0.0, 0.0, 1.0/8.0, -1.0/4.0, 1.0/8.0, 1.0/4.0, -1.0/2.0, 1.0/4.0, 0.0, 0.0,
                         0.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, -1.0/8.0, 1.0/4.0, -1.0/8.0, 0.0, 0.0, 0.0, 1.0/8.0, -1.0/4.0, 1.0/8.0},
                         {-1.0/8.0, 0.0, 1.0/8.0, 1.0/4.0, 0.0, -1.0/4.0, -1.0/8.0, 0.0, 1.0/8.0, 1.0/4.0, 0.0, -1.0/4.0, -1.0/2.0, 0.0,
                         1.0/2.0, 1.0/4.0, 0.0, -1.0/4.0, -1.0/8.0, 0.0, 1.0/8.0, 1.0/4.0, 0.0, -1.0/4.0, -1.0/8.0, 0.0, 1.0/8.0},
                         {1.0/8.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 1.0/2.0, -1.0,
                         1.0/2.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/2.0, -1.0/4.0, 1.0/8.0, -1.0/4.0, 1.0/8.0}};

static CCTK_REAL gdn_derivatives(CCTK_INT d, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_INT j1, CCTK_INT j2)
{
  CCTK_REAL ret, a;
  CCTK_INT k1, k2;
  ret = 0.0;
  a = gstate.ksa;
  if (j1 > j2)
  {
    k1 = j2;
    k2 = j1;
  } else
  {
    k1 = j1;
    k2 = j2;
  }

  if (d == 0)
  {
    ret = 0.0;
  } else if (k1 == 0)
  {
    if (k2 == 0)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = -4.0*(gstate.ksa*gstate.ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*gstate.ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))));
      }
    } else if (k2 == 1)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*((-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))) + 
              2*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*((-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*x/sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))) + 2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - 
              x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - 
              z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = 1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*gstate.ksm*x*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 4.0*(gstate.ksa*gstate.ksa*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) +
              3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(gstate.ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z +
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }
    } else if (k2 == 2)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 
              3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*x - 
              sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*((-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))) - 
              2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
       }else if (d == 2)
      {
        ret = 4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - 
              x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 
              1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*((-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*y/sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))) + 2*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = 1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*gstate.ksm*y*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 4.0*(gstate.ksa*gstate.ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - 
              y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*x - 
              sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*gstate.ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*gstate.ksm*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - 
              x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = -4.0*(gstate.ksa*gstate.ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*gstate.ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*z/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*gstate.ksm/(gstate.ksa*gstate.ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }
    }
  } else if (k1 == 1)
  {
    if (k2 == 1)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*
            (gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*
            sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 
            4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(gstate.ksa*y + 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*((-1.0*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*
            x/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))) + 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*x/sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 
            2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 
            3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*x*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(gstate.ksa*gstate.ksa*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 
            4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y +
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*x)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    }else if (k2 == 2)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z +
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))) - 1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)))*y)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*((-1.0*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))) - 2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) -
            1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*x/sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 
            2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 
            3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*y + 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*x)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)) + 1.0*y)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*x*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*gstate.ksm*y*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 
            4.0*(gstate.ksa*gstate.ksa*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 
            1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*((-1.0*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 
            1.0*y)*x/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*gstate.ksa)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(gstate.ksa*y +
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 1.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*gstate.ksm*x*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(gstate.ksa*gstate.ksa*z +
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 
            2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*x)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*y + sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    }
  } else if (k1 == 2)
  {
    if (k2 == 2)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) +
            3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))) - 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) - 
            2*gstate.ksa)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) +
            3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)))*y)*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*y)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*y*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(gstate.ksa*gstate.ksa*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 
            4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 
            4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 
            4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - 
            sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 
            1.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) - 2*gstate.ksa)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = 4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z +
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*(-1.0*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 
            1.0*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*((-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)) + 1.0*y)*y/sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 1.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*gstate.ksm*y*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 4.0*(gstate.ksa*gstate.ksa*z +
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 
            2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*y)*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(gstate.ksa*x - sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*y)*gstate.ksm/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    }
  } else if (k1 == 3)
  {
    if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))) + 1.0*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*x/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*x)*gstate.ksm*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + 
            (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))) + 1.0*(-1.0*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*y/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - 
            y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)) + 1.0*y)*gstate.ksm*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*sqrt(-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -4.0*(gstate.ksa*gstate.ksa*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*gstate.ksm*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))))) + 1.0*gstate.ksm*(1.0*z + 0.25*(8.0*gstate.ksa*gstate.ksa*z - 4*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*z*z/((gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - 
            z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - 
            x*x - y*y - z*z))))*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y -
            z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))) + 4.0*sqrt(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + 
            (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*gstate.ksm*z/(gstate.ksa*gstate.ksa*z*z + (-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z)))*(-0.50*gstate.ksa*gstate.ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*gstate.ksa*gstate.ksa*z*z + (gstate.ksa*gstate.ksa - x*x - y*y - z*z)*(gstate.ksa*gstate.ksa - x*x - y*y - z*z))));
      }
    }
  }
  return ret;
}

static CCTK_REAL tsgtt(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*r/(gstate.ksa*gstate.ksa*z*z+r*r*r*r)-1.0);
}

static CCTK_REAL tsgtx(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*r*(gstate.ksa*y+r*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r)));
}

static CCTK_REAL tsgty(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*r*(r*y-gstate.ksa*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r)));
}

static CCTK_REAL tsgtz(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*z/(gstate.ksa*gstate.ksa*z*z+r*r*r*r));
}

static CCTK_REAL tsgxx(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*r*(gstate.ksa*y+r*x)*(gstate.ksa*y+r*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r))+1.0);
}

static CCTK_REAL tsgyy(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*r*r*(r*y-gstate.ksa*x)*(r*y-gstate.ksa*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r))+1.0);
}

static CCTK_REAL tsgzz(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return (2.0*gstate.ksm*r*z*z/(gstate.ksa*gstate.ksa*z*z+r*r*r*r)+1.0);
}

static CCTK_REAL tsgxy(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return 2.0*gstate.ksm*r*r*r*(gstate.ksa*y+r*x)*(r*y-gstate.ksa*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r));
}

static CCTK_REAL tsgxz(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return 2.0*gstate.ksm*r*r*z*(gstate.ksa*y+r*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r));
}

static CCTK_REAL tsgyz(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z)
{
  CCTK_REAL r1, r;
  r1 = x*x+y*y+z*z-gstate.ksa*gstate.ksa;
  r = sqrt((sqrt(r1*r1+4.0*gstate.ksa*gstate.ksa*z*z)+r1)/2.0);
  return 2.0*gstate.ksm*r*r*z*(r*y-gstate.ksa*x)/((gstate.ksa*gstate.ksa+r*r)*(gstate.ksa*gstate.ksa*z*z+r*r*r*r));
}

static CCTK_REAL tg4dn(CCTK_REAL t, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_INT j1, CCTK_INT j2)
{
  CCTK_REAL ret;
  CCTK_INT i;
  ret = 0.0;
  if (j1 > j2)
  {
    i = j1;
    j1 = j2;
    j2 = i;
  }
  if (j1 == 0)
  {
    if (j2 == 0)
    {
      ret = tsgtt(t, x, y, z);
    } else if (j2 == 1)
    {
      ret = tsgtx(t, x, y, z);
    } else if (j2 == 2)
    {
      ret = tsgty(t, x, y, z);
    } else if (j2 == 3)
    {
      ret = tsgtz(t, x, y, z);
    }
  } else if (j1 == 1)
  {
    if (j2 == 1)
    {
      ret = tsgxx(t, x, y, z);
    } else if (j2 == 2)
    {
      ret = tsgxy(t, x, y, z);
    } else if (j2 == 3)
    {
      ret = tsgxz(t, x, y, z);
    }
  } else if (j1 == 2)
  {
    if (j2 == 1)
    {
      ret = tsgxy(t, x, y, z);
    } else if (j2 == 2)
    {
      ret = tsgyy(t, x, y, z);
    } else if (j2 == 3)
    {
      ret = tsgyz(t, x, y, z);
    }
  } else if (j1 == 3)
  {
    if (j2 == 1)
    {
      ret = tsgxz(t, x, y, z);
    } else if (j2 == 2)
    {
      ret = tsgyz(t, x, y, z);
    } else if (j2 == 3)
    {
      ret = tsgzz(t, x, y, z);
    }
  }
  return ret;
}

static CCTK_REAL g4dn(CCTK_INT tl, CCTK_INT j1, CCTK_INT j2, CCTK_INT ix, CCTK_INT iy, CCTK_INT iz)
{
  CCTK_REAL ret;
  CCTK_INT index, i1, i2;

  if (ix < 1 || iy < 1 || iz < 1 ||
    ix > gview.lsh[0] || iy > gview.lsh[1] || iz > gview.lsh[2])
  {
    CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
      "g4dn: index out of bounds ix=%d iy=%d iz=%d lsh=(%d,%d,%d)",
      ix, iy, iz, gview.lsh[0], gview.lsh[1], gview.lsh[2]);
    return 0.0;
  }

  index = CCTK_GFINDEX3D(gview.GH, ix-1, iy-1, iz-1);
  if (j1 > j2)
  {
    i1 = j2;
    i2 = j1;
  }else
  {
    i1 = j1;
    i2 = j2;
  }
  ret = 0.0;
  if (i1 == 0)
  {
    CCTK_REAL gxxL, gxyL, gxzL, gyyL, gyzL, gzzL;
    CCTK_REAL lapse, LAPSE_SQUARED, shiftx, shifty, shiftz, shift_x, shift_y, shift_z;

    gxxL = gview.gxx[tl][index];
    gxyL = gview.gxy[tl][index];
    gxzL = gview.gxz[tl][index];
    gyyL = gview.gyy[tl][index];
    gyzL = gview.gyz[tl][index];
    gzzL = gview.gzz[tl][index];
    lapse = gview.alp[tl][index];
    LAPSE_SQUARED = lapse*lapse;
    shiftx = gview.betax[tl][index];
    shifty = gview.betay[tl][index];
    shiftz = gview.betaz[tl][index];
    shift_x = gxxL*shiftx + gxyL*shifty + gxzL*shiftz;
    shift_y = gxyL*shiftx + gyyL*shifty + gyzL*shiftz;
    shift_z = gxzL*shiftx + gyzL*shifty + gzzL*shiftz;

    if (i2 == 0)
    {
      ret = -LAPSE_SQUARED + (shiftx*shift_x + shifty*shift_y + shiftz*shift_z);
    } else if (i2 == 1)
    {
      ret = shift_x;
    } else if (i2 == 2)
    {
      ret = shift_y;
    } else if (i2 == 3)
    {
      ret = shift_z;
    }
  } else if (i1 == 1)
  {
    if (i2 == 1)
    {
      ret = gview.gxx[tl][index];
    } else if (i2 == 2)
    {
      ret = gview.gxy[tl][index];
    } else if (i2 == 3)
    {
      ret = gview.gxz[tl][index];
    }
  } else if (i1 == 2)
  {
    if (i2 == 1)
    {
      ret = gview.gxy[tl][index];
    } else if (i2 == 2)
    {
      ret = gview.gyy[tl][index];
    } else if (i2 == 3)
    {
      ret = gview.gyz[tl][index];
    }
  } else if (i1 == 3)
  {
    if (i2 == 1)
    {
      ret = gview.gxz[tl][index];
    } else if (i2 == 2)
    {
      ret = gview.gyz[tl][index];
    } else if (i2 == 3)
    {
      ret = gview.gzz[tl][index];
    }
  }
  return ret;
}


