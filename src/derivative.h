
/*
  Geodesic -- pure geodesic particle tracker (Cactus/Einstein Toolkit thorn)
  Copyright (C) 2026  Youhua Li

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
    - g4dn: grid metric access via the cached unigrid pointers (sGH)
*/

double gdn_derivatives(int d, double x, double y, double z, int j1, int j2);
double tg4dn(double t, double x, double y, double z, int j1, int j2);
double g4dn(int m, int j1, int j2, int ix, int iy, int iz);

/* MtxA: 27x27 coefficient matrix of the 27-point Lagrange interpolant.
   Nodes {-1,0,1} per axis (centered stencil; middle node = center cell center,
   matching the cell-center sample positions). row = monomial index (same order
   as cf2_func/poly27_eval), col = iz*9+iy*3+ix (x fastest). */
  double MtxA[27][27] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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

double gdn_derivatives(int d, double x, double y, double z, int j1, int j2)
{
  double ret, a;
  int k1, k2;
  ret = 0.0;
  a = ksa;
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
              1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*ksm/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = -4.0*(ksa*ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 3.0*sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
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
              y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*((-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))) + 
              2*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*((-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*x/sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))) + 2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - 
              x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*(-1.0*(a*a - x*x - y*y - 
              z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = 1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*ksm*x*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 4.0*(ksa*ksa*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) +
              3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(ksa*y + sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*x)*ksm*(1.0*z +
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }
    } else if (k2 == 2)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 
              3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z)))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*x - 
              sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 1.0*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*((-1.0*(a*a - 
              x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))) - 
              2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
       }else if (d == 2)
      {
        ret = 4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - 
              x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + 
              (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + ksa*ksa + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 
              1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))))*((-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*y/sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - 
              y*y - z*z))) + 2*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = 1.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*ksm*y*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 4.0*(ksa*ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - 
              y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*x - 
              sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*y)*ksm/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 
              0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*sqrt((-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*y)*ksm*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - 
              x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 
              0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 
              ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*(-1.0*(a*a - x*x - y*y - z*z)*x/sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*x)*ksm*z/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 2)
      {
        ret = -4.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y -
              z*z)))*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 
              1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*(-1.0*(a*a - x*x - y*y - z*z)*y/sqrt(4.0*a*a*z*z + (a*a - 
              x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)) + 1.0*y)*ksm*z/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))));
      }else if (d == 3)
      {
        ret = -4.0*(ksa*ksa*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - 
              x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z))))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))*ksm*z/((ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
              0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 
              0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z))))) + 2.0*ksm*(1.0*z + 
              0.25*(8.0*a*a*z - 4*(a*a - x*x - y*y - z*z)*z)/sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - 
              z*z)))*z/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - y*y - 
              z*z)*(a*a - x*x - y*y - z*z)))*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + (a*a - x*x - 
              y*y - z*z)*(a*a - x*x - y*y - z*z)))) + 2.0*(-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*a*a*z*z + 
              (a*a - x*x - y*y - z*z)*(a*a - x*x - y*y - z*z)))*ksm/(ksa*ksa*z*z + (-0.50*a*a + 0.50*x*x + 0.50*y*y + 0.50*z*z +
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
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*
            (ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*
            sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 
            4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(ksa*y + 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*((-1.0*(ksa*ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*
            x/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))) + 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*((-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*x/sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) + 
            2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 
            3.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*x*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(ksa*ksa*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 
            4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y +
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*x)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    }else if (k2 == 2)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*
            (ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z +
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))) - 1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y -
            z*z)))*y)*((-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*((-1.0*(ksa*ksa - x*x - 
            y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))) - 2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) -
            1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*((-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*x/sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) + 
            2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 
            3.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*y + 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*x)*((-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)) + 1.0*y)*y/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*x*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*ksm*y*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 
            4.0*(ksa*ksa*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 3.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 
            1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*((-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)) + 1.0*x)*x/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm*z/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*((-1.0*(ksa*ksa - 
            x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 
            1.0*y)*x/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*ksa)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(ksa*y +
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)) + 1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 1.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*ksm*x*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(ksa*ksa*z +
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 
            2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*x)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*y + sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    }
  } else if (k1 == 2)
  {
    if (k2 == 2)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) +
            3.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))) - 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*((-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) - 
            2*ksa)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) +
            3.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y -
            z*z)))*y)*((-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*y)*y/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*y*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(ksa*ksa*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 
            4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 3.0*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 
            4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 4.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y -
            z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 
            4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    } else if (k2 == 3)
    {
      if (d == 1)
      {
        ret = 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - 
            sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)) + 1.0*x)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 
            1.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*((-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)) + 1.0*x)*y/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))) - 2*ksa)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = 4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z +
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - x*x - y*y - 
            z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z/((ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*(-1.0*(ksa*ksa - 
            x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 
            1.0*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 1.0*(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*((-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)) + 1.0*y)*y/sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))) + 2*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = 1.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*ksm*y*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 4.0*(ksa*ksa*z +
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 
            2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*y)*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))) - 2.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(ksa*x - sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*y)*ksm/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    }
  } else if (k1 == 3)
  {
    if (k2 == 3)
    {
      if (d == 1)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))) + 1.0*(-1.0*(ksa*ksa - x*x - y*y - z*z)*x/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*x)*ksm*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 2)
      {
        ret = -4.0*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z)))*sqrt((-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z))))*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*(ksa*ksa*z*z + 
            (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - 
            y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z))))) + 1.0*(-1.0*(ksa*ksa - x*x - y*y - z*z)*y/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - 
            y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)) + 1.0*y)*ksm*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*sqrt(-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      } else if (d == 3)
      {
        ret = -4.0*(ksa*ksa*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - 
            z*z)*(ksa*ksa - x*x - y*y - z*z)))*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - z*z)*z)/sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z +
            0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*ksm*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - 
            x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))))) + 1.0*ksm*(1.0*z + 0.25*(8.0*ksa*ksa*z - 4*(ksa*ksa - x*x - y*y - 
            z*z)*z)/sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*z*z/((ksa*ksa*z*z + (-0.50*ksa*ksa + 
            0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - 
            z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - 
            x*x - y*y - z*z))))*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y -
            z*z)*(ksa*ksa - x*x - y*y - z*z)))) + 4.0*sqrt(-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + 
            (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*ksm*z/(ksa*ksa*z*z + (-0.50*ksa*ksa + 0.50*x*x + 0.50*y*y + 
            0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z)))*(-0.50*ksa*ksa + 0.50*x*x + 
            0.50*y*y + 0.50*z*z + 0.50*sqrt(4.0*ksa*ksa*z*z + (ksa*ksa - x*x - y*y - z*z)*(ksa*ksa - x*x - y*y - z*z))));
      }
    }
  }
  return ret;
}

double tsgtt(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*r/(ksa*ksa*z*z+r*r*r*r)-1.0);
}

double tsgtx(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*r*(ksa*y+r*x)/((ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r)));
}

double tsgty(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*r*(r*y-ksa*x)/((ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r)));
}

double tsgtz(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*z/(ksa*ksa*z*z+r*r*r*r));
}

double tsgxx(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*r*(ksa*y+r*x)*(ksa*y+r*x)/((ksa*ksa+r*r)*(ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r))+1.0);
}

double tsgyy(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*r*r*(r*y-ksa*x)*(r*y-ksa*x)/((ksa*ksa+r*r)*(ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r))+1.0);
}

double tsgzz(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return (2.0*ksm*r*z*z/(ksa*ksa*z*z+r*r*r*r)+1.0);
}

double tsgxy(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return 2.0*ksm*r*r*r*(ksa*y+r*x)*(r*y-ksa*x)/((ksa*ksa+r*r)*(ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r));
}

double tsgxz(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return 2.0*ksm*r*r*z*(ksa*y+r*x)/((ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r));
}

double tsgyz(double t, double x, double y, double z)
{
  double r1, r;
  r1 = x*x+y*y+z*z-ksa*ksa;
  r = sqrt((sqrt(r1*r1+4.0*ksa*ksa*z*z)+r1)/2.0);
  return 2.0*ksm*r*r*z*(r*y-ksa*x)/((ksa*ksa+r*r)*(ksa*ksa*z*z+r*r*r*r));
}

double tg4dn(double t, double x, double y, double z, int j1, int j2)
{
  double ret;
  int i;
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

double g4dn(int m, int j1, int j2, int ix, int iy, int iz)
{
  double ret;
  int index, i1, i2;
  
  if (ix < 1 || iy < 1 || iz < 1 ||
    ix > lsh[0] || iy > lsh[1] || iz > lsh[2])
  {
    CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
      "g4dn: index out of bounds ix=%d iy=%d iz=%d lsh=(%d,%d,%d)",
      ix, iy, iz, (int)lsh[0], (int)lsh[1], (int)lsh[2]);
    return 0.0;
  }

  index = CCTK_GFINDEX3D(sGH, ix-1, iy-1, iz-1);
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
    double gxxL, gxyL, gxzL, gyyL, gyzL, gzzL;
    double lapse, LAPSE_SQUARED, shiftx, shifty, shiftz, shift_x, shift_y, shift_z;

    gxxL = sgxx[index];
    gxyL = sgxy[index];
    gxzL = sgxz[index];
    gyyL = sgyy[index];
    gyzL = sgyz[index];
    gzzL = sgzz[index];
    lapse = salp[index];
    LAPSE_SQUARED = lapse*lapse;
    shiftx = sbetax[index];
    shifty = sbetay[index];
    shiftz = sbetaz[index];
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
      ret = sgxx[index];
    } else if (i2 == 2)
    {
      ret = sgxy[index];
    } else if (i2 == 3)
    {
      ret = sgxz[index];
    }
  } else if (i1 == 2)
  {
    if (i2 == 1)
    {
      ret = sgxy[index];
    } else if (i2 == 2)
    {
      ret = sgyy[index];
    } else if (i2 == 3)
    {
      ret = sgyz[index];
    }
  } else if (i1 == 3)
  {
    if (i2 == 1)
    {
      ret = sgxz[index];
    } else if (i2 == 2)
    {
      ret = sgyz[index];
    } else if (i2 == 3)
    {
      ret = sgzz[index];
    }
  }
  return ret;
}


