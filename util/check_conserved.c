/*
  check_conserved -- Geodesic thorn testsuite postprocessor.

  Reads a particle state dump (the file written by the Geodesic
  thorn's particle_dump_file option, columns:

      t x y z u^t u^x u^y u^z tau

  one row per particle per dump) and prints, for every data row, the
  two constants of motion of the Kerr metric evaluated with the exact
  Kerr--Schild formulas:

      E   = -(g_tt u^t + g_tx u^x + g_ty u^y + g_tz u^z)   (energy)
      L_z = x p_y - y p_x                                  (axial angular
                                                           momentum, with
                                                           p_mu = g_mu nu u^nu)

  The test parfiles run a prograde equatorial circular orbit with
  M = 2.0, a = 0.5, so the constants are hard-coded here.  On a
  correct run both columns are constant to integrator precision;
  the testsuite compares the post-processed output against the
  reference data row by row.

  Build (plain C, libc + libm only):
      cc -O2 -o check_conserved check_conserved.c -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_KS 2.0
#define A_KS 0.5

int main(int argc, char **argv)
{
  FILE *fp;
  char line[1024];

  if (argc != 2)
  {
    fprintf(stderr, "usage: %s <particle-dump-file>\n", argv[0]);
    return 1;
  }

  fp = fopen(argv[1], "r");
  if (!fp)
  {
    fprintf(stderr, "check_conserved: cannot open %s\n", argv[1]);
    return 1;
  }

  while (fgets(line, sizeof(line), fp))
  {
    double t, x, y, z, ut, ux, uy, uz, tau;
    double a = A_KS, Mk = M_KS;
    double r1, r, r2, a2, a2z2, r4, a2r2, den, gtt, gtx, gty, gtz;
    double gxx, gxy, gyy, gxz, gyz;
    double E, px, py, Lz;

    if (line[0] == '#' || line[0] == '\n' || line[0] == '"')
      continue;

    if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &t, &x, &y, &z, &ut, &ux, &uy, &uz, &tau) != 9)
    {
      fprintf(stderr, "check_conserved: bad line: %s", line);
      return 1;
    }

    /* Kerr--Schild radius: r^2 is the positive root of
       r^4 - (x^2+y^2+z^2 - a^2) r^2 - a^2 z^2 = 0 */
    a2 = a * a;
    r1 = x * x + y * y + z * z - a2;
    r2 = sqrt(r1 * r1 + 4.0 * a2 * z * z);
    r  = sqrt((r2 + r1) / 2.0);          /* r = coordinate radius */

    a2z2 = a2 * z * z;
    r4   = r * r * r * r;
    a2r2 = a2 + r * r;
    den  = a2z2 + r4;

    gtt = 2.0 * Mk * r * r * r / den - 1.0;
    gtx = 2.0 * Mk * r * r * r * (a * y + r * x) / (a2r2 * den);
    gty = 2.0 * Mk * r * r * r * (r * y - a * x) / (a2r2 * den);
    gtz = 2.0 * Mk * r * r * z / den;

    gxx = 1.0 + 2.0 * Mk * r * r * r * (a * y + r * x) * (a * y + r * x)
          / (a2r2 * a2r2 * den);
    gxy = 2.0 * Mk * r * r * r * (a * y + r * x) * (r * y - a * x)
          / (a2r2 * a2r2 * den);
    gyy = 1.0 + 2.0 * Mk * r * r * r * (r * y - a * x) * (r * y - a * x)
          / (a2r2 * a2r2 * den);
    gxz = 2.0 * Mk * r * r * z * (a * y + r * x) / (a2r2 * den);
    gyz = 2.0 * Mk * r * r * z * (r * y - a * x) / (a2r2 * den);

    E  = -(gtt * ut + gtx * ux + gty * uy + gtz * uz);
    px = gxx * ux + gxy * uy + gxz * uz;
    py = gxy * ux + gyy * uy + gyz * uz;
    Lz = x * py - y * px;

    printf("%.15g %.15g\n", E, Lz);
  }

  fclose(fp);
  return 0;
}
