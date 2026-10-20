
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
#include <stdlib.h>                    /* atoi (GEODESIC_ENGINE_N debug cap) */
#include <stdio.h>                     /* fopen/fprintf for the particle dump */
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

/* --------------------------------------------------------------------
   MultiPatch (Coordinates thorn) function prototypes.

   Geodesic uses the Coordinates thorn's MultiPatch functions only when that
   thorn is present (all calls are guarded by CCTK_IsFunctionAliased).
   Geodesic's own include chain (cctk.h -> Geodesic_Prototypes.h) does not
   include Coordinates_Prototypes.h, so the calls would otherwise be
   implicitly declared.  GCC>=12 rejects implicit function declaration as
   an error; declare the two functions explicitly here.  The runtime guards
   ensure no call is ever made when the functions are not registered.
   -------------------------------------------------------------------- */
extern CCTK_INT MultiPatch_GetMap (const CCTK_POINTER_TO_CONST cctkGH);
extern CCTK_INT MultiPatch_GetMaps (const CCTK_POINTER_TO_CONST cctkGH);
extern CCTK_INT MultiPatch_GlobalToLocal (const CCTK_POINTER_TO_CONST cctkGH ,
 const CCTK_INT ndims ,
 const CCTK_INT npoints ,
 const CCTK_POINTER_TO_CONST globalcoords ,
 CCTK_INT* patch ,
 const CCTK_POINTER localcoords ,
 const CCTK_POINTER dadx ,
 const CCTK_POINTER ddadxdx);
extern CCTK_INT MultiPatch_LocalToGlobal (const CCTK_POINTER_TO_CONST cctkGH ,
 const CCTK_INT ndims ,
 const CCTK_INT npoints ,
 const CCTK_INT* patch ,
 const CCTK_POINTER_TO_CONST localcoords ,
 const CCTK_POINTER globalcoords ,
 const CCTK_POINTER dxda ,
 const CCTK_POINTER det_dxda ,
 const CCTK_POINTER dadx ,
 const CCTK_POINTER ddxdada ,
 const CCTK_POINTER ddadxdx ,
 const CCTK_POINTER dddxdadada);

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
GeodesicState gstate = { .seed = 123456789u,
                         .l0_seen_iter = -1, .last_integ_iter = -1,
                         .last_dump_iter = -1,
                         .last_integ_time = 0.0,
                         .batch_iter = -1, .batch_maxlvl = -1,
                         .samr_levels = 1, .nlev_seen = 1,
                         .aei_handle = -1, .aei_coordsys = -1 };

/* ============================ AMR support (max GEODESIC_MAX_LEVELS levels) ====

   Single process, multi-thread.  geodesics_integrate is called once per
   refinement level at CCTK_POSTSTEP (Carpet level-mode traversal); each
   call only stores the level's data POINTERS + geometry into LD[map][level]
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
   switch_to_level() from gstate.LD[map][level].  All grid access in this
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
  /* stride the arrays this view points at are laid out with: lsh for the
     raw patch arrays, lsh+2*HALO_W when the view points at the
     halo-padded copies */
  CCTK_INT rsh[3];
  /* valid read-index range in the local frame: the raw patch when no
     halo is active, [-HALO_W, lsh-1+HALO_W] when the view points at
     the halo-padded arrays */
  CCTK_INT halo_lo[3], halo_hi[3];
  CCTK_INT map, level, gen;
} GeodesicView;
static GEODESIC_TLS GeodesicView gview = { .map = -1, .level = -1, .gen = -1 };

/* linear index into the current thread's level arrays (i,j,k are 0-based) */
#define LD_INDEX(i,j,k) ((i) + gview.rsh[0]*((j) + gview.rsh[1]*(k)))

/* point this thread's grid view at (map, level l) (self-guarded, cheap) */
static void switch_to_level(int map, int l)
{
  if (gview.map == map && gview.level == l && gview.gen == gstate.ld_generation) return;
  gview.map = map;
  gview.level = l;
  gview.gen   = gstate.ld_generation;

  LevelData *L = &gstate.LD[map][l];
  gview.GH = &L->gh;
  {
    /* halo[0][0] non-NULL  <=>  this level has halo-padded buffers
       (multi-process grid mode with a stencil sampler); the view then
       reads the padded copies, whose pointers are offset so that
       LD_INDEX with LOCAL indices addresses the halo band as well */
    const int use_halo = (L->halo[0][0] != NULL);
    CCTK_REAL **pp[10] = { L->gxx, L->gxy, L->gxz, L->gyy, L->gyz, L->gzz,
                           L->alp, L->betax, L->betay, L->betaz };
    CCTK_REAL **qq[10] = { gview.gxx, gview.gxy, gview.gxz, gview.gyy, gview.gyz, gview.gzz,
                           gview.alp, gview.betax, gview.betay, gview.betaz };
    for (int i=0; i<10; i++)
      for (int tl=0; tl<3; tl++)
        qq[i][tl] = use_halo ? L->halo[i][tl] : pp[i][tl];
  }
  gview.dt = L->dt;
  gview.metric_static = L->metric_static;
  for (int i=0; i<3; i++)
  {
    gview.lsh[i]         = L->lsh[i];
    gview.rsh[i]         = (L->halo[0][0] != NULL)
                             ? L->lsh[i] + 2 * GEODESIC_HALO_W : L->lsh[i];
    gview.delta_space[i] = L->dx[i];
    gview.delta_inv[i]   = L->inv_dx[i];
    gview.origin_space[i]= L->origin[i];
    gview.r_lbnd[i]      = L->r_lbnd[i];
    gview.r_ubnd[i]      = L->r_ubnd[i];
    if (L->halo[0][0] != NULL)
    {
      gview.halo_lo[i] = L->halo_lo[i];
      gview.halo_hi[i] = L->halo_hi[i];
    }
    else
    {
      gview.halo_lo[i] = 0;
      gview.halo_hi[i] = L->lsh[i] - 1;
    }
  }
}

/* 1 if the metric stored at time levels 0 and 2 agrees (interior cells,
   every 4th cell, all 10 fields, relative tolerance 1e-12): a
   stationary metric, so d_t g = 0 and the time-derivative FD can be
   skipped. */
static int metric_is_static(const LevelData *L)
{
  CCTK_REAL *const *f[10] = {
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

/* ============================================================================
   Multi-process stencil halo (grid mode, l27/local27 samplers)

   The stencil samplers read a window centred on the particle's nearest node:
   [tp-5, tp+5] for local27 (5x5x5 nodes + 6th-order FD metric derivatives),
   [tp-4, tp+4] for l27.  With a domain decomposition, a particle near a rank
   face has its window sticking out of the local patch: the patch's own ghost
   zones (3 cells) are too thin, and clipping the centre to the patch breaks
   the stencil's centring -- equatorial particles on a z=0 rank face acquired
   a spurious vertical velocity (the bug reported by E. Schnetter: MPI runs
   disagreeing with scalar runs).

   Fix: keep a halo-padded copy of the ten ADMBase fields (all three time
   levels) per level, and exchange the outer HALO_W-cell band with the
   face-adjacent ranks at every acquire.  The view then reads the padded
   arrays (switch_to_level), the stencil centre stays exactly centred on the
   particle in every decomposition, and the rank count stops being an
   accuracy parameter.  Values across a face are bit-identical copies of the
   neighbour's own array (same-level face exchange), so 1/2/4-rank runs of
   the same par file agree bit-for-bit.

   Geometry (per direction d, global patch indices, ghost zones included):
   the patch's INTERIOR is [lbnd+ngz, ubnd-ngz]; its interior faces are the
   rank faces.  Each rank needs HALO_W cells beyond each interior face; the
   neighbour across the face owns exactly those global cells in its interior
   or ghost band, and sends them verbatim.  The same global cells are
   computed from both sides, so the Sendrecv counts match on both ranks.

   Called from the acquire phase for every rank and every level, in the same
   (level-mode) order on all ranks, so its collectives are safe.  No-op when
   single-process or with a pointwise sampler (aei/carpet) or Exact=yes.
   ============================================================================ */
static void geodesic_halo_update(cGH *cctkGH, int map, int level)
{
#ifdef CCTK_MPI
  const int npr = CCTK_nProcs(cctkGH);
  const int me  = CCTK_MyProc(cctkGH);
#else
  const int npr = 1, me = 0;
#endif
  LevelData *L = &gstate.LD[map][level];
  const int HW = GEODESIC_HALO_W;

  if (npr <= 1 || gstate.sexact ||
      (gstate.sinterp_mode != 0 && gstate.sinterp_mode != 3))
    return;

  /* ---- size (and (re)allocate) the padded buffers when lsh changed ---- */
  int ash[3];
  size_t pstride = 1, slab_max = 1;
  for (int d = 0; d < 3; d++)
  {
    ash[d] = (int)L->lsh[d] + 2 * HW;
    pstride *= (size_t)ash[d];
    L->halo_lo[d] = -HW;
    L->halo_hi[d] = (int)L->lsh[d] - 1 + HW;
  }
  if (L->halo_alloc_lsh[0] != L->lsh[0] ||
      L->halo_alloc_lsh[1] != L->lsh[1] ||
      L->halo_alloc_lsh[2] != L->lsh[2])
  {
    for (int f = 0; f < 10; f++)
      for (int tl = 0; tl < 3; tl++)
        free(L->halo_base[f][tl]);
    for (int f = 0; f < 10; f++)
      for (int tl = 0; tl < 3; tl++)
      {
        /* offset so that halo[f][tl][LD_HALO_INDEX(i,j,k)] addresses the
           local index (i,j,k) in [-HW, lsh-1+HW] */
        CCTK_REAL *b = (CCTK_REAL *)malloc(pstride * sizeof(CCTK_REAL));
        if (!b)
          CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: halo buffer allocation failed.");
        /* poison with NaN: halo cells that are never written (corners,
           transverse out-of-overlap strips) then read as NaN, exactly like
           g4dn's out-of-range return on a single rank, so the isfinite
           scans discard the fit the same way */
        for (size_t q = 0; q < pstride; q++) b[q] = NAN;
        L->halo_base[f][tl] = b;
        L->halo[f][tl] = b + (size_t)HW * (1 + (size_t)ash[0] * (1 + (size_t)ash[1]));
      }
    for (int d = 0; d < 3; d++) L->halo_alloc_lsh[d] = L->lsh[d];
  }
  const size_t hs0 = (size_t)ash[0], hs1 = (size_t)ash[1];
#define LD_HALO_INDEX(i,j,k) \
  ((size_t)(i) + hs0 * ((size_t)(j) + hs1 * (size_t)(k)))

  /* ---- copy the local patch (its ghost zones included: they hold the
          correct same-level face values) into the padded interior ---- */
  CCTK_REAL **raw[10] = { L->gxx, L->gxy, L->gxz, L->gyy, L->gyz, L->gzz,
                          L->alp, L->betax, L->betay, L->betaz };
  for (int f = 0; f < 10; f++)
    for (int tl = 0; tl < 3; tl++)
    {
      CCTK_REAL *dst = L->halo[f][tl];
      const CCTK_REAL *src = raw[f][tl];
      if (src == NULL)
      {
        /* past time level not readable (ADMBase timelevels < 3): poison
           the whole padded array with NaN.  The static-metric path never
           reads these levels; a read that does happen is caught by the
           isfinite scans (a silent 0.0 would pass them and corrupt the
           fit).  The exchange below skips this (f,tl) on both ranks. */
        for (CCTK_INT k = L->halo_lo[2]; k <= L->halo_hi[2]; k++)
          for (CCTK_INT j = L->halo_lo[1]; j <= L->halo_hi[1]; j++)
            for (CCTK_INT i = L->halo_lo[0]; i <= L->halo_hi[0]; i++)
              dst[LD_HALO_INDEX(i, j, k)] = NAN;
        continue;
      }
      for (CCTK_INT k = 0; k < L->lsh[2]; k++)
        for (CCTK_INT j = 0; j < L->lsh[1]; j++)
          memcpy(dst + LD_HALO_INDEX(0, j, k),
                 src + ((size_t)k * (size_t)L->lsh[1] + (size_t)j) * (size_t)L->lsh[0],
                 (size_t)L->lsh[0] * sizeof(CCTK_REAL));
    }

#ifdef CCTK_MPI
  /* ---- gather every rank's patch index box for this level ---- */
  int *bb = (int *)malloc((size_t)npr * 6 * sizeof(int));
  if (!bb)
    CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: halo box buffer allocation failed.");
  for (int d = 0; d < 3; d++)
  {
    bb[me * 6 + d]     = (int)L->lbnd[d];
    bb[me * 6 + 3 + d] = (int)L->ubnd[d];
  }
  MPI_Allgather(bb + me * 6, 6, MPI_INT, bb, 6, MPI_INT, MPI_COMM_WORLD);

  /* worst-case slab volume: HW cells thick, at most the local transverse
     extent wide */
  for (int d = 0; d < 3; d++)
  {
    size_t vol = (size_t)HW;
    for (int e = 0; e < 3; e++)
      if (e != d) vol *= (size_t)L->lsh[e];
    if (vol > slab_max) slab_max = vol;
  }
  CCTK_REAL *sbuf = (CCTK_REAL *)malloc(slab_max * sizeof(CCTK_REAL));
  CCTK_REAL *rbuf = (CCTK_REAL *)malloc(slab_max * sizeof(CCTK_REAL));
  if (!sbuf || !rbuf)
    CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: halo slab buffer allocation failed.");

  if (getenv("GEODESIC_HALO_DEBUG"))
  {
    fprintf(stderr, "HALODBG rank=%d lvl=%d lsh=(%d,%d,%d) lbnd=(%d,%d,%d) ubnd=(%d,%d,%d) ngz=(%d,%d,%d)\n",
            me, level, L->lsh[0], L->lsh[1], L->lsh[2],
            (int)L->lbnd[0], (int)L->lbnd[1], (int)L->lbnd[2],
            (int)L->ubnd[0], (int)L->ubnd[1], (int)L->ubnd[2],
            (int)L->nghostzones[0], (int)L->nghostzones[1], (int)L->nghostzones[2]);
    for (int r = 0; r < npr; r++)
      fprintf(stderr, "HALODBG rank=%d box r=%d lo=(%d,%d,%d) hi=(%d,%d,%d)\n",
              me, r, bb[r * 6], bb[r * 6 + 1], bb[r * 6 + 2],
              bb[r * 6 + 3], bb[r * 6 + 4], bb[r * 6 + 5]);
  }

  for (int d = 0; d < 3; d++)
  for (int s = 0; s < 2; s++)
  for (int r = 0; r < npr; r++)
  {
    if (r == me) continue;
    const int nb = r;
    /* The padded copy of the raw array fills the halo only over the raw
       (interior+ghost) range [lbnd, ubnd]; the halo cells that still hold
       poison are the HW cells just OUTSIDE that range.  The exchange must
       therefore cross the raw OUTER edge, not the interior face:
       s==1 receives [ubnd+1, ubnd+HW], s==0 receives [lbnd-HW, lbnd-1].
       Both ranges are clipped to the intersection of the two raw ranges,
       which each rank computes from the allgathered boxes, so the range we
       receive equals the range the peer sends and vice versa.  Note that a
       rank can have MORE THAN ONE candidate supplier per face (Carpet's
       overlapping sub-grids), so every rank pair is considered rather than
       the first match; suppliers then fill the same halo cells with
       identical values. */
    const int rlo = bb[r * 6 + d];
    const int rhi = bb[r * 6 + 3 + d];
    int rx_lo, rx_hi, sx_lo, sx_hi;
    if (s)
    {
      rx_lo = ((int)L->ubnd[d] + 1 > rlo) ? (int)L->ubnd[d] + 1 : rlo;
      rx_hi = ((int)L->ubnd[d] + HW < rhi) ? (int)L->ubnd[d] + HW : rhi;
      sx_lo = (rlo - HW > (int)L->lbnd[d]) ? rlo - HW : (int)L->lbnd[d];
      sx_hi = (rlo - 1 < (int)L->ubnd[d]) ? rlo - 1 : (int)L->ubnd[d];
    }
    else
    {
      rx_lo = ((int)L->lbnd[d] - HW > rlo) ? (int)L->lbnd[d] - HW : rlo;
      rx_hi = ((int)L->lbnd[d] - 1 < rhi) ? (int)L->lbnd[d] - 1 : rhi;
      sx_lo = (rhi + 1 > (int)L->lbnd[d]) ? rhi + 1 : (int)L->lbnd[d];
      sx_hi = (rhi + HW < (int)L->ubnd[d]) ? rhi + HW : (int)L->ubnd[d];
    }
    if (rx_lo > rx_hi && sx_lo > sx_hi) continue;

    /* transverse overlap of the two raw (interior+ghost) index ranges:
       computed identically from both ranks' boxes, so the Sendrecv
       counts match regardless of how the two ranks are decomposed */
    int o0[3], o1[3];
    int ok = 1;
    for (int e = 0; e < 3; e++)
    {
      if (e == d) continue;
      const int rlo_e = bb[r * 6 + e];
      const int rhi_e = bb[r * 6 + 3 + e];
      o0[e] = (int)L->lbnd[e] > rlo_e ? (int)L->lbnd[e] : rlo_e;
      o1[e] = (int)L->ubnd[e] < rhi_e ? (int)L->ubnd[e] : rhi_e;
      if (o0[e] > o1[e]) { ok = 0; break; }
    }
    if (!ok) continue;

    /* transverse overlap in our local frame (the neighbour computes the
       same global extents in its own frame; extents and counts match) */
    int pl[3], ph[3];
    for (int e = 0; e < 3; e++)
    {
      if (e == d) continue;
      pl[e] = o0[e] - (int)L->lbnd[e];
      ph[e] = o1[e] - (int)L->lbnd[e];
    }
    const int nx = sx_hi - sx_lo + 1;   /* sent thickness  (== neighbour's received) */
    const int mx = rx_hi - rx_lo + 1;   /* received thickness (== neighbour's sent) */
    size_t cnt_s = (size_t)(nx > 0 ? nx : 0), cnt_r = (size_t)(mx > 0 ? mx : 0);
    for (int e = 0; e < 3; e++)
      if (e != d)
      {
        cnt_s *= (size_t)(ph[e] - pl[e] + 1);
        cnt_r *= (size_t)(ph[e] - pl[e] + 1);
      }
    if (cnt_s > slab_max || cnt_r > slab_max) continue;  /* defensive */

    /* the two ranks of a face pair have opposite s, and either rank may be
       the one that initiated the pairing, so the tags must name the
       *sender*: sendtag carries our own rank, recvtag the peer's */
    const int tagbase = 7000 + (level * 6 + d * 2) * 512;
    const int sendtag = tagbase + s * 512 + (me & 511);
    const int recvtag = tagbase + (1 - s) * 512 + (nb & 511);
    const int a0 = pl[(d+1)%3], a1 = ph[(d+1)%3];
    const int b0 = pl[(d+2)%3], b1 = ph[(d+2)%3];
    for (int f = 0; f < 10; f++)
    for (int tl = 0; tl < 3; tl++)
    {
      if (raw[f][tl] == NULL) continue;   /* poisoned level: nothing to
                                             exchange (the neighbour
                                             skips it identically) */
      /* pack the send slab from our raw array (local indices) */
      size_t o = 0;
      for (int a = a0; a <= a1; a++)
      for (int b = b0; b <= b1; b++)
      for (int g = 0; g < nx; g++)
      {
        const CCTK_INT ix = (CCTK_INT)(sx_lo - (int)L->lbnd[d] + g);
        int idx[3];
        idx[d] = ix; idx[(d+1)%3] = a; idx[(d+2)%3] = b;
        sbuf[o++] = raw[f][tl][((size_t)idx[2] * (size_t)L->lsh[1]
                                + (size_t)idx[1]) * (size_t)L->lsh[0]
                               + (size_t)idx[0]];
      }
      MPI_Sendrecv(sbuf, (int)cnt_s, MPI_DOUBLE, nb, sendtag,
                   rbuf, (int)cnt_r, MPI_DOUBLE, nb, recvtag,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* unpack the received slab into our halo (local indices may be
         negative; the halo covers them).  The transverse local indices are
         in our own frame (pl..ph): the buffer is ordered by the global
         overlap range, identical on both sides. */
      o = 0;
      for (int a = pl[(d+1)%3]; a <= ph[(d+1)%3]; a++)
      for (int b = pl[(d+2)%3]; b <= ph[(d+2)%3]; b++)
      for (int g = 0; g < mx; g++)
      {
        int idx[3];
        idx[d] = rx_lo - (int)L->lbnd[d] + g;
        idx[(d+1)%3] = a; idx[(d+2)%3] = b;
        L->halo[f][tl][LD_HALO_INDEX(idx[0], idx[1], idx[2])] = rbuf[o++];
      }
    }
  }
  free(sbuf); free(rbuf); free(bb);
#endif
#undef LD_HALO_INDEX
}

/* safe-region margin in cells: the local27 (poly27) branch reads a 3x3x3
   stencil (metric fit) and a 6th-order central FD for metric derivatives,
   both centred on the node `point` and needing [point-3, point+3] (the
   FD uses nodes point-3..point+3; the 3x3x3 fit needs only point-1..point+1).
   Hence the centre must be at least 3 cells from each edge of the local
   patch -> margin 3.  (The earlier value of 7 over-shrinks the safe region
   and wrongly rejects valid particles that sit within 3..6 cells of the
   patch edge -- e.g. an equatorial particle on a z=0 symmetry plane inside
   a 1p MultiPatch block.  The probe confirmed: z=0 particle rejected when
   the safe region started 7 cells from the edge.)
   The aei/carpet branches (sinterp_mode 1/2) are pointwise (a 3-node
   quadratic {tp-2,tp-1,tp}), so a margin of 2 cells is correct for them.
   The local27 branch (sinterp_mode 3) fits a 5x5x5 stencil (nodes
   tp-2..tp+2) whose nodal Christoffels take 6th-order FD derivatives
   (+/-3 around each node): the full read window is [tp-5, tp+5], hence
   margin 5. */
static int safe_margin_cells(void)
{
  if (gstate.sinterp_mode == 0) return 3;
  if (gstate.sinterp_mode == 3) return 5;
  return 2;
}

/* in-range test for one direction of one level:
   use the margin-appropriate safe region when it is non-empty; if the
   region is degenerate (level too thin in that direction), fall back to
   the interior box.  A particle on the symmetry plane there has its
   derivative in that direction vanish by symmetry, so the interior-box
   test is the correct "in-range" criterion. */
static int in_level_dir(const double y[], const LevelData *L, int d)
{
  const int margin = safe_margin_cells();
  double lo = L->origin[d] + (CCTK_REAL)margin * L->dx[d];
  double hi = L->origin[d] + (CCTK_REAL)(L->lsh[d]-1-margin) * L->dx[d];
  if (lo > hi) { lo = L->i_lbnd[d]; hi = L->i_ubnd[d]; }
  return (y[d+1] >= lo && y[d+1] <= hi);
}

/* finest level (of the given map) whose SAFE region contains y; -1 if none */
static int pick_level(const double y[], int map)
{
  for (int l = gstate.samr_levels-1; l >= 0; l--)
  {
    if (!gstate.LD[map][l].valid) continue;
    const LevelData *L = &gstate.LD[map][l];
    if (in_level_dir(y, L, 0) && in_level_dir(y, L, 1) &&
        in_level_dir(y, L, 2))
      return l;
  }
  return -1;
}

/* ============================================================================================
   Phase B: multipatch (Thornburg04) support.

   A particle's y[1..3] is expressed in the LOCAL coordinates of its current
   block (gstate.pgeodesic[m].cur_map).  For the center Cartesian block
   (map 0) these coincide with global physical coordinates; for the six outer
   blocks they are that block's own (curvilinear) coordinates, related to the
   global physical coordinates by the block Jacobian.

   The geodesic ODE is covariant: inside a single block the particle is
   integrated with that block's own metric + Christoffels (no transformation
   of the Christoffels is needed -- only the metric/Christoffel are re-sampled
   on the new block's grid).  Two things change when a particle crosses a block
   boundary:
     1. its position must be re-expressed in the new block's local coords;
     2. its spatial 4-velocity components transform between the two blocks'
        local bases as a vector:  u_new = J_new * J_old^{-1} * u_old,
        where J_block = d a_block / d x_global (the block Jacobian).
        The time component u^t is unchanged (the block time coord == global t).

   locate_and_cross():  on call, y[1..3] are in the current block's local coords.
     Returns  0  -> particle still inside the current block's coarse safe
                    region: nothing changed (fast path; the short single-block
                    run never leaves this branch, hence bit-identical).
     Returns  1  -> particle crossed into a neighbouring block; y[1..7] and
                    gstate.pgeodesic[m].cur_map are updated IN PLACE to the
                    new block's local coords + transformed velocity.  The
                    caller must then force a re-fit (the position/level change
                    does this via the existing cf2_fitting trigger).
     Returns -1  -> particle outside the whole multipatch domain, or in the
                    inter-block shadow zone (no block's coarse region covers
                    it): caller marks outbd.
   ============================================================================================ */
/* 3x3 inverse via the adjugate.  Returns 1 on success, 0 if (near) singular.
   r is not touched on failure. */
static int inv3x3(const double a[3][3], double r[3][3])
{
  double det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
             - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
             + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if (!isfinite(det) || fabs(det) < 1.0e-30) return 0;
  const double idet = 1.0/det;
  r[0][0] = (a[1][1]*a[2][2]-a[1][2]*a[2][1])*idet;
  r[0][1] = (a[1][2]*a[2][0]-a[0][2]*a[2][1])*idet;
  r[0][2] = (a[0][2]*a[1][1]-a[0][1]*a[1][2])*idet;
  r[1][0] = (a[1][2]*a[2][0]-a[1][0]*a[2][2])*idet;
  r[1][1] = (a[0][0]*a[2][2]-a[0][2]*a[2][0])*idet;
  r[1][2] = (a[0][2]*a[1][0]-a[0][0]*a[1][2])*idet;
  r[2][0] = (a[1][0]*a[2][1]-a[1][1]*a[2][0])*idet;
  r[2][1] = (a[0][1]*a[2][0]-a[0][0]*a[2][1])*idet;
  r[2][2] = (a[0][0]*a[1][1]-a[0][1]*a[1][0])*idet;
  return 1;
}

static int locate_and_cross(int m, double y[], cGH *cctkGH)
{
  const int ndims = 3;
  const int npoints = 1;
  const int pm = gstate.pgeodesic[m].cur_map;

  /* fast path: still inside the current block's full interior region?
     Use the full interior box (not just the 7-cell safe margin) as the
     "stay" criterion.  This is a strict superset of the safe-region check:
     a particle deep inside its block — including in the margin band between
     the safe region and the grid edge — stays put.  cf2_fitting() clamps the
     27-point stencil to the valid cell range, so the particle is still
     integrated correctly from the current block.  This prevents the spurious
     hand-off to a neighbouring map that was the root cause of the
     multi-patch drop bug (lower-half particles being routed to the upper
     block where they do not exist).  Only when the particle is genuinely
     outside the current block's interior does it proceed to locate_and_cross
     block-transfer logic. */
  const LevelData *L0 = &gstate.LD[pm][0];
  if (y[1] >= L0->i_lbnd[0] && y[1] <= L0->i_ubnd[0] &&
      y[2] >= L0->i_lbnd[1] && y[2] <= L0->i_ubnd[1] &&
      y[3] >= L0->i_lbnd[2] && y[3] <= L0->i_ubnd[2])
    return 0;
  if (in_level_dir(y, L0, 0) && in_level_dir(y, L0, 1) && in_level_dir(y, L0, 2))
    return 0;

  /* 1) current block local coords -> global physical coords
        (also yields J_old = d a_old / d x_global, row-major: J[row][col] = dadx_old[row*3+col]).
        llama's Coordinates_LocalToGlobal / _GlobalToLocal interpret the coord & Jacobian
        args as *pointer arrays* (d-major): coords is CCTK_REAL*[ndims] (each pointer ->
        npoints doubles), dadx is CCTK_REAL*[ndims*ndims] (each -> npoints).  Geodesic's
        flat arrays are wired to those pointer arrays below (npoints==1 here). */
  double localv[ndims], xg[ndims], dadx_old[ndims*ndims];
  for (int d = 0; d < ndims; d++) localv[d] = y[d+1];
  CCTK_REAL* localp[ndims];
  for (int d = 0; d < ndims; d++) localp[d] = &localv[d];
  CCTK_REAL* xgp[ndims];
  for (int d = 0; d < ndims; d++) xgp[d] = &xg[d];
  CCTK_REAL* dadxop[ndims*ndims];
  for (int k = 0; k < ndims*ndims; k++) dadxop[k] = &dadx_old[k];
  {
    CCTK_INT patch = pm;
    int rc = MultiPatch_LocalToGlobal(cctkGH, ndims, npoints,
                                      &patch, localp, xgp,
                                      NULL, NULL, dadxop, NULL, NULL, NULL);
    if (rc != 0) return -1;
  }

  /* 2) global -> (block, block-local coords, J_new = d a_new / d x_global
         col-major: J[row][col] = dadx_new[col*ndims+row]) */
  double newy[ndims];
  double dadx_new[ndims*ndims];
  CCTK_REAL* newyp[ndims];
  for (int d = 0; d < ndims; d++) newyp[d] = &newy[d];
  CCTK_REAL* dadxnp[ndims*ndims];
  for (int k = 0; k < ndims*ndims; k++) dadxnp[k] = &dadx_new[k];
  int newmap;
  {
    CCTK_INT patch[1];
    int rc = MultiPatch_GlobalToLocal(cctkGH, ndims, npoints,
                                      xgp, &patch[0], newyp,
                                      dadxnp, NULL);
    if (rc != 0 || patch[0] < 0 || patch[0] >= GEODESIC_MAX_MAPS)
      return -1;   /* unknown point: outside the whole multipatch domain */
    newmap = patch[0];
  }

  /* 3) still the same block but outside its safe region -> shadow-zone point
        (stencil fits in neither block's grid) -> out of domain. */
  if (newmap == pm)
    return -1;

  /* 4) usable by the neighbour?  require the point to lie in the safe
        region (or, if that region is degenerate, the interior box) of SOME
        level of the new block -- exactly the criterion pick_level uses to
        decide whether the block can integrate the point there.  This is
        tighter than the old "coarsest-level interior box" test and stops
        a particle from being handed to a block that has no usable
        stencil at that spot (inter-block shadow zone). */
  {
    double nyp[4];
    nyp[0] = 0.0;
    for (int d = 0; d < ndims; d++) nyp[d+1] = newy[d];
    if (pick_level(nyp, newmap) < 0)
      return -1;   /* no usable level in the new block -> shadow zone / out of domain */
  }

  /* 5) build the two Jacobians. */
  double Jold[3][3], Jnew[3][3], Jinv[3][3];
  for (int row = 0; row < ndims; row++)
    for (int col = 0; col < ndims; col++)
    {
      Jold[row][col] = dadx_old[row*ndims + col];          /* row-major */
      Jnew[row][col] = dadx_new[col*ndims + row];          /* col-major  */
    }

  /* 6) transform the spatial velocity:  u_new = Jnew * inv(Jold) * u_old.
         v = inv(Jold) * u_old  (global physical velocity components).       */
  if (inv3x3(Jold, Jinv) != 1)
    return -1;
  double v[3];
  for (int col = 0; col < ndims; col++) {
    v[col] = 0.0;
    for (int j = 0; j < ndims; j++)
      v[col] += Jinv[col][j] * y[j+5];   /* spatial u = y[5..7] = (u^x,u^y,u^z) */
  }
  for (int row = 0; row < ndims; row++) {
    double u = 0.0;
    for (int col = 0; col < ndims; col++)
      u += Jnew[row][col] * v[col];
    y[row+5] = u;   /* write transformed spatial u into y[5..7] */
  }

  /* 7) adopt the new block.  Force a re-fit in the new block's grid by
         setting the level to a sentinel; the existing cf2_fitting trigger
         (lvl != level) will pick the new block's level and re-sample. */
  for (int d = 0; d < ndims; d++) y[d+1] = newy[d];
  gstate.pgeodesic[m].cur_map = newmap;
  gstate.pgeodesic[m].level    = -1;
  gstate.pgeodesic[m].flag     = 0;
  return 1;
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
   further below */
static void fit_metric_poly27(int m, const gd_data src[3][3][3],
                              CCTK_REAL dst[4][4][POLY27],
                              CCTK_INT dstc[4][4]);
static void metric_at_cell_tl(int tl, int ix, int iy, int iz, gd_data *out);
static void pointwise_cf2_tl(int m, CCTK_INT tl, CCTK_INT i1, CCTK_INT i2, CCTK_INT i3,
                             double cf2out[4][4][4]);
/* local27 (sinterp_mode 3) helpers, defined further below */
static void p125_eval_deriv(const CCTK_REAL c[POLY125],
                            double tx, double ty, double tz,
                            double *val, double *dtx, double *dty, double *dtz);
static void fit_metric_poly125(const gd_data src[125],
                               CCTK_REAL dst[4][4][POLY125],
                               CCTK_INT dstc[4][4]);
static void fit_cf2_poly125(const double G[125][4][4][4],
                            CCTK_REAL dst[4][4][4][POLY125],
                            CCTK_INT dstc[4][4][4]);
static void local27_time_weights(int m, double y0, double w[3], int *nlev);
static int local27_christoffel_analytic(int m, const double y[8], double cf2[4][4][4]);
/* 1D 3-node (order-2) time-Lagrange weights, defined further below */
static inline double fourd_tw(int lvl, double s);
static inline double fourd_tw_a(int lvl, double s);

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
   local27 (sinterp_mode 3): 125-term quartic tensor-product polynomial
   (5 nodes per axis, offsets {-2..+2} from 'below'; term index
   k = a*25 + b*5 + c is tx^a ty^b tz^c, coefficients from the V5INV
   separable fit).  Nested evaluation gives the value and all three
   spatial derivatives w.r.t. the NORMALIZED coords (tx,ty,tz) in one
   pass (~200 FLOPs; physical derivatives require division by dx at the
   caller).
   ============================================================================================ */
static void p125_eval_deriv(const CCTK_REAL c[POLY125],
                            double tx, double ty, double tz,
                            double *val, double *dtx, double *dty, double *dtz)
{
  double px[5], py[5], pz[5];
  int a, b;
  px[0] = 1.0; py[0] = 1.0; pz[0] = 1.0;
  for (int i = 1; i < 5; i++) { px[i] = px[i-1]*tx; py[i] = py[i-1]*ty; pz[i] = pz[i-1]*tz; }
  double v = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;
  for (a = 0; a < 5; a++)
  {
    /* P_a   = sum_b ty^b      Q_ab(tz)
       dP_a  = sum_b b ty^{b-1} Q_ab(tz)          (d/dty)
       dR_a  = sum_b ty^b      Q'_ab(tz)          (d/dtz) */
    double Pa = 0.0, dPa = 0.0, dRa = 0.0;
    for (b = 0; b < 5; b++)
    {
      const CCTK_REAL *cp = c + a*25 + b*5;
      double Q  = cp[0] + cp[1]*pz[1] + cp[2]*pz[2] + cp[3]*pz[3] + cp[4]*pz[4];
      double dQ = cp[1] + 2.0*cp[2]*pz[1] + 3.0*cp[3]*pz[2] + 4.0*cp[4]*pz[3];
      Pa  += py[b]*Q;
      dRa += py[b]*dQ;
      if (b > 0) dPa += (double)b * py[b-1] * Q;
    }
    v  += px[a]*Pa;
    vy += px[a]*dPa;
    vz += px[a]*dRa;
    if (a > 0) vx += (double)a * px[a-1] * Pa;
  }
  if (val) *val = v;
  if (dtx) *dtx = vx;
  if (dty) *dty = vy;
  if (dtz) *dtz = vz;
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

  /* absolute grid indices (1-based): the stencil reads the three nodes
     (point-1, point, point+1) (0-based), centred on the particle's
     nearest node 'point' (= the reference node of 'below' =
     origin+point*dx).  With tx = (y-below)/dx the three nodes sit at
     tx = -1, 0, +1 -- exactly the {-1,0,1} node set of MtxA (middle
     column 13 = reference node) -- and the particle has
     tx in [-0.5, +0.5): always inside the stencil, pure interpolation
     (no extrapolation). */
  i1 = i1 + gstate.pgeodesic[m].point[1];
  i2 = i2 + gstate.pgeodesic[m].point[2];
  i3 = i3 + gstate.pgeodesic[m].point[3];

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
   tl (0 = t, 1 = t - dt), for the local27 (sinterp_mode 3) per-level
   Christoffel sampling (both levels of each stencil are precomputed).
   Operation-identical to the cf2_grid core (3+1 inverse from the level
   fields, 6th-order FD spatial metric derivatives at the same level)
   except:
     - the metric time level is tl (cf2_grid hard-codes level 0),
     - the metric time derivative is taken AT THAT SAME LEVEL: the
       3-node time-Lagrange derivative of the stored levels, i.e.
       (3g0-4g1+g2)/2dt at tl=0, (g0-g2)/2dt at tl=1 and
       (-g0+4g1-3g2)/2dt at tl=2 -- the same weights fourd_tw_a() uses
       for the metric blend, so the Christoffel and the metric see one
       time stencil.  It degrades to (g0-g1)/dt when only two levels are
       usable (local27_lvl2_bad) and to 0 with a single usable level
       (old_level_bad) or a stationary metric, matching
       local27_time_weights and local27_christoffel_analytic,
     - the result goes to the caller's array (cf2_grid caches in cf2g).
   A (near-)singular spatial 3-metric poisons cf2out with NaN so the
   caller's isfinite() gate discards the stencil fit instead of fitting
   garbage.
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
  /* usable stored time levels for this particle: same degraded-blend rule
     as local27_time_weights (the flags come from the previous refit's
     level scans, which is conservative -- a newly filled level only costs
     one refit at the lower-order stencil) */
  int nlev = 3;
  const double dt = gview.dt;

  if (gstate.pgeodesic[m].old_level_bad)       nlev = 1;
  else if (gstate.pgeodesic[m].local27_lvl2_bad) nlev = 2;

  index = LD_INDEX(i1-1, i2-1, i3-1);

  gxxL = gview.gxx[tl][index];  gxyL = gview.gxy[tl][index];  gxzL = gview.gxz[tl][index];
  gyyL = gview.gyy[tl][index];  gyzL = gview.gyz[tl][index];  gzzL = gview.gzz[tl][index];

  det = -gxzL*gxzL*gyyL + 2*gxyL*gxzL*gyzL - gxxL*gyzL*gyzL - gxyL*gxyL*gzzL + gxxL*gyyL*gzzL;
  if (!isfinite(det) || fabs(det) < 1.0e-30)
  {
    for (j1=0; j1<4; j1++)
      for (j2=0; j2<4; j2++)
        for (j3=0; j3<4; j3++)
          cf2out[j1][j2][j3] = NAN;
    return;
  }
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
        {
          /* d_t g_{j2 j3} at the time of level tl (see the header comment) */
          if (gview.metric_static || nlev == 1 || !(dt > 0.0))
            dgd[j1][j2][j3] = 0.0;
          else if (nlev == 2)
            dgd[j1][j2][j3] = (g4dn(0, j2, j3, i1, i2, i3)
                               - g4dn(1, j2, j3, i1, i2, i3)) / dt;
          else
          {
            const double g0 = g4dn(0, j2, j3, i1, i2, i3);
            const double g1 = g4dn(1, j2, j3, i1, i2, i3);
            const double g2 = g4dn(2, j2, j3, i1, i2, i3);
            if (tl == 0)
              dgd[j1][j2][j3] = (3.0*g0 - 4.0*g1 + g2) * (0.5 / dt);
            else if (tl == 1)
              dgd[j1][j2][j3] = (g0 - g2) * (0.5 / dt);
            else
              dgd[j1][j2][j3] = (-g0 + 4.0*g1 - 3.0*g2) * (0.5 / dt);
          }
        }
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

/* A 4-metric is "usable" for the geodesic integration iff it is finite and
   has the Lorentzian signature we need: a positive-definite spatial block
   (Sylvester's criterion on the 3x3 block g_ij) and a negative 4-metric
   determinant, which together force exactly one negative eigenvalue.  This
   rejects the degenerate all-zero fit (g_ij = 0, det = 0) that poisons the
   l27/local27 cheap paths, as well as any non-finite fit.

   The signature test must not look at g_00: that component changes sign
   wherever the time basis vector becomes spatial (the Kerr-Schild g_00 = 0
   surface at r_KS = 2M), which is a property of the coordinates, not of the
   fit. */

static double det3x3(const double a[3][3])
{
  return a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
       - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
       + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
}

static int metric_valid(const double gdn[4][4])
{
  double s[3][3], m[3][3];
  double det_g = 0.0;
  int mu, nu, k, j;

  for (mu=0;mu<4;mu++)
    for (nu=0;nu<4;nu++)
      if (!isfinite(gdn[mu][nu])) return 0;

  for (mu=0;mu<3;mu++)
    for (nu=0;nu<3;nu++)
      s[mu][nu] = gdn[mu+1][nu+1];

  /* Sylvester: leading principal minors of the spatial block all positive */
  if (s[0][0] <= 0.0) return 0;
  if (s[0][0]*s[1][1] - s[0][1]*s[1][0] <= 0.0) return 0;
  if (det3x3(s) <= 0.0) return 0;

  /* det(g_mn) expanded along the first row must be negative */
  for (j=0;j<4;j++)
  {
    for (mu=0;mu<3;mu++)
    {
      k = 0;
      for (nu=0;nu<4;nu++)
        if (nu != j) m[mu][k++] = gdn[mu+1][nu];
    }
    det_g += ((j & 1) ? -gdn[0][j] : gdn[0][j]) * det3x3(m);
  }
  if (det_g >= 0.0) return 0;

  return 1;
}

int cf2_fitting (int m, const double y[])
{
  CCTK_INT i, j1, j2, j3, j4, j5, j6, j7, j8;
  const int pm = gstate.pgeodesic[m].cur_map;   /* owning block */

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
                     : (gstate.multi_exact ? 0 : pick_level(y, pm));
  if (lvl < 0)
  {
    /* The margin-3 (6th-order FD) safe region rejects the particle.  No
       mode has an extrapolant fallback in this version (l27 is fully
       self-contained by design), so the particle is dropped. */
    gstate.pgeodesic[m].outbd = 1;
    return 0;
  }
  /* make sure this thread's grid view matches the particle's (map, level)
     (self-guarded; also covers the no-rebuild path when this thread
     previously worked on a different particle/map/level) */
  switch_to_level(pm, lvl);

  /* compute base grid point index for the current particle position (on level lvl) */
  CCTK_INT tpoint[4];
  tpoint[0] = 0;
  for (i=1; i<4; i++)
  {
    tpoint[i] = lrint (floor ((y[i] - gstate.LD[pm][lvl].origin[i-1]) * gstate.LD[pm][lvl].inv_dx[i-1] + 0.5));
  }

  /* multi_amr gap band (no-halo path only): a particle that lies in the
     gap between rank sub-patches maps its raw stencil center tpoint
     outside [1, lsh-1].  The clip below would then centre the 27-point
     stencil 4+ cells away, giving a huge extrapolation and a garbage fit
     (superluminal velocity -> drop).  A particle in the *face cell*
     (tpoint within [1, lsh-1]) is fine (the centred stencil still
     contains the particle), so only the genuine out-of-box case is
     dropped (l27_bad gate; l27 has no extrapolant in this version).  In
     single-proc runs the owning box is the full grid, so a particle is
     never out-of-box.  With the multi-process halo active the stencil
     window is checked centred against the halo extent below (the centre
     must lie in the patch interior), which subsumes this position-based
     shadow-zone test, so it is skipped there. */
  if (gstate.multi_amr && gstate.sinterp_mode == 0 &&
      gstate.LD[pm][lvl].halo[0][0] == NULL &&
      lvl >= 0 && lvl < gstate.samr_levels)
  {
    const LevelData *L = &gstate.LD[pm][lvl];
    /* Shadow-zone check: the multi-amr claim box is the interior box
       (i_lbnd/i_ubnd), which sits ~4-7 cells inside the grid edge.  A
       particle claimed to this rank but lying OUTSIDE the interior box is
       in the gap/shadow zone (between the interior edge and the grid edge).
       The full l27 stencil (margin-3 + 6th-order FD, 4 cells each way) is
       not guaranteed to fit there, so the particle is dropped (l27_bad
       gate; l27 has no extrapolant in this version).  In single-proc the
       owning box is the full grid, so a particle is never out-of-box. */
    double yx = y[1], yy = y[2], yz = y[3];
    if (yx <  L->i_lbnd[0] || yy <  L->i_lbnd[1] || yz <  L->i_lbnd[2] ||
        yx >= L->i_ubnd[0] || yy >= L->i_ubnd[1] || yz >= L->i_ubnd[2])
    {
      gstate.pgeodesic[m].l27_bad = 1;   /* gate to the drop policy */
      return 0;
    }
  }

  /* Stencil-fit test with the multi-process halo active: the only hard
     requirement is that the full window [tp-W, tp+W] fits inside this
     patch's halo extent, i.e. every node the stencil reads actually holds
     valid data here.  W = 5 for local27 (5x5x5 nodes + 6th-order FD),
     4 for l27.  The stencil centre tp is the node nearest the particle;
     the particle itself is inside the owner's interior box, but the
     nearest node may sit in a ghost zone (e.g. a particle just below a
     ghost-zone node).  That is legitimate and decomposition-independent:
     ghost cells are filled by Carpet's inter-processor boundaries and are
     bit-identical to the neighbour's interior cells, so the same global
     node and the same values are used in a 1-, 2- and 4-rank run.
     Requiring an interior centre instead would reject such particles and
     make the result depend on where the rank face happens to fall.
     (No silent shift: the window test is a hard validity failure, since
     an off-centre stencil is what broke the equatorial symmetry across
     the z=0 rank face.) */
  if (gstate.LD[pm][lvl].halo[0][0] != NULL && !gstate.sexact &&
      (gstate.sinterp_mode == 0 || gstate.sinterp_mode == 3))
  {
    const LevelData *L = &gstate.LD[pm][lvl];
    const int W = (gstate.sinterp_mode == 3) ? 5 : 4;
    for (i=1; i<4; i++)
    {
      if (tpoint[i] - W < L->halo_lo[i-1] || tpoint[i] + W > L->halo_hi[i-1])
      {
        if (gstate.sinterp_mode == 3) gstate.pgeodesic[m].local27_bad = 1;
        else                          gstate.pgeodesic[m].l27_bad = 1;
        return 0;
      }
    }
  }

  /* l27 (sinterp_mode 0, no halo): clip the index into the stencil-safe range.
     Every grid read here goes through g4dn() or a direct LD_INDEX, which
     accept only valid cells (1-based ix in [1, lsh]).  The stencil is the
     three nodes (tpoint-1, tpoint, tpoint+1) (0-based), centred on the
     nearest node tpoint (= reference node of 'below'), and cf2_grid
     additionally takes 6th-order FD metric derivatives at each stencil
     node (+/-3 nodes around it), so the full read window around the
     centred stencil is [tpoint-4, tpoint+4] (0-based) -- the safe range
     for tpoint is [4, lsh-5], one cell inside the box's first boundary
     face.  A particle sitting in a face cell (e.g. a z=0 particle on the
     z+ rank, the shared boundary of the 2x2 decomposition) would
     otherwise map to tp=3 and read ix=0 out of bounds.  Clipping tp to
     [4, lsh-5] keeps every read (including the direct LD_INDEX reads in
     cf2_grid) inside valid cells.  For interior particles the clip never
     fires and the particle sits at tx in [-0.5, +0.5] relative to
     'below', inside the centred stencil (pure interpolation);
     near-boundary particles that the clip pushes out of the stencil are
     dropped instead (l27_bad gate).  NOTE: single-process only (the halo
     path above takes precedence); the clip is kept bit-identical for
     single-process runs so existing references stay valid. */
  if (gstate.sinterp_mode == 0 && !gstate.sexact &&
      gstate.LD[pm][lvl].halo[0][0] == NULL)
  {
    for (i=1; i<4; i++)
    {
      if (tpoint[i] < 4)
        tpoint[i] = 4;
      if (tpoint[i] > gstate.LD[pm][lvl].lsh[i-1] - 5)
        tpoint[i] = gstate.LD[pm][lvl].lsh[i-1] - 5;
    }
  }
  /* local27 (sinterp_mode 3, no halo): the 5x5x5 stencil reads nodes tp-2..tp+2 and the
     nodal Christoffels take 6th-order FD derivatives (+/-3 around each
     node), so the full read window is [tp-5, tp+5] (0-based); the safe
     range for tpoint is [5, lsh-6].  Unlike the mode-0 clip above this is
     a hard validity failure, not a shift: a clipped centre is no longer
     centred on the particle, so the quartic fit loses its symmetry and the
     nodal Christoffels acquire a spurious odd part in the clipped
     direction.  In a multi-rank run, where a rank face can cut through the
     particle region, that pushed equatorial particles off the symmetry
     plane (z and u_z grew from 1e-9 to 4e-6 over t=20 while the
     single-rank run kept them at roundoff).  The particle is rejected
     through local27_bad and the loss policy instead.  With the halo
     active the centred window always exists and the halo test above
     applies instead. */
  if (gstate.sinterp_mode == 3 && !gstate.sexact &&
      gstate.LD[pm][lvl].halo[0][0] == NULL)
  {
    for (i=1; i<4; i++)
    {
      if (tpoint[i] < 5 ||
          tpoint[i] > gstate.LD[pm][lvl].lsh[i-1] - 6)
      {
        gstate.pgeodesic[m].local27_bad = 1;   /* gate to the drop policy */
        return 0;
      }
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
      /* below = position of node tpoint, the particle's NEAREST node
         (tpoint = lrint(u+0.5)): the stencil nodes {tpoint-1, tpoint,
         tpoint+1} sit at tx = (y-below)/dx in {-1,0,+1} (the MtxA node
         set, middle node = below) and the particle in [-0.5,+0.5) --
         always strictly inside the stencil, so poly27_eval/cf2_func are
         pure interpolations (no extrapolation). */
      gstate.pgeodesic[m].below[i] = gstate.LD[pm][lvl].origin[i-1] + tpoint[i] * gstate.LD[pm][lvl].dx[i-1];
      gstate.pgeodesic[m].dxp[i-1] = gstate.LD[pm][lvl].dx[i-1];
    }

    /* !sexact: sample metric + Christoffel symbols on the stencil.
       In the sexact branch both are exact Kerr-Schild analytic values,
       evaluated directly by tg4dn/gdn_derivatives inside func_ode, so
       no sampling is needed. */
    if (!gstate.sexact && gstate.sinterp_mode == 0)
    {
      /* l27 (sinterp_mode 0): the 27-point stencil of the CURRENT time
         level only: cf2_grid fills the nodal 4-metric (pgeodesic[m].gd)
         and the nodal Christoffel symbols (cf2g) from the 6th-order FD
         metric derivatives + nodal inverse.  The historical level-0-only
         sampling: no past-time-level read in this mode. */
      for (j1=0; j1<3; j1++)
        for (j2=0; j2<3; j2++)
          for (j3=0; j3<3; j3++)
            cf2_grid(m, j1, j2, j3);
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
  /* l27 fits the Christoffel symbols and the metric (current level only);
     local27 and aei/carpet fit nothing here (the 125-node fits
     below / pointwise interpolation) */
  if (gstate.sinterp_mode == 0)
  {
  /* NaN-poison gate: cf2_grid's 6th-order FD window reads the cells
     point-3..point+3 around each stencil node, so a single non-finite
     grid cell (e.g. the Kerr-Schild r = 0 coordinate singularity) can
     poison the nodal Christoffel samples even when all 27 stencil
     nodes themselves are finite.  If any sample is non-finite the pcf2
     fit below is useless; l27_bad is forced after the metric staging
     (which would otherwise clear it) and the particle is then dropped
     by the loss policy (no fallback interpolant in this mode).
     Re-evaluated at every refit (non-sticky), consistent with the
     metric staging. */
  int cf2_fin = 1, s5, s6, s7, sa, sb, sc;
  for (s5=0; s5<3 && cf2_fin; s5++)
  for (s6=0; s6<3 && cf2_fin; s6++)
  for (s7=0; s7<3 && cf2_fin; s7++)
    for (sa=0; sa<4 && cf2_fin; sa++)
      for (sb=0; sb<4 && cf2_fin; sb++)
        for (sc=0; sc<4 && cf2_fin; sc++)
          if (!isfinite((double)gstate.pgeodesic[m].cf2g[s7][s6][s5].cf2[sa][sb][sc]))
            cf2_fin = 0;
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
     ---- staging ("preparation zone") ----
     The current-level 27-point fit is first written into a TEMP buffer.
     Only after the staged metric has been validated at the particle position
     (metric_valid) is it copied into the official pgd/pgdc slot.  If it is
     invalid (degenerate all-zero / non-finite 4-metric), the last good
     coefficients are kept and l27_bad is set, so the metric/Christoffel
     path drops the particle for this refit (loss policy).
     --------------------------------------------- */
  {
    CCTK_REAL tmp_pgd[4][4][POLY27];
    CCTK_INT  tmp_pgd_c[4][4];
    fit_metric_poly27(m, gstate.pgeodesic[m].gd, tmp_pgd, tmp_pgd_c);
    double gtmp[4][4];
    int mu,nu,good;
    for (mu=0;mu<4;mu++)
      for (nu=0;nu<4;nu++)
        gtmp[mu][nu]=0.0;
    for (mu=0;mu<4;mu++)
      for (nu=mu;nu<4;nu++)
        gtmp[mu][nu] = poly27_eval(y,m,tmp_pgd[mu][nu],tmp_pgd_c[mu][nu]);
    for (mu=1;mu<4;mu++)
      for (nu=0;nu<mu;nu++)
        gtmp[mu][nu] = gtmp[nu][mu];
    good = metric_valid(gtmp);
    if (good)
    {
      for (mu=0;mu<4;mu++)
        for (nu=0;nu<4;nu++)
        {
          for (j4=0;j4<POLY27;j4++)
            gstate.pgeodesic[m].pgd[mu][nu][j4] = tmp_pgd[mu][nu][j4];
          gstate.pgeodesic[m].pgdc[mu][nu] = tmp_pgd_c[mu][nu];
        }
      gstate.pgeodesic[m].l27_bad = 0;
    }
    else
    {
      /* keep the last good coefficients; drop the particle this refit */
      gstate.pgeodesic[m].l27_bad = 1;
    }
  }
  if (!cf2_fin)
  {
    /* Christoffel stencil samples were non-finite (FD-window poison):
       the pcf2 fit above is invalid.  The metric staging just run may
       have cleared the flag, so re-set it here; the cheap Christoffel
       path (func_ode) and the metric path (metric_at_point) then drop
       the particle (loss policy) for this refit. */
    gstate.pgeodesic[m].l27_bad = 1;
  }

  } /* end l27: Christoffel/metric fitting */
  else if (gstate.sinterp_mode == 3)
  {
    /* local27: 125-node (5x5x5) quartic tensor-product fits on the three
       stored time levels.  Per level: sample the nodal 4-metrics
       (metric_at_cell_tl) and the nodal Christoffel symbols
       (pointwise_cf2_tl: 6th-order FD metric derivatives + nodal
       inverse, the l27 convention), fit the 125-term polynomial (the
       separable V5INV transforms = the 125x125 inverse Vandermonde), and
       probe the level-1 / level-2 metric-sample validity.  The particle
       Christoffel is the nlev-level time-Lagrange blend of the level
       fits (fast path); when the nodal Christoffels are NaN-poisoned
       (some 6th-order FD window hits the central singularity / excised
       region) the Christoffel is assembled analytically from the metric
       fits (local27_christoffel_analytic).  Fully self-contained: no AEI
       thorn, no CarpetInterp. */
    double cf2n[125][4][4][4];   /* nodal Christoffel, one level at a time */
    gd_data g125[125];           /* nodal 4-metric, one level at a time */
    CCTK_REAL tmp_g[3][4][4][POLY125];
    CCTK_INT tmp_gc[3][4][4];
    int cf2_fin = 1, mu, nu, dd;
    for (CCTK_INT tl=0; tl<3; tl++)
    {
      /* 125 stencil nodes (offsets -2..+2 from tpoint), level tl */
      for (CCTK_INT ix=0; ix<5; ix++)
      for (CCTK_INT iy=0; iy<5; iy++)
      for (CCTK_INT iz=0; iz<5; iz++)
      {
        int i1 = (int)(ix-2) + gstate.pgeodesic[m].point[1];
        int i2 = (int)(iy-2) + gstate.pgeodesic[m].point[2];
        int i3 = (int)(iz-2) + gstate.pgeodesic[m].point[3];
        /* i1..i3 are 0-based node indices (point[] is 0-based):
           metric_at_cell_tl takes 0-based directly; pointwise_cf2_tl
           subtracts 1 internally (LD_INDEX(i-1)), so it takes 1-based. */
        metric_at_cell_tl((int)tl, i1, i2, i3, &g125[ix*25+iy*5+iz]);
        if (cf2_fin)
          pointwise_cf2_tl(m, tl, i1+1, i2+1, i3+1, cf2n[ix*25+iy*5+iz]);
      }
      fit_metric_poly125(g125, tmp_g[tl], tmp_gc[tl]);
      /* NaN-poison gate: the 6th-order FD window of some stencil node can
         reach non-finite cells even when the 125 nodes themselves are
         finite; any non-finite nodal Christoffel kills the level fit.
         A stored past time level that is not yet filled at this cell has
         the same effect, and that is what fires in practice: at the first
         refit of a run whose time levels are not yet filled, about half of
         a 100-particle ring trips here.  The resulting local27_cf2_bad is
         sticky (see Geodesic.h), so those particles keep the analytic
         Christoffel assembly for the rest of the run.
         (Scanned only while cf2_fin, so a later level's un-filled cf2n
         is never read.) */
      if (cf2_fin)
      {
        for (int k=0; k<125 && cf2_fin; k++)
        for (mu=0; mu<4 && cf2_fin; mu++)
        for (nu=0; nu<4 && cf2_fin; nu++)
        for (dd=0; dd<4 && cf2_fin; dd++)
          if (!isfinite((double)cf2n[k][mu][nu][dd]))
          { cf2_fin = 0; }
        if (cf2_fin)
          fit_cf2_poly125(cf2n, gstate.pgeodesic[m].pcf2_125[tl],
                          gstate.pgeodesic[m].pcf2_125c[tl]);
      }
      /* older-level validity from this level's metric samples
         (all-zero or non-finite -> the past level is not filled) */
      if (tl == 1)
      {
        int old_bad = 0; double maxabs = 0.0;
        for (int k=0; k<125 && !old_bad; k++)
        for (mu=0; mu<4 && !old_bad; mu++)
        for (nu=0; nu<4 && !old_bad; nu++)
        {
          double v = g125[k].gd[mu][nu];
          if (!isfinite(v))
            old_bad = 1;
          else if (fabs(v) > maxabs)
            maxabs = fabs(v);
        }
        gstate.pgeodesic[m].old_level_bad =
          (!old_bad && maxabs < 1e-100) ? 1 : old_bad;
      }
      if (tl == 2)
      {
        int lvl2_bad = 0; double maxabs = 0.0;
        for (int k=0; k<125 && !lvl2_bad; k++)
        for (mu=0; mu<4 && !lvl2_bad; mu++)
        for (nu=0; nu<4 && !lvl2_bad; nu++)
        {
          double v = g125[k].gd[mu][nu];
          if (!isfinite(v))
            lvl2_bad = 1;
          else if (fabs(v) > maxabs)
            maxabs = fabs(v);
        }
        gstate.pgeodesic[m].local27_lvl2_bad =
          (!lvl2_bad && maxabs < 1e-100) ? 1 : lvl2_bad;
      }
    }
    /* stage the blended metric at the particle position, validate, commit
       (the last good fits are kept if it is invalid; the particle is then
       dropped via local27_bad -- no AEI fallback in this mode) */
    {
      double tx = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
      double ty = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
      double tz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
      double w[3]; int nlev;
      local27_time_weights(m, y[0], w, &nlev);
      double gtmp[4][4];
      for (mu=0;mu<4;mu++) for (nu=0;nu<4;nu++) gtmp[mu][nu]=0.0;
      for (mu=0;mu<4;mu++)
      for (nu=mu;nu<4;nu++)
      {
        double g = 0.0;
        for (int l=0; l<nlev; l++)
        {
          double cv;
          p125_eval_deriv(tmp_g[l][mu][nu], tx,ty,tz, &cv,NULL,NULL,NULL);
          g += w[l]*cv;
        }
        gtmp[mu][nu] = g;
      }
      for (mu=1;mu<4;mu++)
        for (nu=0;nu<mu;nu++)
          gtmp[mu][nu] = gtmp[nu][mu];
      if (metric_valid(gtmp))
      {
        for (int l=0; l<3; l++)
        for (mu=0;mu<4;mu++)
        for (nu=0;nu<4;nu++)
        {
          for (int j4=0;j4<POLY125;j4++)
            gstate.pgeodesic[m].pgd125[l][mu][nu][j4] = tmp_g[l][mu][nu][j4];
          gstate.pgeodesic[m].pgd125c[l][mu][nu] = tmp_gc[l][mu][nu];
        }
        gstate.pgeodesic[m].local27_bad = 0;
      }
      else
      {
        gstate.pgeodesic[m].local27_bad = 1;
      }
    }
    if (!cf2_fin)
      gstate.pgeodesic[m].local27_cf2_bad = 1;
  }
  } /* end if (!gstate.sexact): Christoffel/metric fitting */

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
   Grid-mode sampler helpers (metric_interp = l27 / local27 / aei /
   carpet).  See the parfile keywords for the mode meanings.
   ============================================================================================ */

/* ADMBase fields in the fixed order used by aei/carpet interpolation
   (the betax/betay/betaz fields hold the contravariant shift beta^i) */
static const char *const adm_field[10] = { "ADMBASE::gxx", "ADMBASE::gxy", "ADMBASE::gxz",
                                           "ADMBASE::gyy", "ADMBASE::gyz", "ADMBASE::gzz",
                                           "ADMBASE::alp", "ADMBASE::betax", "ADMBASE::betay",
                                           "ADMBASE::betaz" };
static int adm_varidx[10];
static int adm_varidx_ok = 0;


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

/* ============================================================================================
   local27 (sinterp_mode 3): 125-node fits (one time level each).
   Node order: j = ix*25 + iy*5 + iz (ix outermost), node offsets {-2..+2}
   from the stencil centre.  The fit is three separable 5x5 V5INV
   transforms per component -- exactly the 125x125 inverse Vandermonde
   (MtxA125 = V5INV ^| V5INV ^| V5INV) -- with the same constant-field
   shortcut as the 27-point fits.  Grid points inside the stencil are
   reproduced exactly (each 5-node transform is identity on its nodes).
   ============================================================================================ */

/* 125-node metric fit (10 independent components) for ONE time level */
static void fit_metric_poly125(const gd_data src[125],
                               CCTK_REAL dst[4][4][POLY125],
                               CCTK_INT dstc[4][4])
{
  const double rel_eps = 1e-8;
  double ly[125], t1[5][25], t2[5][25];
  int mu, nu, k, a, b, c;
  for (mu=0;mu<4;mu++)
  for (nu=mu;nu<4;nu++)
  {
    double tmpy=0.0, tol=0.0; int vary=0;
    for (k=0;k<125;k++)
    {
      ly[k] = src[k].gd[mu][nu];
      if (k==0) { tmpy=ly[0]; tol=fabs(tmpy)*rel_eps+rel_eps; vary=0; }
      else if (!vary) { if (ly[k]>tmpy+tol || ly[k]<tmpy-tol) vary=1; }
    }
    if (vary)
    {
      /* pass x: t1[a][iy*5+iz] = sum_ix V5INV[a][ix] ly[ix*25+iy*5+iz] */
      for (a=0;a<5;a++)
      for (int iy=0;iy<5;iy++)
      for (int iz=0;iz<5;iz++)
      {
        double s=0.0;
        for (int ix=0;ix<5;ix++) s += V5INV[a][ix]*ly[ix*25+iy*5+iz];
        t1[a][iy*5+iz]=s;
      }
      /* pass y: t2[a][b*5+iz] = sum_iy V5INV[b][iy] t1[a][iy*5+iz] */
      for (a=0;a<5;a++)
      for (b=0;b<5;b++)
      for (int iz=0;iz<5;iz++)
      {
        double s=0.0;
        for (int iy=0;iy<5;iy++) s += V5INV[b][iy]*t1[a][iy*5+iz];
        t2[a][b*5+iz]=s;
      }
      /* pass z: dst[k=a*25+b*5+c] = sum_iz V5INV[c][iz] t2[a][b*5+iz] */
      for (a=0;a<5;a++)
      for (b=0;b<5;b++)
      for (c=0;c<5;c++)
      {
        double s=0.0;
        for (int iz=0;iz<5;iz++) s += V5INV[c][iz]*t2[a][b*5+iz];
        dst[mu][nu][a*25+b*5+c]=s;
      }
      dstc[mu][nu]=0;
    }
    else
    {
      for (k=1;k<POLY125;k++) dst[mu][nu][k]=0.0;
      dst[mu][nu][0]=tmpy;
      dstc[mu][nu]=1;
    }
  }
  /* symmetry in the lower indices: g_{mu nu} = g_{nu mu} */
  for (mu=1;mu<4;mu++)
  for (nu=0;nu<mu;nu++)
  {
    for (k=0;k<POLY125;k++) dst[mu][nu][k]=dst[nu][mu][k];
    dstc[mu][nu]=dstc[nu][mu];
  }
}

/* 125-node fit of the pointwise Christoffel symbols at the 5x5x5 stencil
   nodes for ONE time level (the caller precomputes the three levels with
   pointwise_cf2_tl); same constant-field shortcut and lower-index
   symmetry as fit_cf2_poly27. */
static void fit_cf2_poly125(const double G[125][4][4][4],
                            CCTK_REAL dst[4][4][4][POLY125],
                            CCTK_INT dstc[4][4][4])
{
  const double rel_eps = 1e-8;
  double ly[125], t1[5][25], t2[5][25];
  int j1, j2, j3, k, a, b, c;
  for (j1=0;j1<4;j1++)
  for (j2=0;j2<4;j2++)
  for (j3=j2;j3<4;j3++)
  {
    double tmpy=0.0, tol=0.0; int vary=0;
    for (k=0;k<125;k++)
    {
      ly[k] = G[k][j1][j2][j3];
      if (k==0) { tmpy=ly[0]; tol=fabs(tmpy)*rel_eps+rel_eps; vary=0; }
      else if (!vary) { if (ly[k]>tmpy+tol || ly[k]<tmpy-tol) vary=1; }
    }
    if (vary)
    {
      for (a=0;a<5;a++)
      for (int iy=0;iy<5;iy++)
      for (int iz=0;iz<5;iz++)
      {
        double s=0.0;
        for (int ix=0;ix<5;ix++) s += V5INV[a][ix]*ly[ix*25+iy*5+iz];
        t1[a][iy*5+iz]=s;
      }
      for (a=0;a<5;a++)
      for (b=0;b<5;b++)
      for (int iz=0;iz<5;iz++)
      {
        double s=0.0;
        for (int iy=0;iy<5;iy++) s += V5INV[b][iy]*t1[a][iy*5+iz];
        t2[a][b*5+iz]=s;
      }
      for (a=0;a<5;a++)
      for (b=0;b<5;b++)
      for (c=0;c<5;c++)
      {
        double s=0.0;
        for (int iz=0;iz<5;iz++) s += V5INV[c][iz]*t2[a][b*5+iz];
        dst[j1][j2][j3][a*25+b*5+c]=s;
      }
      dstc[j1][j2][j3]=0;
    }
    else
    {
      for (k=1;k<POLY125;k++) dst[j1][j2][j3][k]=0.0;
      dst[j1][j2][j3][0]=tmpy;
      dstc[j1][j2][j3]=1;
    }
  }
  /* symmetry in the lower indices: Gamma^mu_{ab} = Gamma^mu_{ba} */
  for (j1=0;j1<4;j1++)
  for (j2=1;j2<4;j2++)
  for (j3=0;j3<j2;j3++)
  {
    for (k=0;k<POLY125;k++) dst[j1][j2][j3][k] = dst[j1][j3][j2][k];
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

/* theta of the particle's coordinate time y[0] within the level's two
   stored time levels.  L = the particle's level; its levels span
   [L->time - L->dt, L->time].  Absolute mapping (independent of sstep
   or bg) keeps the blend phase-correct under time subcycling, where a
   particle may sit anywhere in the window [t-dt, t].  Clamps to [0,1]:
   below the window -> oldest level, above -> newest (clamped). */
static inline double geodesic_time_theta(int m, double y0)
{
  const LevelData *L =
    &gstate.LD[gstate.pgeodesic[m].cur_map][gstate.pgeodesic[m].level];
  const double dt = L->dt;
  if (dt <= 0.0)
    return 1.0;
  double theta = (y0 - (L->time - dt)) / dt;
  if (theta < 0.0)
    theta = 0.0;
  if (theta > 1.0)
    theta = 1.0;
  return theta;
}

/* local27 (sinterp_mode 3): time-Lagrange weights of the stored levels for
   the particle's coordinate time y0.  The three stored levels sit at the
   local fractional time s in [-1,1] as level 0 (t) = +1, level 1 (t-dt) = 0,
   level 2 (t-2dt) = -1, with s = 2*theta - 1 (theta = geodesic_time_theta,
   clamped to [0,1]).  Degraded blends: old_level_bad -> level 0 only
   (theta pinned to 1); local27_lvl2_bad -> two-level blend of [t, t-dt] with
   weights (theta, 1-theta).  nlev = number of active levels (1..3). */
static void local27_time_weights(int m, double y0, double w[3], int *nlev)
{
  double theta = geodesic_time_theta(m, y0);
  if (gstate.pgeodesic[m].old_level_bad)
  {
    *nlev = 1;
    w[0] = 1.0; w[1] = 0.0; w[2] = 0.0;
    return;
  }
  if (gstate.pgeodesic[m].local27_lvl2_bad)
  {
    *nlev = 2;
    w[0] = theta; w[1] = 1.0 - theta; w[2] = 0.0;
    return;
  }
  *nlev = 3;
  {
    const double s = 2.0*theta - 1.0;
    w[0] = fourd_tw(0, s);
    w[1] = fourd_tw(1, s);
    w[2] = fourd_tw(2, s);
  }
}

/* local27 (sinterp_mode 3) fallback: assemble the Christoffel symbols
   analytically from the 125-node METRIC fits (used when the nodal
   Christoffel samples are NaN-poisoned).  Metric value and spatial
   derivatives from the level-0 fit (p125_deriv); the time derivative from
   the nlev-level time blend of the level fits (fourd_tw_a/dt for 3 levels,
   (g0 - g1)/dt for 2, 0 for 1); then the 4x4 inverse + cf2_pointwise.
   Mirrors cf2_metric_off on the 125-node polynomial. */
static int local27_christoffel_analytic(int m, const double y[8], double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  double beta_up[3], gamma_dn[3][3], gamma_up[3][3];
  double w[3], wa[3];
  int nlev, mu, nu;
  const double tx = (y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
  const double ty = (y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
  const double tz = (y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
  local27_time_weights(m, y[0], w, &nlev);
  {
    double theta = geodesic_time_theta(m, y[0]);
    if (gstate.pgeodesic[m].old_level_bad) theta = 1.0;
    if (nlev == 3)
    {
      const double s = 2.0*theta - 1.0;
      wa[0] = fourd_tw_a(0, s)/gview.dt;
      wa[1] = fourd_tw_a(1, s)/gview.dt;
      wa[2] = fourd_tw_a(2, s)/gview.dt;
    }
    else if (nlev == 2)
      { wa[0] = 1.0/gview.dt; wa[1] = -1.0/gview.dt; wa[2] = 0.0; }
    else
      { wa[0] = wa[1] = wa[2] = 0.0; }
  }
  (void)w;
  for (mu=0;mu<4;mu++)
    for (nu=0;nu<4;nu++)
    {
      double g0 = 0.0, dx = 0.0, dy = 0.0, dz = 0.0;
      p125_eval_deriv(gstate.pgeodesic[m].pgd125[0][mu][nu], tx,ty,tz, &g0,&dx,&dy,&dz);
      gdn[mu][nu] = gdn[nu][mu] = g0;
      dgd[1][mu][nu] = dgd[1][nu][mu] = dx/gview.delta_space[0];
      dgd[2][mu][nu] = dgd[2][nu][mu] = dy/gview.delta_space[1];
      dgd[3][mu][nu] = dgd[3][nu][mu] = dz/gview.delta_space[2];
      dgd[0][mu][nu] = dgd[0][nu][mu] = wa[0]*g0;
    }
  for (mu=0;mu<4;mu++)
  for (nu=mu;nu<4;nu++)
  {
    if (nlev >= 2)
    {
      double g1 = 0.0;
      p125_eval_deriv(gstate.pgeodesic[m].pgd125[1][mu][nu], tx,ty,tz, &g1,NULL,NULL,NULL);
      dgd[0][mu][nu] += wa[1]*g1;
      dgd[0][nu][mu] = dgd[0][mu][nu];
    }
    if (nlev >= 3)
    {
      double g2 = 0.0;
      p125_eval_deriv(gstate.pgeodesic[m].pgd125[2][mu][nu], tx,ty,tz, &g2,NULL,NULL,NULL);
      dgd[0][mu][nu] += wa[2]*g2;
      dgd[0][nu][mu] = dgd[0][mu][nu];
    }
  }
  if (!build_gup_from_gdn(gdn, NULL, beta_up, gamma_dn, gamma_up, gup))
    return 0;
  cf2_pointwise(dgd, gup, cf2);
  return 1;
}

/* ===================== 3-node time-Lagrange weights (local27) ============ */

/* 1D 3-node (order-2) Lagrange weight for time level lvl at fractional time
   s in [0,1].  Local node positions: level0 (t) = +1, level1 (t-dt) = 0,
   level2 (t-2dt) = -1.  Sum over lvl of fourd_tw(lvl, s) == 1 for all s. */
static inline double fourd_tw(int lvl, double s)
{
  switch (lvl) {
    case 0: return s * (s + 1.0) / 2.0;   /* level 0 at s = +1  */
    case 1: return 1.0 - s * s;           /* level 1 at s =  0  */
    case 2: return s * (s - 1.0) / 2.0;   /* level 2 at s = -1  */
  }
  return 0.0;
}

/* derivative of fourd_tw with respect to the local fractional time s; the
   physical time derivative is (this) / gview.dt. */
static inline double fourd_tw_a(int lvl, double s)
{
  switch (lvl) {
    case 0: return (2.0 * s + 1.0) / 2.0;
    case 1: return -2.0 * s;
    case 2: return (2.0 * s - 1.0) / 2.0;
  }
  return 0.0;
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
    /* l27: the current-level 27-point metric fit (pgd).  A failed staged
       fit (l27_bad) leaves gdn zeroed (the norm check below then flags
       the particle out-of-bounds): there is no fallback interpolant in
       this mode. */
    if (gstate.pgeodesic[m].l27_bad) return;
    for (mu=0;mu<4;mu++) for (nu=mu;nu<4;nu++)
      gdn[mu][nu] =
        poly27_eval(y,m,gstate.pgeodesic[m].pgd[mu][nu],
                    gstate.pgeodesic[m].pgdc[mu][nu]);
    for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
    return;
  }
  if (gstate.sinterp_mode == 3)
  {
    /* local27: nlev-level time blend of the 125-term quartic tensor-product
       metric fits (committed by cf2_fitting's mode-3 staging).  A failed
       staged metric (local27_bad) leaves gdn zeroed and the caller drops the
       particle -- there is no AEI fallback in this mode. */
    if (gstate.pgeodesic[m].local27_bad) return;
    const double tx=(y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
    const double ty=(y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
    const double tz=(y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
    double w[3]; int nlev;
    local27_time_weights(m, y[0], w, &nlev);
    for (mu=0;mu<4;mu++)
    for (nu=mu;nu<4;nu++)
    {
      double g=0.0;
      for (int l=0;l<nlev;l++)
      {
        double cv;
        p125_eval_deriv(gstate.pgeodesic[m].pgd125[l][mu][nu],tx,ty,tz,&cv,NULL,NULL,NULL);
        g+=w[l]*cv;
      }
      gdn[mu][nu]=g;
    }
    for (mu=1;mu<4;mu++) for (nu=0;nu<mu;nu++) gdn[mu][nu]=gdn[nu][mu];
    return;
  }
  if (gstate.sinterp_mode == 1)
  {
    double g31[10], dg31[10][3];
    {
      int st = aei_interp_point(0,y[1],y[2],y[3],g31,dg31);
      if (st != 0) { gstate.pgeodesic[m].outbd=1; return; }
    }
    {
      double g31a[10], dg31a[10][3];
      /* the two stored ADMBase time levels span [t-dt, t] with t the level
         data time; theta is the particle's absolute phase in that window
         (geodesic_time_theta), clamped to [0,1] */
      double theta = geodesic_time_theta(m, y[0]);
      if (aei_interp_point(1,y[1],y[2],y[3],g31a,dg31a) != 0) { gstate.pgeodesic[m].outbd=1; return; }
      for (mu=0;mu<10;mu++)
      {
        g31[mu]=g31a[mu]+theta*(g31[mu]-g31a[mu]);
        for (nu=0;nu<3;nu++) dg31[mu][nu]=dg31a[mu][nu]+theta*(dg31[mu][nu]-dg31a[mu][nu]);
      }
      compose_4metric(g31,dg31,gdn,NULL,NULL);
    }
    return;
  }
  { /* carpet */
    double g31[10], dg31[10][3];
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) { gstate.pgeodesic[m].outbd=1; return; }
    compose_4metric(g31,dg31,gdn,NULL,NULL);
  }
}

/* Christoffel symbols at a particle position for func_ode (modes aei /
   carpet / local27; the l27 mode evaluates its fitted Christoffel
   polynomial directly in func_ode).  Returns 1 on success, 0 on failure
   (the caller flags the particle and aborts the step). */
static int christoffel_at_point(int m, const double y[8], double bg, double cf2[4][4][4])
{
  double gdn[4][4], dgd[4][4][4], gup[4][4];
  int j1, j2, j3;
  if (gstate.sinterp_mode == 3)
  {
    /* local27: nlev-level time blend of the 125-term Christoffel fits
       (fast path: nodal 6th-order-FD Christoffels fit per level, the l27
       convention).  When the nodal samples are NaN-poisoned
       (local27_cf2_bad) the Christoffel is assembled analytically from the
       metric fits instead.  A failed staged metric (local27_bad) fails the
       call and the ODE drops the particle. */
    if (gstate.pgeodesic[m].local27_bad) return 0;
    if (gstate.pgeodesic[m].local27_cf2_bad)
      return local27_christoffel_analytic(m, y, cf2);
    const double tx=(y[1]-gstate.pgeodesic[m].below[1])/gview.delta_space[0];
    const double ty=(y[2]-gstate.pgeodesic[m].below[2])/gview.delta_space[1];
    const double tz=(y[3]-gstate.pgeodesic[m].below[3])/gview.delta_space[2];
    double w[3]; int nlev;
    local27_time_weights(m, y[0], w, &nlev);
    for (j1=0;j1<4;j1++)
    for (j2=0;j2<4;j2++)
    for (j3=j2;j3<4;j3++)
    {
      double c=0.0;
      for (int l=0;l<nlev;l++)
      {
        double cv;
        p125_eval_deriv(gstate.pgeodesic[m].pcf2_125[l][j1][j2][j3],tx,ty,tz,&cv,NULL,NULL,NULL);
        c+=w[l]*cv;
      }
      cf2[j1][j2][j3]=c;
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
    {
      double g31a[10], dg31a[10][3], gdn0[4][4], gdn1[4][4];
      /* the two stored ADMBase time levels span [t-dt, t] with t the level
         data time; theta is the particle's absolute phase in that window
         (geodesic_time_theta), clamped to [0,1] */
      double theta = geodesic_time_theta(m, y[0]);
      if (aei_interp_point(1,y[1],y[2],y[3],g31a,dg31a) != 0) return 0;
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
  { /* carpet:
     Spatial metric + spatial derivatives at the particle position.
     The spatial sampler is the order-4 Lagrange interpolator (the same
     AEILocalInterp-based routine aei uses) -- the previous hand-written
     quadratic (2nd-order, 3 nodes per dim) sampler was too inaccurate
     in the strong-field region and caused the particle-1 superluminal
     drop; the order-4 sampler fixes it.  The time derivative is the
     carpet-style 2nd-order 3-point difference of the FULL 4-metric at the
     particle position: d/dt = (3*g0 - 4*g1 + g2)/(2*dt). */
    double g31[10], dg31[10][3];
    double g31a[10], g31b[10];
    double gnn1[4][4], gnn2[4][4];
    if (aei_interp_point(0,y[1],y[2],y[3],g31,dg31) != 0) return 0;
    compose_4metric(g31,dg31,gdn,dgd,gup);
    if (gview.metric_static)
    {
      for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++) dgd[0][j1][j2]=0.0;
      cf2_pointwise(dgd,gup,cf2);
      return 1;
    }
    if (aei_interp_point(1,y[1],y[2],y[3],g31a,NULL) != 0) return 0;
    if (aei_interp_point(2,y[1],y[2],y[3],g31b,NULL) != 0) return 0;
    compose_4metric(g31a,NULL,gnn1,NULL,NULL);
    compose_4metric(g31b,NULL,gnn2,NULL,NULL);
    for (j1=0;j1<4;j1++) for (j2=0;j2<4;j2++)
      dgd[0][j1][j2]=(3.0*gdn[j1][j2]-4.0*gnn1[j1][j2]+gnn2[j1][j2])*(0.5/gview.dt);
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
  /* Phase A: all particles are pinned to map 0; Phase B will assign it
     to the block the particle currently sits in. */
  const int pm = gstate.pgeodesic[m].cur_map;

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
    /* Phase B (multi-block) crossing logic is armed only when there is
       genuinely more than one block.  A single-block run (per-hole AMR,
       plain unigrid) has nmaps <= 1 and must never enter
       locate_and_cross(); a particle outside the block's coarse safe
       region there is simply out of the domain.  nmaps is set by the
       one-shot init (0 until then); the > 1 test is safe in all cases. */
    if (!gstate.multi_exact && !gstate.multi_amr &&
        gstate.nmaps > 1 &&
        CCTK_IsFunctionAliased("MultiPatch_GetMap") &&
        !(in_level_dir(y, &gstate.LD[pm][0], 0) &&
          in_level_dir(y, &gstate.LD[pm][0], 1) &&
          in_level_dir(y, &gstate.LD[pm][0], 2)))
    {
      /* Phase B: particle left the current block's margin-3 safe region.
         It may have crossed into a neighbouring block of the multipath.
         locate_and_cross() re-expresses the position in the new block's
         local coords and transforms the spatial 4-velocity so the
         physical motion is continuous.  On success (rc==1) we flag crossed
         and abort this ODE step (GSL_EBADFUNC); the integrator restarts
         from the transformed state.  rc!=1 means no neighbouring block
         owns the point; it may still sit inside the current block's full
         cell (margin shadow zone), in which case cf2_fitting() is re-run
         for it and the particle is dropped there (no extrapolant fallback
         in this version).  Only a point outside the block's full cell
         (with ~2-cell tolerance) is marked out of bounds here. */
      int rc = locate_and_cross(m, (double *)y, odedata->cctkGH);
      if (rc == 1)
      {
        gstate.pgeodesic[m].crossed = 1;
        for (int k = 0; k < 8; k++)
          gstate.pgeodesic[m].crossed_y[k] = y[k];
        return GSL_EBADFUNC;
      }
      /* rc != 1: test whether the particle is still inside the full cell
         of its current block (coarsest level).  If yes it is in the margin
         shadow zone: the existing outbd flag is cleared and cf2_fitting()
         is re-run, which drops the particle there (no extrapolant fallback
         in this version).  If it is outside the cell (beyond ~2 cells), it
         is a genuine out-of-domain event. */
      {
        const LevelData *Lo = &gstate.LD[pm][0];
        if (!Lo->valid)
        {
          gstate.pgeodesic[m].outbd = 1;
        }
        else
        {
          int inside = 1;
          for (int d = 0; d < 3; d++)
          {
            double lo = Lo->i_lbnd[d] - 2.0 * Lo->dx[d];
            double hi = Lo->i_ubnd[d] + 2.0 * Lo->dx[d];
            if (y[d+1] < lo || y[d+1] > hi) { inside = 0; break; }
          }
          if (inside)
          {
            gstate.pgeodesic[m].outbd = 0;
            cf2_fitting(m, y);
          }
          else
          {
            gstate.pgeodesic[m].outbd = 1;
          }
        }
      }
    }else
    {
      const int lvl = gstate.multi_amr ? gstate.owner_lvl[m]
                         : (gstate.multi_exact ? 0 : pick_level(y, pm));
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
          const LevelData *Lo = &gstate.LD[pm][lvl];
          for (int d = 0; d < 3; d++)
          {
            /* mid-step reachability: the stencil is never clipped, so the
               only requirement is that its full window around the nearest
               node fits inside the halo-extended local array, which is
               checked when the stencil is fitted (see the W-window test).
               Here the particle is simply confined to the local grid
               [origin, origin+(lsh-1)*dx].  The old 0.5-cell tolerance
               around the interior box was far too tight for shadow-zone
               particles (outside the interior box but inside the grid, e.g.
               a y=0 equatorial particle on the y+ rank whose interior box
               starts at y=0.25 and which moves in -y): it flagged them
               outbd on the first step. */
            const double lo = Lo->origin[d];
            const double hi = Lo->origin[d]
                              + (CCTK_REAL)((CCTK_INT)Lo->lsh[d] - 1) * Lo->dx[d];
            if (y[d+1] < lo || y[d+1] > hi)
              gstate.pgeodesic[m].outbd = 1;
          }
        }
        /* refit window: the two-cell box [below-dx, below+dx] centred on
           the reference node (below = position of node tpoint = nearest
           node).  The particle sits at local (y-below)/dx in [-0.5,+0.5),
           so it stays inside this box until its nearest node changes;
           refitting exactly at that moment rebuilds the stencil centred on
           the new node and keeps the Christoffel fit continuous.  The 27-pt
           stencil {tpoint-1, tpoint, tpoint+1} spans exactly [below-dx,
           below+dx], so the particle is always inside it -- pure
           interpolation. */
        if (!gstate.pgeodesic[m].outbd &&
            (lvl != gstate.pgeodesic[m].level ||
             (y[1] < (gstate.pgeodesic[m].below[1] - gstate.pgeodesic[m].dxp[0]) || (y[1] > (gstate.pgeodesic[m].below[1] + gstate.pgeodesic[m].dxp[0]))) ||
             (y[2] < (gstate.pgeodesic[m].below[2] - gstate.pgeodesic[m].dxp[1]) || (y[2] > (gstate.pgeodesic[m].below[2] + gstate.pgeodesic[m].dxp[1]))) ||
             (y[3] < (gstate.pgeodesic[m].below[3] - gstate.pgeodesic[m].dxp[2]) || (y[3] > (gstate.pgeodesic[m].below[3] + gstate.pgeodesic[m].dxp[2])))))
        {
          cf2_fitting(m, y);
        }
      }
    }
  }

  if (gstate.pgeodesic[m].outbd)
  {
    return GSL_EBADFUNC;
  }

  /* AMR: ensure this thread's grid view (delta_space, data pointers, ghost cGH)
     matches the particle's level before any interpolation or FD stencil
     access.  switch_to_level is self-guarded (cheap no-op when already on
     the correct level+generation). */
  switch_to_level(pm, gstate.pgeodesic[m].level);

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
    if (gstate.sinterp_mode == 0)
    {
      /* l27: the historical 27-point Lagrange polynomial of the
         current-level fitted Christoffel (cf2_func).  A failed staged fit
         (l27_bad) has no fallback in this mode: flag the particle
         out-of-bounds and let the loss policy drop it. */
      if (gstate.pgeodesic[m].l27_bad)
      {
        gstate.pgeodesic[m].outbd = 1;
        return GSL_EBADFUNC;
      }
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

/* Write the dump preamble: revision stamp, the settings that produced the
   file, then the column names.  Every line is a '#' comment so existing
   readers (which skip '#' lines) are unaffected. */
static void geodesic_dump_header(FILE *fp)
{
  DECLARE_CCTK_PARAMETERS

  fprintf(fp, "# Geodesic revision %s\n", GEODESIC_REVISION);
  fprintf(fp, "# config: excised_radius=%g initial_radius=%g Exact=%s"
              " metric_interp=%s particle_loss_policy=%s particle_n_total=%d\n",
          (double)excised_radius, (double)initial_radius,
          Exact ? "yes" : "no", metric_interp, particle_loss_policy,
          (int)particle_n_total);
  fprintf(fp, "# t x y z ut ux uy uz tau\n");
}

#ifdef CCTK_MPI
/* Multi-process AMR: exchange the full shared particle state with
   MPI_Allgather so every rank ends with the identical state.
   Payload per particle (GEODESIC_MPI_NPAY doubles):
   tt tx ty tz ttv txv tyv tzv ttw dead renorm_warn norm_bad
   freeze_warn rl_last outbd local27_cf2_bad old_level_bad
   local27_lvl2_bad local27_bad.  Full exchange (200 x 19 =
   30.4 KB/rank) keeps the code simple; every rank holds the complete
   state.  The particle arrays are
   DISTRIB=CONSTANT grid variables, addressed through the patch-local
   pointers that DECLARE_CCTK_ARGUMENTS injects (cctkGH). */
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
    p[15] = (double)gstate.pgeodesic[mi].local27_cf2_bad;
    p[16] = (double)gstate.pgeodesic[mi].old_level_bad;
    p[17] = (double)gstate.pgeodesic[mi].local27_lvl2_bad;
    p[18] = (double)gstate.pgeodesic[mi].local27_bad;
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
    gstate.pgeodesic[mi].local27_cf2_bad  = (CCTK_INT)p[15];
    gstate.pgeodesic[mi].old_level_bad    = (CCTK_INT)p[16];
    gstate.pgeodesic[mi].local27_lvl2_bad = (CCTK_INT)p[17];
    gstate.pgeodesic[mi].local27_bad      = (CCTK_INT)p[18];
    /* The sticky fit-history flags above decide, per particle, whether the
       local27 Christoffels come from the fitted poly or from the analytic
       assembly.  They must travel with the particle: if a new owner lost
       them it would integrate the same geodesic through a different
       Christoffel path.  The cached fits themselves (pgd125/pcf2_125 and
       their validity flag) deliberately do NOT travel -- the new owner does
       not have them and re-fits from the grid it now owns. */
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
   grid time level): the linear time blend in func_ode /
   christoffel_at_point anchors theta = (y[0] - bg)/sstep at bg, so
   bg must stay at the step start for every chunk of the multi_amr
   chunked integration.  Anchoring it at the particle's chunk-start
   time instead shifts the metric blend window by the elapsed chunks
   (the metric lags behind the true time), which corrupted the linear
   time interpolation in multi-process runs (1p is unaffected: one
   chunk, bg == step start).
------------------------------------------------------------------------------------------------- */
/* multi_amr: the gathered per-rank per-level boxes (Phase 1a) and the claim
   margin used by geodesic_recompute_owner_map, shared with
   geodesic_integrate_particle so the per-particle ownership-consistency
   check can use the SAME box + margin the owner map used (rather than a
   strict interior-box membership that would reject every gap-claimed
   particle).  bd is a fresh allocation on each geodesics_integrate call
   and is freed after the chunk loop, so it is valid for the whole
   integration. */
#ifdef CCTK_MPI
static double *s_owner_bd = NULL;
static double  s_owner_margin = 4.0;
#endif
static void geodesic_integrate_particle(cGH *cctkGH, const int mi, const double t_end, const double t_step0)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  const int pm = gstate.pgeodesic[mi].cur_map;

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
         (particle_tz[mi] != 0.0 && (!isnormal(particle_tz[mi]))) )
    {
      gstate.pgeodesic[mi].outbd = 1;
    }
    else if (!gstate.multi_exact && !gstate.multi_amr)
    {
      /* margin-appropriate coarse-level (level 0) safe region, using the
         same margin as in_level_dir / pick_level.  For the pointwise
         aei/carpet branch (sinterp_mode 1/2) the margin is small so a
         valid particle near a patch edge is not wrongly rejected
         (the old 7-cell margin rejected equatorial particles in the
         7-cell shadow zone of an off-centre AMR block).
         The l27 path (sinterp_mode 0) is skipped here: cf2_fitting does
         the full margin-3 (6th-order-FD) check and, when that safe region
         rejects the particle, sets l27_bad and the particle is dropped by
         the loss policy (no fallback interpolant in l27).  Skipping the
         quick check here avoids a premature outbd that cf2_fitting would
         otherwise be unable to undo. */
      if (gstate.sinterp_mode != 0)
      {
        const LevelData *L0 = &gstate.LD[pm][0];
        const int margin = safe_margin_cells();
        const double y[3] = { particle_tx[mi], particle_ty[mi], particle_tz[mi] };
        for (int d = 0; d < 3; d++)
        {
          const double lo = L0->origin[d] + (CCTK_REAL)margin * L0->dx[d];
          const double hi = L0->origin[d] + (CCTK_REAL)(L0->lsh[d]-1-margin) * L0->dx[d];
          if (y[d] < lo || y[d] > hi) { gstate.pgeodesic[mi].outbd = 1; break; }
        }
      }
    }

    /* ownership consistency (multi_amr): at a chunk start the owner map
       (recomputed from the synced positions at step start and at every
       chunk boundary) must agree with the position.  A particle that has
       drifted beyond the claim margin from its owner's box means stale
       ownership -- hand it to the loss policy.  In normal operation this
       never fires: the re-owning is computed from the same synced positions
       and uses the same claim margin.
       CRITICAL: compare against the owner's GATHERED box (Phase 1a,
       s_owner_bd) with the claim margin, NOT a strict interior-box
       membership test.  The owner map claims a particle to a box even when
       the particle lies OUTSIDE that box (gap up to margin cells) -- that
       is how the inter-rank / inter-level gap band is covered (e.g. an
       equatorial particle at z=0 claimed to a coarse box whose interior
       edge sits at z=+0.25, or a particle in the fine gap band claimed to
       the coarse level).  A strict interior-box test would falsely reject
       every such gap-claimed particle (outbd=1 on the very first
       integration step).  A claimed particle is always within margin of the
       patch edge where the stencil is clamped, so it is safe to integrate. */
#ifdef CCTK_MPI
    if (gstate.multi_amr)
    {
      const int owner = gstate.owner_of[mi];
      const int ol    = gstate.owner_lvl[mi];
      if (owner >= 0 && ol >= 0 && s_owner_bd != NULL)
      {
        const int nlev = geodesic_amr_levels;
        const double *bo = s_owner_bd + (size_t)owner * (size_t)nlev * 11
                                 + (size_t)ol * 11;
        const double px = particle_tx[mi], py = particle_ty[mi], pz = particle_tz[mi];
        double gx = (px < bo[0]) ? (bo[0] - px) : ((px >= bo[3]) ? (px - bo[3]) : 0.0);
        double gy = (py < bo[1]) ? (bo[1] - py) : ((py >= bo[4]) ? (py - bo[4]) : 0.0);
        double gz = (pz < bo[2]) ? (bo[2] - pz) : ((pz >= bo[5]) ? (pz - bo[5]) : 0.0);
        double gap     = fmax(gx, fmax(gy, gz));
        const double dxmax = fmax(bo[8], fmax(bo[9], bo[10]));
        const double gap_cells = (dxmax > 0.0) ? (gap / dxmax) : 1e100;
        if (gap_cells > s_owner_margin)
        {
          gstate.pgeodesic[mi].outbd = 1;
        }
      }
    }
#else
#endif

    /* ---------------- integrate one particle ---------------- */
    memset((void *)&odedata, 0, sizeof(odedata));
    odedata.m  = mi;
    odedata.bg = t_step0;
    odedata.st = &gstate;
    odedata.cctkGH = cctkGH;   /* Phase B: multipatch transform geometry */

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
      else if (gstate.sinterp_mode == 0)
      {
        /* local27: full 4-velocity renormalization: rescale all four
           components by 1/sqrt(-rl) so that g(u,u) = -1 holds exactly in
           this metric frame; the ODE-produced direction is kept as-is.
           (This full rescale fixes the local27 norm drift.) */
        const double inv_n = 1.0 / sqrt(-rl);
        u[0] *= inv_n; u[1] *= inv_n; u[2] *= inv_n; u[3] *= inv_n;
        if (!isfinite(u[0]) || u[0] <= 0.0)
        {
          gstate.pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
          gstate.pgeodesic[mi].outbd = 1;   /* let the loss policy below handle this particle */
        }
        else
        {
          ys[4] = u[0]; ys[5] = u[1]; ys[6] = u[2]; ys[7] = u[3];
          particle_ttv[mi] = ys[4];
        }
      }
      else
      {
        /* aei / carpet: keep the spatial velocity unchanged and re-solve
           u^0 (legacy behavior).  A full rescale would amplify the spatial
           velocity when the IC norm deviates from -1 in this metric (the
           TP-ring IC has u.u well off -1), flinging particles deeper into
           the strong field than intended. */
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
    /* integrate in tau until coordinate time reaches target */
    const double t_target = t_end;
    CCTK_REAL h = hstart;
    int retc = GSL_SUCCESS, max_iter = 0;

    while (ys[0] < t_target && max_iter < max_iter_cap)
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
        /* Phase B: the only non-genuine ODE failure we emit on purpose
           is GSL_EBADFUNC from a detected block crossing (func_ode set
           crossed=1 and stored the transformed state in crossed_y[]).
           In that case the step is abandoned and the ODE is restarted
           from the transformed state in the NEW block, at the time at
           which the crossing was detected.  Any other failure is a real
           numerical out-of-bounds. */
        if (gstate.pgeodesic[mi].crossed)
        {
          gstate.pgeodesic[mi].crossed = 0;
          memcpy(ys, gstate.pgeodesic[mi].crossed_y, 8 * sizeof(double));
          ts = ys[0];          /* time at the crossing */
          h  = hstart;         /* re-converge the step size */
          continue;           /* loop recomputes tau_end from the new ts */
        }
        gstate.pgeodesic[mi].outbd = 1;
        ys[0] = t_target;
        break;
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
          if (gstate.sinterp_mode == 0)
          {
            /* local27: full 4-velocity renormalization (same as step-start):
               rescale all four components by 1/sqrt(-rl) so that g(u,u) = -1
               holds exactly in this metric frame (rl = raw pre-renorm value) */
            const double inv_n = 1.0 / sqrt(-rl);
            u[0] *= inv_n; u[1] *= inv_n; u[2] *= inv_n; u[3] *= inv_n;
            if (!isfinite(u[0]) || u[0] <= 0.0)
            {
              gstate.pgeodesic[mi].norm_bad = 1;   /* aggregated after the loop */
              gstate.pgeodesic[mi].outbd = 1;
            }
            else
            {
              ys[4] = u[0]; ys[5] = u[1]; ys[6] = u[2]; ys[7] = u[3];
            }
          }
          else
          {
            /* aei / carpet: u^i unchanged, re-solve u^0 (legacy behavior;
               see the step-start branch for the rationale) */
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
/* multi_amr: (re-)compute the owner map from the gathered per-rank
   per-level boxes (Phase 1b).  A particle is claimed to the (rank,
   level) with the SMALLEST L-infinity gap to that level's box, where
   the box is the safe region for l>=1 (r_lbnd/r_ubnd) and the full
   interior box for l==0.  The gap is expressed in cells of the claimed
   level (gap / dxmax) and must not exceed the claim margin (default
   4 cells, tune via GEODESIC_OWNER_MARGIN).  This mirrors the
   single-process pick_level() fallback: a particle in the fine gap
   has a smaller gap to a coarse box and so falls to the coarse level.
   A particle whose gap to EVERY box exceeds the margin is genuinely
   un-integrable: it is flagged outbd (owner=-1) and the chunk loop's
   owner<0 branch hands it to the loss policy.  The patch extends ~7-8
   cells beyond the safe region, so a margin of 4 keeps a claimed
   particle inside the patch, where the (clipped) local27 stencil and
   the pointwise robust path read valid cells -- matching the 0-NaN
   single-process run.  Every rank computes the identical map.  Called
   at the step start (Phase 1) and at every chunk boundary (re-owning);
   the boxes are fixed during the step, so the Phase 1 buffer is reused. */
static void geodesic_recompute_owner_map(cGH *cctkGH, const double *bd, const int npr, const int nlev)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS
  /* claim margin in cells; default 4.  Tune via GEODESIC_OWNER_MARGIN
     without recompiling.  Must stay well below the ~7-8 cell gap between
     the safe region and the patch edge so a claimed particle stays inside
     the patch (valid stencil reads). */
  double env_margin = 4.0;
  {
    const char *em = getenv("GEODESIC_OWNER_MARGIN");
    if (em) env_margin = atof(em);
  }
#ifdef CCTK_MPI
  s_owner_margin = env_margin;
#endif
  int mi;
  for (mi = 0; mi < particle_n_total; mi++)
  {
    int owner = -1, olvl = -1, claimable = 0;
    double best_gap = 1e100;
    int best_r = -1, best_l = -1;
    double best_dx0 = 1.0, best_dx1 = 1.0, best_dx2 = 1.0;
    if (!gstate.pgeodesic[mi].dead)
    {
      const double px = particle_tx[mi], py = particle_ty[mi], pz = particle_tz[mi];
      /* iterate all (level,rank); finest level first so a tie in gap is
         broken toward the finer level.  Strict "<" keeps the first
         (finer) block on equal gaps. */
      for (int l = nlev - 1; l >= 0; l--)
        for (int r = 0; r < npr; r++)
        {
          const double *b = bd + (size_t)r * nlev * 11 + (size_t)l * 11;
          if (b[6] == 0.0) continue;
          double gx = (px < b[0]) ? (b[0] - px) : ((px >= b[3]) ? (px - b[3]) : 0.0);
          double gy = (py < b[1]) ? (b[1] - py) : ((py >= b[4]) ? (py - b[4]) : 0.0);
          double gz = (pz < b[2]) ? (b[2] - pz) : ((pz >= b[5]) ? (pz - b[5]) : 0.0);
          double gap = fmax(gx, fmax(gy, gz));
          if (gap < best_gap)
          {
            best_gap = gap;
            best_r = r; best_l = l;
            best_dx0 = b[8]; best_dx1 = b[9]; best_dx2 = b[10];
          }
        }
      if (best_r >= 0)
      {
        double dxmax = fmax(best_dx0, fmax(best_dx1, best_dx2));
        double gap_cells = (dxmax > 0.0) ? (best_gap / dxmax) : 1e100;
        if (gap_cells <= env_margin)
        {
          owner = best_r; olvl = best_l; claimable = 1;
        }
      }
    }
    if (!claimable)
    {
      /* genuinely un-integrable: every box is farther than the margin.
         Flag outbd + owner=-1; the chunk loop's outbd branch (owner<0)
         drops it via the loss policy. */
      owner = -1;
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
  const int lvl = (int)lrint(log2((double)cctk_levfac[0]));

  /* The particle-state diagnostic dump -- exactly one block per
     cctk_iteration, in time order, particles in index order (so a plotting
     program sees, for every particle, exactly one record per point,
     chronologically) -- is written AFTER the ODE loop below, on the
     finest-level call (the only call that integrates).  Each block therefore
     holds the POST-integration state (the particle's position at the end of
     the step).  last_dump_iter guarantees one block per iteration even
     though several levels report each iteration; rank 0 only; gated by
     particle_dump_every (N -> one block per N iterations); cadence is
     independent of CarpetRegrid2::regrid_every.  See the block after the
     multi_amr S2 sync. */

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
     Only a 2x2 (y,z) decomposition of at most 4 ranks is supported.
     This is the ONLY valid multi-process mode for Exact = no: the
     alternative (rank 0 integrates alone) cannot work because rank 0
     owns only its own patch of the decomposed grid, so particles in
     the other ranks' quadrants interpolate out-of-grid and are dropped
     (the np4 NaN bug).  Hence multi_amr is enabled for EVERY
     multi-process non-exact run, including a single tracked level
     (geodesic_amr_levels = 1): the level-0 interior boxes of the ranks
     tile the domain, so the owner map is well-defined. */
  gstate.multi_amr = (CCTK_nProcs(cctkGH) > 1) && !Exact;
  if (gstate.multi_amr && CCTK_nProcs(cctkGH) > 4)
    CCTK_ERROR("Geodesic (multi-process AMR): at most 4 MPI ranks are supported.");

  /* parse metric_interp before the acquire block: the halo update below
     must know whether the sampler is a stencil (l27/local27) or pointwise.
     The keyword is fixed for the run, so parse once and cache.  (The
     authoritative parse that selects the particle cache pools runs later
     in this routine, on every call; same string, same result.) */
  {
    static int sinterp_mode_cached = -1;
    if (sinterp_mode_cached < 0)
    {
      if      (strcmp(metric_interp, "aei")     == 0) sinterp_mode_cached = 1;
      else if (strcmp(metric_interp, "carpet")  == 0) sinterp_mode_cached = 2;
      else if (strcmp(metric_interp, "local27") == 0) sinterp_mode_cached = 3;
      else                                            sinterp_mode_cached = 0;
    }
    gstate.sinterp_mode = sinterp_mode_cached;
  }

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
    /* Guard: in a single-patch (unigrid/AMR) run the MultiPatch thorn is not
       active, so MultiPatch_GetMap's alias is unregistered; treat map as 0.
       In a genuine multipatch run it returns the current map index. */
    const int map = CCTK_IsFunctionAliased("MultiPatch_GetMap")
                    ? MultiPatch_GetMap(cctkGH)
                    : 0;
    LevelData *L;
    if (level >= GEODESIC_MAX_LEVELS)
      return;   /* more levels than LD[] holds (GEODESIC_MAX_LEVELS): ignore */

    if (level < 0)
      CCTK_WARN(CCTK_WARN_ABORT,
                "Geodesic: negative refinement level is invalid.");
    /* Levels beyond geodesic_amr_levels are also acquired and tracked in
       nlev_seen so the trigger can fire on the finest level actually present
       (subcycling / regrid).  geodesic_amr_levels still only controls the
       particle's level choice (samr_levels), unchanged. */

    /* With time subcycling, Carpet calls CCTK_POSTSTEP once per level per
       substep; the trigger logic at the end of this block integrates
       only at the moment all levels hold data at the same time. */

    L = &gstate.LD[map][level];
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
         x(0) is reconstructed with the same convention used by the other
         thorns that locate grid cells (see SphericalSurface_SetupRes,
         "physical(i) = CCTK_ORIGIN_SPACE(d) + (cctk_lbnd[d] + i) *
          CCTK_DELTA_SPACE(d)"), so array index 0 is
           x(0) = CCTK_ORIGIN_SPACE(i) + cctk_lbnd[i] * CCTK_DELTA_SPACE(i)
         with CCTK_DELTA_SPACE = cctk_delta_space/levfac = dx_lev and
         CCTK_ORIGIN_SPACE = cctk_origin_space + dx_lev*levoff/levoffdenom.
         This is level-independent (level 0 has lbnd=0, levoff=0, so
         x(0)=cctk_origin_space) and patch-independent (centred 1p, off-
         centre 1p, and per-rank multi-process quadrant patches alike).  The
         alignment check below is informational only: an un-aligned 1p patch
         is handled correctly by the lbnd-based origin, so we warn (not
         abort) so 1p AMR runs can be diagnosed. */
      L->dx[i]     = cctk_delta_space[i] / cctk_levfac[i];
      L->inv_dx[i] = 1.0 / L->dx[i];
      /* origin = physical position of the patch's array index 0.

         Mirrors the convention used by the other thorns that locate grid
         cells in the GW150914 run.  SphericalSurface_SetupRes computes the
         physical position of local grid index j as
            physical(j) = CCTK_ORIGIN_SPACE(d) + (cctk_lbnd[d] + j)
                          * CCTK_DELTA_SPACE(d)
         so array index 0 (which is what L->origin must be) is at
            L->origin[d] = CCTK_ORIGIN_SPACE(d) + cctk_lbnd[d] * CCTK_DELTA_SPACE(d)
         with
            CCTK_DELTA_SPACE(d) = cctk_delta_space[d]/cctk_levfac[d] = L->dx[d],
            CCTK_ORIGIN_SPACE(d) = cctk_origin_space[d] + L->dx[d]
                                   * cctk_levoff[d]/cctk_levoffdenom[d].
         This is level-independent (level 0 has lbnd=0, levoff=0 ->
         cctk_origin_space) and patch-independent: it is the index-0 position
         of whichever patch Carpet hands this rank, whether a full centred
         patch (1p), an off-centre patch, or a per-rank quadrant of a
         multi-process AMR run.  It uses only geometry already in the cGH,
         so no patch-geometry reconstruction is needed. */
      L->origin[i] = CCTK_ORIGIN_SPACE(i) + (CCTK_REAL)cctk_lbnd[i] * L->dx[i];
      /* Alignment check (single-process fine levels): the local array
         holds the regrid box plus ghost zones on both sides, so the
         interior box spans nint = lsh - 2*ngz cells.  For a patch that
         aligns with the coarse grid AND is centred on the origin, nint must
         be a multiple of levfac (divisibility) and nint/levfac even (centred).
         Off-centre AMR patches (e.g. GW150914 double-centre) and
         mis-aligned regrid boxes fail one or both tests.  The patch origin
         is computed from the real lbnd (see above), so an un-aligned 1p
         patch is handled correctly regardless.  These diagnostics are
         expected for per-hole AMR and are emitted only when
         Geodesic::amr_align_warn="yes" (off by default), so they do not
         flood the log on every acquire.  Level 0 is exempt (a plain
         unigrid domain need not be centered). */
      if (amr_align_warn)
      {
        if (level > 0 && CCTK_nProcs(cctkGH) == 1)
        {
          const int nint = (int)cctk_lsh[i] - 2 * (int)cctk_nghostzones[i];
          if (nint <= 0 || nint % cctk_levfac[i] != 0)
            CCTK_WARN(CCTK_WARN_ALERT,
                      "Geodesic: fine-level interior box (lsh - 2*ngz) is not "
                      "a multiple of levfac; the patch does not align with the "
                      "coarse grid.  Origin taken from lbnd (un-aligned 1p "
                      "patch, handled).");
          if (nint > 0 && (nint / cctk_levfac[i]) % 2 != 0)
            CCTK_WARN(CCTK_WARN_ALERT,
                      "Geodesic: fine-level interior box is not symmetric about "
                      "the origin (nint/levfac odd).  Origin taken from lbnd "
                      "(off-centre 1p patch, handled).");
        }
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

    L->dt   = cctk_delta_time;
    L->time = cctk_time;

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
      L->metric_static = 1;
    }

    /* multi-process stencil halo: build the padded copies and exchange the
       face bands with the neighbouring ranks (no-op single-process,
       Exact=yes, and the pointwise samplers).  Runs on every rank for
       every level, in the same order everywhere, so its collectives
       match; must run before the integration trigger below. */
    geodesic_halo_update(cctkGH, map, level);

    /* minimal ghost cGH exposing THIS level's geometry.  Only the fields
       set here are valid (everything else is zero); this lets legacy
       code such as g4dn() in derivative.h keep using
       CCTK_GFINDEX3D(gview.GH,...) against the current level's view.  It
       must NOT be used for CCTK_VarDataPtrI or any other driver API. */
    memset(&L->gh, 0, sizeof(cGH));
    for (int d = 0; d < 3; d++)
      L->halo_ash[d] = (L->halo[0][0] != NULL)
                         ? L->lsh[d] + 2 * GEODESIC_HALO_W : L->lsh[d];
    L->gh.cctk_lsh          = L->lsh;
    L->gh.cctk_ash          = L->halo_ash; /* CCTK_GFINDEX3D uses ash, not lsh */
    L->gh.cctk_lbnd         = L->lbnd;
    L->gh.cctk_ubnd         = L->ubnd;
    L->gh.cctk_nghostzones  = L->nghostzones;
    L->gh.cctk_levfac       = L->levfac;
    L->gh.cctk_origin_space = L->origin;
    L->gh.cctk_delta_space  = L->dx;
    L->gh.cctk_delta_time   = cctk_delta_time;
    L->gh.cctk_time         = cctk_time;
    L->gh.cctk_iteration    = cctk_iteration;

    L->valid = 1;
    gstate.ld_generation++;

    /* ---------------------------------------------------------------
       Trigger: integrate exactly once per call-batch, on the batch's
       finest-level call.

       A "batch" is one cctk_iteration's worth of POSTSTEP acquires: under
       time subcycling the finest level's POSTSTEP marks the batch (it is
       due every trf-fine substep) and coarser levels join the batch when
       their own substep is due, in coarse-to-fine order.  The trigger fires
       on the call whose level is the tracked finest (nlev_seen-1) -- the
       last call of the batch -- so each batch integrates once, advancing a
       particle by the finest level's own dt (sstep = elapsed cctk_time, set
       later).  That call's acquire has just run, so its data is fresh.

       Batch tracking (batch_iter / batch_maxlvl, init -1): a new
       cctk_iteration marks a batch boundary; at each boundary we apply the
       regrid decay -- if the previous batch never reached nlev_seen-1, that
       level was removed by a regrid and nlev_seen would otherwise wait
       forever for it; decay nlev_seen to the previous batch's actual
       highest level + 1.  nlev_seen also grows on any level first seen
       (level + 1 > nlev_seen).

       Iteration 0: the it==0 guard below waits for the finest level's first
       call so the one-shot u^mu init sees fresh LevelData on every level
       (the coarsest is acquired first).  Single-level runs
       (geodesic_amr_levels = 1) fire on their only call.
       --------------------------------------------------------------- */
    /* Batch boundary: a new cctk_iteration ends the previous batch (apply
       the regrid decay) and starts this one. */
    if (gstate.batch_iter != cctk_iteration)
    {
      if (gstate.batch_iter >= 0 && gstate.batch_maxlvl >= 0 &&
          gstate.batch_maxlvl + 1 < gstate.nlev_seen)
        gstate.nlev_seen = gstate.batch_maxlvl + 1;
      gstate.batch_iter   = cctk_iteration;
      gstate.batch_maxlvl = -1;
    }
    /* Track this batch's highest level and the global highest seen. */
    if (level > gstate.batch_maxlvl)
      gstate.batch_maxlvl = level;
    if (level + 1 > gstate.nlev_seen)
      gstate.nlev_seen = level + 1;
    if (level == 0)
      gstate.l0_seen_iter = cctk_iteration;

    if (level != gstate.nlev_seen-1)
      return;                               /* integrate only on the finest-level call */
    if (cctk_iteration == 0 && level < geodesic_amr_levels - 1)
      return;                               /* iteration 0: wait for the finest level  */
    if (gstate.last_integ_iter == cctk_iteration)
      return;                               /* already integrated in this batch        */
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
     per iteration at 10^4 particles.

     The stencil arrays are sampler-specific (see Geodesic.h): the
     27-node group is only read by metric_interp = l27, the 125-node
     group only by metric_interp = local27, and aei / carpet /
     Exact = yes use neither.  Only the pool of the selected sampler is
     allocated; every particle binds pointers to its own pool slot, so
     an unused sampler costs 0 bytes per particle. */

  /* metric_interp (grid mode only; the Exact = yes branch ignores it).
     Parsed here, before the cache is allocated, because the sampler
     selects the cache pools.  The remaining linear constraints
     (Exact = no, not carpet, ADMBase metric_timelevels >= 2) are
     checked in ParamCheck. */
  if (strcmp(metric_interp, "aei") == 0)
    gstate.sinterp_mode = 1;
  else if (strcmp(metric_interp, "carpet") == 0)
    gstate.sinterp_mode = 2;
  else if (strcmp(metric_interp, "local27") == 0)
    gstate.sinterp_mode = 3;
  else
    gstate.sinterp_mode = 0;   /* "l27" */

  if (gstate.pgeodesic == NULL || gstate.pgeodesic_capacity < particle_n_total)
  {
    free(gstate.pgeodesic);
    free(gstate.pool27);
    free(gstate.pool125);
    gstate.pgeodesic = NULL;
    gstate.pool27 = NULL;
    gstate.pool125 = NULL;

    gstate.pgeodesic = (ParticleGeodesic * ) calloc(particle_n_total, sizeof(ParticleGeodesic));
    if (!gstate.pgeodesic)
      CCTK_WARN(CCTK_WARN_ABORT, "Error: calloc(pgeodesic) failed.");

    if (gstate.sinterp_mode == 3)
    {
      gstate.pool125 = (cache125 * ) calloc(particle_n_total, sizeof(cache125));
      if (!gstate.pool125)
        CCTK_WARN(CCTK_WARN_ABORT, "Error: calloc(pool125) failed (metric_interp = local27).");
    }
    else if (gstate.sinterp_mode == 0)
    {
      gstate.pool27 = (cache27 * ) calloc(particle_n_total, sizeof(cache27));
      if (!gstate.pool27)
        CCTK_WARN(CCTK_WARN_ABORT, "Error: calloc(pool27) failed (metric_interp = l27).");
    }

    gstate.pgeodesic_capacity = particle_n_total;

    /* Phase A: all particles are pinned to the center block (map 0);
       Phase B (multi-block support) reassigns cur_map at each integration.
       The pool pointers are bound once and stay valid until the next
       reallocation (the capacity only ever grows). */
    for (i = 0; i < particle_n_total; i++)
    {
      ParticleGeodesic *p = gstate.pgeodesic + i;
      p->cur_map = 0;
      if (gstate.pool125)
      {
        p->pcf2_125  = gstate.pool125[i].pcf2_125;
        p->pcf2_125c = gstate.pool125[i].pcf2_125c;
        p->pgd125    = gstate.pool125[i].pgd125;
        p->pgd125c   = gstate.pool125[i].pgd125c;
      }
      if (gstate.pool27)
      {
        p->cf2g  = gstate.pool27[i].cf2g;
        p->gd    = gstate.pool27[i].gd;
        p->pcf2  = gstate.pool27[i].pcf2;
        p->pcf2c = gstate.pool27[i].pcf2c;
        p->pgd   = gstate.pool27[i].pgd;
        p->pgdc  = gstate.pool27[i].pgdc;
      }
    }
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

    /* Phase 1a: Allgather of per-rank per-level boxes
       (nlev * 11 doubles/rank: box_lbnd[3], box_ubnd[3], valid,
        metric_static, dx[0..2]).
       The owner map must use the 7-cell safe region, NOT the interior box,
       for the FINE levels: the interior box extends into the ghost region
       (where the fine GF has no data) and beyond the safe margin, so a
       particle in that band would be handed to a fine level whose 27-point
       stencil reads uninitialized cells -> NaN.  The safe region is a strict
       subset of the refinement region in which the stencil is guaranteed to
       fit.  A particle in the gap between rank safe regions (or the
       fine-level gap band) falls through to a coarser level, which is the
       correct, valid-data choice.  For the COARSEST level the safe region
       shrinks the rank quadrant and leaves an inter-rank gap that would
       make a particle unassignable, so we use the full interior box there
       (the 27-point stencil is clamped to the valid cell range in
       cf2_fitting()).  b[6] carries L->valid (acquired at least once), not
       a "fresh this iteration" flag: with subcycled AMR the trigger fires on
       the finest-level call, where only the fine levels have been acquired
       this iteration and the coarse levels are (harmlessly) stale.  A valid
       coarse level is still spatially valid; when its metric is static the
       value is time-independent, and AEI/carpet time-interpolates otherwise,
       so a stale-but-valid box is a correct owner. */
    /* assign the function-scope bd (kept until the chunk loop frees it);
       a block-local declaration here would shadow it, leaving the chunk
       loop's re-owning to dereference NULL */
    /* 11 doubles per (rank,level): box_lbnd[3], box_ubnd[3], valid,
       metric_static, dx[0..2].  dx is gathered so the claim margin can be
       expressed in cells (inter-rank interior-box gaps are ~2*ngz*dx and
       must be covered so a particle in the gap is still claimable). */
    bd = (double *) malloc((size_t)npr * nlev * 11 * sizeof(double));
    if (!bd)
      CCTK_WARN(CCTK_WARN_ABORT, "Geodesic: box Allgather buffer allocation failed.");
    {
      double *s0 = bd + (size_t)me * nlev * 11;
      for (int l = 0; l < nlev; l++)
      {
        LevelData *L = &gstate.LD[0][l];
        double *s = s0 + (size_t)l * 11;
        /* fall back to the interior box if the safe region is degenerate
           (a patch too thin in any direction has r_lbnd > r_ubnd and the
           stencil cannot fit anywhere in it anyway). */
        for (int i = 0; i < 3; i++) {
          if (l == 0) {
            /* Coarsest level: use the interior box (which tiles the domain,
               no gaps between ranks).  The 7-cell safe region shrinks the
               coarsest-level region and creates inter-rank gaps in which a
               particle would be unassigned (owner_lvl=-1).  At the coarsest
               level there is no finer level to fall through to, and the
               27-point stencil is clamped to the valid cell range in
               cf2_fitting(), so using the full interior box is safe. */
            s[i]      = L->i_lbnd[i]; s[i+3] = L->i_ubnd[i];
          } else {
            if (L->r_lbnd[i] < L->r_ubnd[i]) {
              s[i]      = L->r_lbnd[i]; s[i+3] = L->r_ubnd[i];
            } else {
              s[i]      = L->i_lbnd[i]; s[i+3] = L->i_ubnd[i];
            }
          }
        }
        s[6] = L->valid ? 1.0 : 0.0;
        s[7] = L->metric_static;
        s[8] = L->dx[0]; s[9] = L->dx[1]; s[10] = L->dx[2];
      }
      MPI_Allgather(s0, nlev*11, MPI_DOUBLE, bd, nlev*11, MPI_DOUBLE,
                    MPI_COMM_WORLD);
    }
    /* publish the gathered boxes to the per-particle consistency check
       (geodesic_integrate_particle); they are fixed for the whole step. */
#ifdef CCTK_MPI
    s_owner_bd = bd;
#endif

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

  /* The metric_interp keyword is parsed earlier, right before the
     particle cache is allocated (the sampler selects the cache pools). */
  gstate.cur_gh = cctkGH;

  /* Resolve the AEI interpolator handle for the aei / carpet grid runs:
     modes 0 (l27) and 3 (local27) are fully self-contained and must
     run WITHOUT the AEI thorns; modes 1 (aei) and 2 (carpet) call the
     pointwise AEI order-4 interpolant.  The interpolators are registered
     at STARTUP, after PARAMCHECK. */
  if (!gstate.sexact && (gstate.sinterp_mode == 1 || gstate.sinterp_mode == 2)
      && gstate.aei_handle < 0)
  {
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

  /* default: advance by the time elapsed since the previous firing, i.e.
     the interval between consecutive finest-level calls.  Under subcycling
     that is the finest level's own dt; for a single-level run it equals
     the grid dt (bit-identical to the old level-0 value).  cctk_time is
     the newest data time of the finest level; last_integ_time is the
     previous firing's cctk_time.  The it==0 fallback (sstep<=0) uses the
     level-0 dt; the seed block returns before integration so the value is
     irrelevant there. */
  if (step == 0.0)
  {
    gstate.sstep = cctk_time - gstate.last_integ_time;
    gstate.last_integ_time = cctk_time;
    if (gstate.sstep <= 0.0)
      gstate.sstep = gstate.LD[0][0].dt;
  }
  else
    gstate.sstep = step;

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
      if (!gstate.LD[0][lv].valid || gstate.LD[0][lv].metric_static)
        continue;
      if (gstate.LD[0][lv].dt < dt_min)
        dt_min = gstate.LD[0][lv].dt;
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
    gstate.pgeodesic[mi].old_level_bad = 0;
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
  /* Recovery: when a checkpoint is restored, the particle positions +
     4-velocities (the particle_* arrays, checkpointable per
     interface.ccl) are already the restored, normalized state.  The
     one-shot init block must not run: it re-normalizes against the
     parfile initial state and, worse, queries LevelData that (for
     per-hole AMR) is only populated on the fine levels Carpet acquires
     this iteration -- level 0 is never acquired, so pick_level /
     in_level_dir see zero geometry and every particle is dropped.
     In a fresh run the first trigger fires at cctk_iteration == 0
     (no-op); on a recovered run cctk_iteration is the checkpoint
     iteration (> 0) on the first trigger. */
  if (cctk_iteration > 0 && !gstate.init_done)
    gstate.init_done = 1;
  if (!gstate.init_done)
  {
    if (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0)
      CCTK_VInfo(CCTK_THORNSTRING, "Kerr M:%.4g a:%.4g Particle total:%d", gstate.ksm, gstate.ksa, particle_n_total);
    if (CCTK_IsFunctionAliased("MultiPatch_GetMaps"))
      gstate.nmaps = MultiPatch_GetMaps(cctkGH);
    else
      gstate.nmaps = 1;
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
         time-blend paths; the l27 level-0 path ignores bg and is
         unaffected). */
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
      else if (gstate.sinterp_mode == 0)
      {
        /* local27: full 4-velocity renormalization -- rescale all four
           components by 1/sqrt(-norm) so that g(u,u) = -1 holds exactly
           (the parfile direction is kept; u^t must stay future-directed). */
        if (isfinite(norm) && norm < 0.0 && u[0] > 0.0)
        {
          const double inv_n = 1.0 / sqrt(-norm);
          CCTK_VInfo(CCTK_THORNSTRING,
                     "Particle %d init: u^mu rescaled by %.15g "
                     "(u^t %.15g -> %.15g; norm before correction = %.9g) "
                     "at(%.4g,%.4g,%.4g)",
                     mi, inv_n, u[0], u[0]*inv_n, norm, ys[1], ys[2], ys[3]);
          for (int a = 0; a < 4; a++) u[a] *= inv_n;
          ys[4] = u[0]; ys[5] = u[1]; ys[6] = u[2]; ys[7] = u[3];
          particle_ttv[mi] = u[0];
        }
        else
        {
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Particle %d: 4-velocity not normalizable at init "
                     "(norm=%.4g, u^mu=[%g,%g,%g,%g]); keeping parfile values",
                     mi, norm, u[0], u[1], u[2], u[3]);
        }
      }
      else
      {
        /* aei / carpet: re-solve u^0 with u^i fixed (legacy behavior; see
           the step-start branch for the rationale). */
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
      geodesic_dump_header(gstate.dfp);
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
          const double *bo = bd + (size_t)r * nlev * 11 + (size_t)l * 11;
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
            const double tc = t_now + margin * t_hit;
            if (tc - t_now >= 1e-11 * fmax(1.0, fabs(t_target - t_now)) && tc < t_chunk)
              t_chunk = tc;
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

        /* ---- integrate the chunk (each rank: its owned particles) ----
           aei/carpet is thread-safe here: the pointwise AEI interpolation
           uses only per-call local buffers, and the implicit OpenMP
           barrier below keeps every owned particle done before the
           chunk-boundary collective sync */
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
       (aei/carpet included: the pointwise AEI interpolation is
       thread-safe; the implicit OpenMP barrier precedes the post-loop
       S2 sync) */
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

  /* ---------------- particle-state dump (post-integration) ----------------
     Fires on the finest-level call for which the trigger fired this
     iteration (last_integ_iter == cctk_iteration), i.e. the single
     call per iteration that integrates.  At iteration 0 the one-shot t=0
     seed (above) already wrote a block and set last_dump_iter, so the
     post-integration check (last_dump_iter != cctk_iteration) skips the
     double-write; the first post-integration block is the first real
     integration that is a multiple of particle_dump_every.  After the S2
     sync all ranks see the identical final state, so the rank-0-only dump
     writes the coherent post-integration block. */
  if (lvl == gstate.nlev_seen - 1 &&
      gstate.last_integ_iter == cctk_iteration &&
      particle_dump_every > 0 &&
      (cctk_iteration % particle_dump_every == 0) &&
      gstate.last_dump_iter != cctk_iteration &&
      (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
  {
    if (gstate.dfp == NULL)
    {
      geodesic_mkdir_parents(particle_dump_file);
      gstate.dfp = fopen(particle_dump_file, "w");
      if (gstate.dfp)
        geodesic_dump_header(gstate.dfp);
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
    gstate.last_dump_iter = cctk_iteration;
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
    /* rl (radial-light-signal / ray) range is reported once per coarse step,
       at the same level-0 POSTSTEP moments as the dump/integration.  This is
       independent of the AMR regrid schedule; rl_have is 1 only after a
       successful integration, so no line is printed without real data. */
    if (gstate.srverbose && rl_have && (!gstate.multi_amr || CCTK_MyProc(cctkGH) == 0))
      CCTK_VInfo(CCTK_THORNSTRING, "rl: min:%18.15G max:%18.15G", rl_min, rl_max);
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


