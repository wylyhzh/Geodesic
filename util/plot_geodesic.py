#!/usr/bin/env python3
"""plot_geodesic.py -- plot geodesic-particle trajectories from a Geodesic
thorn dump file.

Dump format (plain text, one block per dump, appended over time):
    # t x y z  u^t u^x u^y u^z  tau      <- header line(s), skipped
    then npar rows per block; a dropped particle's row keeps the block
    time in t/tau and has NaN in x, y, z (and the velocities).

The number of particles per block is inferred from the data: all rows of
one block share the same t, so consecutive rows with equal t form a
block.  Override with --npar if needed.

Requirements: numpy, matplotlib.

Usage:
    python3 plot_geodesic.py [DATAFILE [BASE]] [options]

    DATAFILE   particle dump (default: geodesic_particles.txt in the
               current directory, else the first geodesic_particles*.txt)
    BASE       output base name (default: DATAFILE without .txt);
               figures are written to BASE.png / BASE_3d.png in
               dirname(BASE)

Options:
    --2d | --3d | --both   which figure(s) to write (default --2d)
    --plane xy|xz|yz       2D projection (default xy)
    --npar N               particles per block (default: inferred)
    --dpi N                PNG resolution (default 150)
    --pdf                  also write a PDF of each figure
    --no-arrows            omit the end-velocity arrows
    -h, --help             this help
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMAP = "viridis"
ARROW_L = 0.35  # end-velocity arrow length, in data units

PLANES = {
    "xy": (1, 2),
    "xz": (1, 3),
    "yz": (2, 3),
}
PLANE_LABELS = {
    "xy": ("x", "y"),
    "xz": ("x", "z"),
    "yz": ("y", "z"),
}


def usage():
    print(__doc__.strip())


def pick_default_data():
    if os.path.isfile("geodesic_particles.txt"):
        return "geodesic_particles.txt"
    cands = sorted(f for f in os.listdir(".")
                   if f.startswith("geodesic_particles") and f.endswith(".txt")
                   and os.path.isfile(f))
    if cands:
        return cands[0]
    return None


def load_rows(path):
    d = np.genfromtxt(path, comments="#")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[1] not in (9, 10):
        raise ValueError("expected 9 columns (t x y z ut ux uy uz tau) or 10 "
                         "with the cctk_time column appended, got %d" % d.shape[1])
    return d


def _candidate_ok(d, ts, tol=1e-6):
    """Does a block size d partition the file into physically consistent
    blocks?  A block's time is the median of its rows' t (robust to the
    frozen/dropped rows, which carry an earlier drop time).  Within a
    block the majority of rows must sit at the block time, no row may lie
    in the future of it, and block times must strictly advance."""
    nblocks = len(ts) // d
    if nblocks < 2:
        return False
    bt_prev = None
    for b in range(nblocks):
        seg = ts[b * d:(b + 1) * d]
        med = float(np.median(seg))
        close = np.abs(seg - med) <= tol * max(1.0, abs(med))
        if close.mean() < 0.51:
            return False
        if seg.max() > med + tol * max(1.0, abs(med)):
            return False
        if bt_prev is not None and med <= bt_prev + tol * max(1.0, abs(bt_prev)):
            return False
        bt_prev = med
    return True


def detect_npar(d):
    """Infer the number of particles per block.

    Each dump block has exactly npar rows (particle index order).  Live
    rows carry the block time up to the floating-point scatter of the
    per-particle tau integration; frozen/dropped rows carry an earlier
    drop time.  So the block time is the per-block median of the t column,
    and the true npar is the largest row count that yields physically
    consistent blocks (see _candidate_ok)."""
    ts = d[:, 0]
    R = len(ts)
    cands = [x for x in range(2, R + 1) if R % x == 0]
    for x in sorted(cands, reverse=True):
        if _candidate_ok(x, ts):
            return x
    return 1  # single-particle dump: one row per block


def load_trajectory(path, npar=None):
    """Return (tvals, snaps, npar).  snaps[k] maps particle index mi to its
    row if alive at block k; tvals[k] is that block's time (median)."""
    d = load_rows(path)
    if npar is None:
        npar = detect_npar(d)
    if len(d) % npar != 0:
        nblocks = len(d) // npar
        print("warning: %d rows do not divide evenly by npar=%d; "
              "ignoring %d trailing rows"
              % (len(d), npar, len(d) % npar))
        d = d[:nblocks * npar]
    else:
        nblocks = len(d) // npar
    if nblocks == 0:
        raise ValueError("no data rows in %s" % path)
    tvals, snaps = [], []
    for k in range(nblocks):
        blk = d[k * npar:(k + 1) * npar]
        alive = [i for i in range(npar)
                 if np.isfinite(blk[i, 1]) and np.isfinite(blk[i, 2])
                 and np.isfinite(blk[i, 3])]
        if not alive:
            continue
        tvals.append(float(np.median(blk[:, 0])))
        snaps.append({mi: blk[mi] for mi in alive})
    if not snaps:
        raise ValueError("no alive particles found in %s" % path)
    return tvals, snaps, npar


def cmap_of(idx, n):
    cmap = plt.get_cmap(CMAP)
    return cmap(0.0 if n <= 1 else idx / (n - 1))


def collect_all(snaps, col):
    return np.array([snaps[k][mi][col] for k in range(len(snaps))
                     for mi in snaps[k]], dtype=float)


def make_2d(tvals, snaps, npar, base, plane, datafile, dpi, arrows, pdf):
    c1, c2 = PLANES[plane]
    lx, ly = PLANE_LABELS[plane]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    s0 = snaps[0]
    ax.scatter([s0[mi][c1] for mi in s0], [s0[mi][c2] for mi in s0],
               s=18, color="tab:blue", alpha=0.35, marker="o",
               label="initial positions (t=0)", zorder=4)

    last = snaps[-1]
    for mi in range(npar):
        ps = [k for k, s in enumerate(snaps) if mi in s]
        if len(ps) < 2:
            continue
        col = cmap_of(mi, npar)
        xs = [snaps[k][mi][c1] for k in ps]
        ys = [snaps[k][mi][c2] for k in ps]
        ax.plot(xs, ys, "-", color=col, lw=0.7, alpha=0.8)
        xend, yend = xs[-1], ys[-1]
        ax.plot(xend, yend, "o", color=col, mfc="none", ms=6, mew=0.9,
                zorder=3)
        if arrows:
            ux = snaps[ps[-1]][mi][c1 + 4]
            uy = snaps[ps[-1]][mi][c2 + 4]
            vn = np.hypot(ux, uy)
            if vn > 1e-12:
                ax.plot([xend, xend + ARROW_L * ux / vn],
                        [yend, yend + ARROW_L * uy / vn], "-", color=col,
                        lw=1.6, solid_capstyle="round", zorder=2)
        if mi not in last:  # dropped: mark last alive position
            ax.plot(xend, yend, "X", color="k", ms=9, mew=1.2, zorder=5)

    xall = collect_all(snaps, c1)
    yall = collect_all(snaps, c2)
    lo = min(xall.min(), yall.min()) - 1.0
    hi = max(xall.max(), yall.max()) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(lx, fontsize=12)
    ax.set_ylabel(ly, fontsize=12)
    n_dropped = sum(1 for mi in range(npar) if mi not in last)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, npar - 1), cmap=CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="particle index")
    title = ("Geodesic trajectories (%s projection): $t$ = %.4f--%.4f\n"
             "%d initial particles, %d dumps, %d dropped"
             % (plane, tvals[0], tvals[-1], npar, len(snaps), n_dropped))
    if datafile:
        title += "\n" + os.path.basename(datafile)
    ax.set_title(title, fontsize=12)
    outdir = os.path.dirname(base)
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, os.path.basename(base) + ".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(outdir, os.path.basename(base) + ".pdf"),
                    bbox_inches="tight")
    plt.close(fig)
    return out_png


def make_3d(tvals, snaps, npar, base, datafile, dpi, arrows, pdf):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    last = snaps[-1]
    for mi in range(npar):
        ps = [k for k, s in enumerate(snaps) if mi in s]
        if not ps:
            continue
        col = cmap_of(mi, npar)
        x = [snaps[k][mi][1] for k in ps]
        y = [snaps[k][mi][2] for k in ps]
        z = [snaps[k][mi][3] for k in ps]
        ax.plot(x, y, z, "-", color=col, lw=0.9, alpha=0.6)
        ax.scatter(x, y, z, color=col, s=8, alpha=0.9)
        if arrows:
            xe, ye, ze = x[-1], y[-1], z[-1]
            ux = snaps[ps[-1]][mi][5]
            uy = snaps[ps[-1]][mi][6]
            uz = snaps[ps[-1]][mi][7]
            vn = np.sqrt(ux * ux + uy * uy + uz * uz)
            if vn > 1e-12:
                ax.plot([xe, xe + ARROW_L * ux / vn],
                        [ye, ye + ARROW_L * uy / vn],
                        [ze, ze + ARROW_L * uz / vn], "-", color=col, lw=1.4)
        if mi not in last:  # dropped: mark last alive position
            ax.scatter(x, y, z, color="k", marker="X", s=25,
                       edgecolors="k", zorder=5)
    xall = collect_all(snaps, 1)
    yall = collect_all(snaps, 2)
    zall = collect_all(snaps, 3)
    lo = min(xall.min(), yall.min(), zall.min()) - 1.0
    hi = max(xall.max(), yall.max(), zall.max()) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    n_dropped = sum(1 for mi in range(npar) if mi not in last)
    title = ("3D geodesic trajectories: $t$ = %.4f--%.4f\n"
             "%d initial particles, %d dumps, %d dropped"
             % (tvals[0], tvals[-1], npar, len(snaps), n_dropped))
    if datafile:
        title += "\n" + os.path.basename(datafile)
    ax.set_title(title, fontsize=12)
    outdir = os.path.dirname(base)
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, os.path.basename(base) + "_3d.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(outdir, os.path.basename(base) + "_3d.pdf"),
                    bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    args = sys.argv[1:]
    do2d = do3d = False
    npar = None
    plane = "xy"
    dpi = 150
    pdf = False
    arrows = True
    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--2d", "2d"):
            do2d, do3d = True, False
        elif a in ("--3d", "3d"):
            do2d, do3d = False, True
        elif a == "--both":
            do2d = do3d = True
        elif a == "--pdf":
            pdf = True
        elif a == "--no-arrows":
            arrows = False
        elif a == "--plane":
            i += 1
            if i >= len(args) or args[i] not in PLANES:
                sys.exit("error: --plane needs one of xy, xz, yz")
            plane = args[i]
        elif a == "--npar":
            i += 1
            if i >= len(args):
                sys.exit("error: --npar needs a number")
            npar = int(args[i])
        elif a == "--dpi":
            i += 1
            if i >= len(args):
                sys.exit("error: --dpi needs a number")
            dpi = int(args[i])
        elif a in ("-h", "--help"):
            usage()
            return
        else:
            pos.append(a)
        i += 1
    if not do2d and not do3d:
        do2d = True
    if len(pos) > 2:
        sys.exit("error: at most two positional arguments (DATAFILE BASE)")
    data = pos[0] if pos else pick_default_data()
    if data is None:
        sys.exit("error: no data file given and no geodesic_particles*.txt "
                 "in the current directory")
    data = os.path.abspath(data)
    if not os.path.isfile(data):
        sys.exit("error: data file not found: %s" % data)
    base = pos[1] if len(pos) > 1 else os.path.splitext(data)[0]
    base = os.path.abspath(base)

    tvals, snaps, npar_eff = load_trajectory(data, npar)
    print("data           : %s" % data)
    print("npar           : %d%s" % (npar_eff,
                                     "" if npar else " (inferred)"))
    print("snapshots      : %d   t = %.4f .. %.4f"
          % (len(snaps), tvals[0], tvals[-1]))
    for k, s in enumerate(snaps):
        print("  dump%3d t=%.6f n_alive=%d" % (k, tvals[k], len(s)))
    if do2d:
        out = make_2d(tvals, snaps, npar_eff, base, plane, data,
                      dpi, arrows, pdf)
        print("2D saved: %s" % out)
    if do3d:
        out = make_3d(tvals, snaps, npar_eff, base, data, dpi, arrows, pdf)
        print("3D saved: %s" % out)


if __name__ == "__main__":
    main()
