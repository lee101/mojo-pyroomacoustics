"""Benchmark mojo-pyroomacoustics against upstream pyroomacoustics.

Every case runs the same call through both implementations and reports the ratio
(upstream time / port time), so a number above 1.0 means the port is faster.
Run under `pixi run bench`, which holds a machine-wide flock; other jobs share
the machine and would otherwise distort the timings.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import numpy as np

import mojopyroomacoustics as mpa
import pyroomacoustics as pra
from pyroomacoustics import beamforming as pra_bf
from pyroomacoustics.simulation import ism as pra_ism


# --------------------------------------------------------------------------
# timing


def timeit(fn, min_time=0.05, min_reps=3, max_time=1.0):
    """Best-of timing: run `fn` until `min_time` seconds have elapsed.

    Returns (best single-call seconds, calls per second). `max_time` bounds the
    loop so a single slow case cannot stall the whole suite; when the bound is
    hit the best time seen so far is still a real observation, so it is returned
    rather than discarded.
    """
    fn()  # warm up
    reps = 0
    best = math.inf
    elapsed = 0.0
    while elapsed < min_time or reps < min_reps:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        elapsed += dt
        reps += 1
        if elapsed >= max_time:
            break
    return best, 1.0 / best


# --------------------------------------------------------------------------
# fixtures at the sizes real callers pass


def image_sources(n_img, dim=3, seed=0):
    """A plausible image-source set: n_img positions spread over a shoebox."""
    rng = np.random.default_rng(seed)
    imgs = rng.uniform(-4.0, 4.0, size=(dim, n_img))
    # keep one arrival at the origin-ish so the direct path is short
    imgs[:, 0] = 0.0
    return imgs


def band_damping(n_img, n_bands=1, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.9, size=(n_bands, n_img))


def speech(n, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    return (
        np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)
        + 0.4 * rng.standard_normal(n)
    )


def band_limited_noise(n, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    # a low-pass shaped signal keeps the correlation meaningfully non-zero
    f = np.fft.rfftfreq(n, 1 / 16000.0)
    x = np.fft.irfft(np.fft.rfft(x) * (1.0 / (1.0 + (f / 3000.0) ** 4)), n=n)
    x = x / np.max(np.abs(x))
    return x


# --------------------------------------------------------------------------
# benchmark cases
#
# Each case: name, inputs factory, port call, upstream call, and an optional
# correctness check. Sizes are chosen from what the libraries' own tests and
# examples use, not to flatter any implementation.


CASES = []


def case(name, size, port, up, check=None, min_time=0.05):
    CASES.append(
        {
            "name": name,
            "size": size,
            "port": port,
            "up": up,
            "check": check,
            "min_time": min_time,
        }
    )


def _rel(ours, theirs):
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    scale = np.max(np.abs(theirs))
    if scale == 0:
        return float(np.max(np.abs(ours)))
    return float(np.max(np.abs(ours - theirs)) / scale)


# ------------------------------------------------------------ utilities


def case_utilities():
    # autocorr, time domain
    for n, p in ((4096, 256), (16384, 1024)):
        x = band_limited_noise(n)
        case(
            f"autocorr (n={n}, p={p})",
            f"{n} samples",
            lambda x=x, p=p: mpa.autocorr(x, p, method="time"),
            lambda x=x, p=p: pra.utilities.autocorr(x, p, method="time"),
            check=lambda a, b: _rel(a, b),
        )

    # lpc
    for n, p in ((4096, 16), (16384, 64)):
        x = speech(n)
        case(
            f"lpc (n={n}, order={p})",
            f"{n} samples",
            lambda x=x, p=p: mpa.lpc(x, p),
            lambda x=x, p=p: pra.utilities.lpc(x, p),
            check=lambda a, b: _rel(a, b),
        )

    # levinson
    for n, p in ((256, 64), (1024, 256)):
        rng = np.random.default_rng(7)
        r = band_limited_noise(n)[: p + 1] * 0.5
        r[0] = 1.0
        b = rng.standard_normal(p)
        case(
            f"levinson (p={p})",
            f"p={p}",
            lambda r=r, b=b, p=p: mpa.levinson(r, b),
            lambda r=r, b=b, p=p: pra.utilities.levinson(r, b),
            check=lambda a, b: _rel(a, b),
        )

    # low_pass_dirac
    for n_rows, n in ((64, 512), (1024, 1024)):
        rng = np.random.default_rng(11)
        t0 = rng.uniform(0, 1, n_rows)
        alpha = rng.uniform(0.5, 1.0, n_rows)
        case(
            f"low_pass_dirac ({n_rows} x {n})",
            f"{n_rows}x{n}",
            lambda t0=t0, a=alpha, n=n, fs=16000: mpa.low_pass_dirac(t0, a, fs, n),
            lambda t0=t0, a=alpha, n=n, fs=16000: pra.utilities.low_pass_dirac(t0, a, fs, n),
            check=lambda a, b: _rel(a, b),
        )

    # fractional_delay_filter_bank
    for n_delays, span in ((256, 8), (4096, 64)):
        rng = np.random.default_rng(13)
        d = np.sort(rng.uniform(0, span, n_delays))
        case(
            f"fractional_delay_filter_bank ({n_delays})",
            f"{n_delays} delays",
            lambda d=d: mpa.fractional_delay_filter_bank(d),
            lambda d=d: pra.utilities.fractional_delay_filter_bank(d),
            check=lambda a, b: _rel(a, b),
        )

    # goertzel
    for n, k in ((16000, 1000), (64000, 4000)):
        x = speech(n)
        case(
            f"goertzel (n={n}, k={k})",
            f"{n} samples",
            lambda x=x, k=k: mpa.goertzel(x, k),
            lambda x=x, k=k: pra.utilities.goertzel(x, k),
            check=lambda a, b: abs(a - b) / (abs(b) + 1e-30),
        )

    # convmtx (dense Toeplitz)
    for n, h_len in ((128, 16), (512, 64)):
        rng = np.random.default_rng(17)
        h = rng.standard_normal(h_len)
        case(
            f"convmtx (n={n}, len={h_len})",
            f"{n}x{h_len}",
            lambda h=h, n=n: mpa.convmtx(h, n),
            lambda h=h, n=n: pra.utilities.convmtx(h, n),
            check=lambda a, b: _rel(a, b),
        )


# ------------------------------------------------------------ RIR / ISM


def case_rir():
    fs = 16000
    for n_img in (100, 1000, 20000):
        imgs = image_sources(n_img)
        damp = band_damping(n_img, 1)

        class _Src:
            pass

        src = _Src()
        src.images = imgs
        src.damping = damp
        src.fs = fs
        mic = np.array([0.1, 0.2, 1.2])

        case(
            f"compute_ism_rir ({n_img} images)",
            f"{n_img} images",
            lambda src=src, mic=mic, fs=fs: mpa.compute_ism_rir(src, mic, fs=fs),
            lambda src=src, mic=mic, fs=fs: pra_ism.compute_ism_rir(src, mic, fs=fs),
            check=lambda a, b: _rel(a, b),
        )

    # 9 octave bands, upstream's Air Absorption default
    for n_img in (100, 5000):
        imgs = image_sources(n_img)
        damp = band_damping(n_img, 9)

        class _Src:
            pass

        src = _Src()
        src.images = imgs
        src.damping = damp
        src.fs = 16000
        mic = np.array([0.1, 0.2, 1.2])
        case(
            f"compute_ism_rir 9-band ({n_img} images)",
            f"{n_img} images x 9 bands",
            lambda src=src, mic=mic: mpa.compute_ism_rir(src, mic, fs=16000),
            lambda src=src, mic=mic: pra_ism.compute_ism_rir(src, mic, fs=16000),
            check=lambda a, b: _rel(a, b),
        )

    # delay_sum
    for n_irs, n_taps in ((256, 256), (4096, 512)):
        rng = np.random.default_rng(19)
        irs = rng.standard_normal((n_irs, n_taps)) * 0.1
        delays = np.sort(rng.integers(0, n_irs, n_irs) * 4).astype(np.float64)

        def port(irs=irs, delays=delays):
            out = np.zeros(int(delays.max()) + irs.shape[1])
            return mpa.delay_sum(irs, delays, output=out)

        def up(irs=irs, delays=delays):
            return pra.room_sim.delay_sum(irs, delays.astype(np.int32))

        case(
            f"delay_sum ({n_irs} x {n_taps})",
            f"{n_irs}x{n_taps}",
            port,
            up,
            check=lambda a, b: _rel(a, b),
        )

    # fractional_delay filters
    for n_times in (256, 8192):
        rng = np.random.default_rng(23)
        t = rng.uniform(0, 1, n_times)

        case(
            f"fractional_delay filters ({n_times})",
            f"{n_times} arrivals",
            lambda t=t: mpa.rir.fractional_delay(t),
            lambda t=t: pra.room_sim.fractional_delay(t),
            check=lambda a, b: _rel(a, b),
        )


# ------------------------------------------------------------ beamforming


def case_beamforming():
    # distance matrix
    for dim, nx, ny in ((2, 64, 64), (2, 512, 512), (3, 128, 128)):
        rng = np.random.default_rng(29)
        x = rng.uniform(-1, 1, size=(dim, nx))
        y = rng.uniform(-1, 1, size=(dim, ny))
        case(
            f"distance ({dim}D, {nx}x{ny})",
            f"{dim}D {nx}x{ny}",
            lambda x=x, y=y: mpa.distance(x, y),
            lambda x=x, y=y: pra_bf.distance(x, y),
            check=lambda a, b: _rel(a, b),
        )

    # steering vectors
    for M in (8, 64, 256):
        R = pra_bf.circular_2D_array(np.zeros(2), M, 0, 0.04)
        bfm = mpa.Beamformer(R, 16000)
        bfp = pra_bf.Beamformer(R, 16000)
        phis = np.linspace(-np.pi, np.pi, 128)
        case(
            f"steering_vector_2D (M={M}, 128 dirs)",
            f"M={M}",
            lambda bfm=bfm, p=phis: bfm.steering_vector_2D(1000.0, p, 10.0),
            lambda bfp=bfp, p=phis: bfp.steering_vector_2D(1000.0, p, 10.0),
            check=lambda a, b: _rel(a, b),
        )
        X = np.stack([np.linspace(1, 4, 32), np.linspace(-2, 2, 32)])
        case(
            f"steering_vector_2D_from_point (M={M}, 32 src)",
            f"M={M}",
            lambda bfm=bfm, X=X: bfm.steering_vector_2D_from_point(1000.0, X),
            lambda bfp=bfp, X=X: bfp.steering_vector_2D_from_point(1000.0, X),
            check=lambda a, b: _rel(a, b),
        )

    # H
    for rows, cols in ((32, 32), (256, 128)):
        rng = np.random.default_rng(31)
        A = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
        case(
            f"H ({rows}x{cols})",
            f"{rows}x{cols}",
            lambda A=A: mpa.H(A),
            lambda A=A: pra_bf.H(A),
            check=lambda a, b: _rel(a, b),
        )

    # mdot3
    for n, m, k, l in ((32, 32, 32, 32), (128, 128, 128, 128)):
        rng = np.random.default_rng(37)
        A = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
        B = rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))
        C = rng.standard_normal((k, l)) + 1j * rng.standard_normal((k, l))
        case(
            f"mdot3 ({n}x{m} {m}x{k} {k}x{l})",
            f"{n}x{m}x{k}x{l}",
            lambda A=A, B=B, C=C: mpa.mdot3(A, B, C),
            lambda A=A, B=B, C=C: pra_bf.mdot3(A, B, C) if hasattr(pra_bf, "mdot3")
            else pra_bf.mdot(pra_bf.H(A), B, C),
            check=lambda a, b: _rel(a, b),
        )

    # far_field_weights + response, the full 1-frequency design loop
    for M, n_freq in ((8, 513), (32, 513), (64, 1025)):
        R = pra_bf.circular_2D_array(np.zeros(2), M, 0, 0.04)
        N = n_freq - 1 if n_freq > 513 else 512
        bfm = mpa.Beamformer(R, 16000, N=N)
        bfp = pra_bf.Beamformer(R, 16000, N=N)
        case(
            f"far_field_weights (M={M}, {n_freq} bins)",
            f"M={M}",
            lambda bfm=bfm, p=np.pi / 2: bfm.far_field_weights(p),
            lambda bfp=bfp, p=np.pi / 2: bfp.far_field_weights(p),
            check=lambda a, b: _rel(np.asarray(a), np.asarray(b)),
        )
        phis = np.linspace(-np.pi, np.pi, 360)
        bfm.far_field_weights(np.pi / 2)
        bfp.far_field_weights(np.pi / 2)
        case(
            f"response (M={M}, 360 dirs)",
            f"M={M}",
            lambda bfm=bfm, p=phis: bfm.response(p, 1000.0),
            lambda bfp=bfp, p=phis: bfp.response(p, 1000.0),
            check=lambda a, b: _rel(np.asarray(a), np.asarray(b)),
        )


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    case_utilities()
    case_rir()
    case_beamforming()

    rows = []
    for c in CASES:
        if args.filter and args.filter not in c["name"]:
            continue
        err = None
        try:
            a = c["port"]()
            b = c["up"]()
            if c["check"] is not None:
                err = c["check"](a, b)
        except Exception as exc:  # noqa: BLE001
            print(f"  {c['name']}: FAILED {type(exc).__name__}: {exc}")
            rows.append({"name": c["name"], "error": str(exc)})
            continue

        p_best = min(
            timeit(c["port"], c["min_time"])[0] for _ in range(args.repeat)
        )
        u_best = min(
            timeit(c["up"], c["min_time"])[0] for _ in range(args.repeat)
        )
        rows.append(
            {
                "name": c["name"],
                "size": c["size"],
                "port_s": p_best,
                "up_s": u_best,
                "speedup": u_best / p_best,
                "port_hz": 1.0 / p_best,
                "up_hz": 1.0 / u_best,
                "rel_err": err,
            }
        )

    hdr = f"{'benchmark':<44} {'port':>12} {'upstream':>12} {'speedup':>9}  err"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<44} {'--':>12} {'--':>12} {'--':>9}  {r['error'][:30]}")
            continue
        err = "-" if r["rel_err"] is None else f"{r['rel_err']:.2e}"
        print(
            f"{r['name']:<44} {r['port_s'] * 1e6:>10.2f}us {r['up_s'] * 1e6:>10.2f}us "
            f"{r['speedup']:>8.2f}x  {err}"
        )

    ok = [r for r in rows if "error" not in r]
    if ok:
        sp = [r["speedup"] for r in ok]
        print("-" * len(hdr))
        print(
            f"{'geomean':<44} {'':>12} {'':>12} "
            f"{math.exp(statistics.fmean(math.log(s) for s in sp)):>8.2f}x"
        )
        worst = sorted(ok, key=lambda r: r["speedup"])[:8]
        print("slowest relative to upstream:")
        for r in worst:
            print(f"  {r['name']:<44} {r['speedup']:>8.2f}x")
        worst_err = max(
            (r["rel_err"] for r in ok if r["rel_err"] is not None), default=0.0
        )
        print(f"worst relative error vs upstream: {worst_err:.3e}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
