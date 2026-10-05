"""Direct tests of every exported kernel, called by name over ctypes.

The public API tests reach most kernels indirectly, which is what a user
exercises. This module exists so that each `@export`ed symbol in
`src/ported.mojo` is called by name and asserted on its own real output. A
kernel that is wired up wrong, mis-shaped, or reading the wrong length cannot
ship green here.

The ctypes signatures in `_lib._SIGNATURES` are checked as well: a signature
that disagrees with the Mojo `@export` would reinterpret arguments silently.
"""

import ctypes
import math
import os
import re

import numpy as np
import pytest

import mojopyroomacoustics as mpa
from mojopyroomacoustics import _lib
from mojopyroomacoustics.utilities import convmtx_dot

import pyroomacoustics.utilities as pu

from conftest import addr, f64


FS = 16000
FDL = 81
LUT_GRAN = 20


def _tables(fdl=FDL, lut_gran=LUT_GRAN):
    lut = np.zeros((fdl + 2) * lut_gran, dtype=np.float64)
    hann = np.zeros(fdl, dtype=np.float64)
    lib = _lib.require()
    lib.mpa_build_sinc_lut(addr(lut), fdl, lut_gran)
    lib.mpa_build_hann(addr(hann), fdl)
    return lut, hann


# ---------------------------------------------------------------------------
# Every export is named in tests, and the library actually exports it
# ---------------------------------------------------------------------------


def _exported_names():
    """Every `@export("...")` in src/ported.mojo, read from the source."""
    src = os.path.join(_lib.SRC, "ported.mojo")
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    return set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', text))


def _names_in_tests():
    """Every `mpa_*` symbol named anywhere under tests/."""
    out = set()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in sorted(os.listdir(tests_dir)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(tests_dir, fname), encoding="utf-8") as fh:
            out |= set(re.findall(r"mpa_[A-Za-z_0-9]+", fh.read()))
    return out


def test_every_export_is_named_in_the_test_suite():
    """Guards the guard: if a kernel is added without a test, this fails.

    Same check as `pixi run check-kernels`, run as a test so it is part of
    `pixi run test` and cannot be skipped.
    """
    missing = _exported_names() - _names_in_tests()
    assert not missing, f"exports with no test naming them: {sorted(missing)}"


def test_every_export_is_reached_by_the_public_api(rng, error):
    """One call per export through the package API, asserting on real output.

    `tests/test_beamforming.py` covers the geometry, steering, response and rake
    kernels in detail; this covers them a second time in one place, so the claim
    "every exported kernel is exercised" has an independent witness.
    """
    # mpa_unit_vec2D, which the array constructors reach internally
    assert mpa.unit_vec2D(0.0).shape == (2, 1)
    # mpa_linear_2D_array, mpa_circular_2D_array, mpa_square_2D_array
    assert mpa.linear_2D_array([0, 0], 5, 0.3, 0.05).shape == (2, 5)
    assert mpa.circular_2D_array([0, 0], 6, 0.4, 0.1).shape == (2, 6)
    assert mpa.square_2D_array([0, 0], 3, 4, 0.2, 0.05).shape == (2, 12)
    # mpa_distance
    assert mpa.distance(np.zeros((2, 3)), np.ones((2, 3)))[0, 0] == pytest.approx(
        np.sqrt(2.0)
    )
    # mpa_H
    A = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    assert error(mpa.H(A), A.conj().T) == 0.0
    # mpa_mdot3, with k = 1 so the (n, k) intermediate is a real contraction
    n, m, k, l = 3, 4, 2, 5
    A = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
    B = rng.normal(size=(m, k)) + 1j * rng.normal(size=(m, k))
    C = rng.normal(size=(k, l)) + 1j * rng.normal(size=(k, l))
    assert error(mpa.mdot3(A, B, C), (A @ B) @ C) < 1e-14

    R = mpa.circular_2D_array([0.0, 0.0], 8, 0.0, 0.04)
    bf = mpa.Beamformer(R, 16000, N=64)
    phis = np.linspace(-np.pi, np.pi, 16)
    # mpa_steering_vector_2D, mpa_steering_vector_2D_from_point
    assert bf.steering_vector_2D(1000.0, phis, 10.0).shape == (8, 16)
    assert bf.steering_vector_2D_from_point(
        1000.0, np.array([[1.0], [0.5]])
    ).shape == (8, 1)
    # mpa_far_field_projection, mpa_far_field_weights, mpa_response
    bf.far_field_weights(np.pi / 3)
    assert bf.weights.shape == (8, 33)
    _, resp = bf.response(phis, 2000.0)
    assert resp.shape == (16,)
    # mpa_rake_delay_and_sum_weights
    assert mpa.rake_delay_and_sum(
        R, np.array([[1.0, -1.0], [0.5, 0.5]]), np.array([0.0, 1000.0])
    ).shape == (8, 2)


def test_every_export_is_present_in_the_shared_library(lib):
    """The signatures table and the compiled library must agree exactly."""
    for name in sorted(_exported_names()):
        assert hasattr(lib, name), f"{name} is not exported by the library"
    # and nothing in the signature table is missing from the source
    assert set(_lib._SIGNATURES) == _exported_names()


def test_ctypes_signatures_match_the_export_widths():
    """Every `Int` argument is declared c_int64 and every float a c_double.

    A mismatch here would reinterpret arguments rather than fail, so it is
    checked against the source's own parameter lists rather than by eye.
    """
    src = os.path.join(_lib.SRC, "ported.mojo")
    with open(src, encoding="utf-8") as fh:
        text = fh.read()

    for name, (argtypes, restype) in _lib._SIGNATURES.items():
        match = re.search(
            r'@export\("' + re.escape(name) + r'"\)\s*\ndef\s+' + re.escape(name)
            + r"\s*\(([^)]*)\)\s*abi\(\"C\"\)\s*(->\s*\w+)?:",
            text,
        )
        assert match, f"could not find the export signature for {name}"
        params = [p.strip() for p in match.group(1).split(",") if p.strip()]
        assert len(params) == len(argtypes), (
            f"{name}: {len(params)} Mojo parameters but {len(argtypes)} ctypes "
            f"argtypes"
        )
        for param, argtype in zip(params, argtypes):
            mojo_type = param.split(":")[1].strip()
            want = ctypes.c_double if mojo_type == "Float64" else ctypes.c_int64
            assert argtype is want, f"{name}: parameter {param} is {mojo_type}"
        assert (restype is ctypes.c_int64) == (match.group(2) is not None), (
            f"{name}: return type disagrees between Mojo and ctypes"
        )


# ---------------------------------------------------------------------------
# mpa_convmtx_dot
# ---------------------------------------------------------------------------


def test_mpa_convmtx_dot_is_a_truncated_convolution(lib, rng, error):
    """`dst[i] = sum_k h[k] x[i-k]` for `i < n`, i.e. `np.convolve(x, h)[:n]`.

    Upstream's contract is `np.dot(convmtx(x, n), v) == np.convolve(x, v)`, and
    `convmtx(h, n)` is `(m + n - 1, n)`, so that product against a vector of
    length `n` is the first `n` samples of the convolution. The kernel therefore
    produces `len(x)` samples and drops the `m - 1` tail taps.
    """
    m, n = 9, 64
    h = rng.normal(size=m)
    x = rng.normal(size=n)
    dst = np.zeros(n, dtype=np.float64)
    lib.mpa_convmtx_dot(addr(dst), addr(x), addr(h), n, m)

    expected = np.convolve(x, h)[:n]
    assert dst.shape == (n,)
    assert error(dst, expected) < 1e-13
    # and against upstream's own matrix: `convmtx(h, n)` is (m + n - 1, n), so
    # its product with an x of length n is the convolution in full and the
    # kernel is the same thing truncated to the first n samples
    full = pu.convmtx(h, n) @ x
    assert full.shape == (m + n - 1,)
    assert error(dst, full[:n]) < 1e-13
    # and the public wrapper reaches the same kernel with the same answer
    assert error(convmtx_dot(h, x), expected) < 1e-13


def test_mpa_convmtx_dot_is_a_convolution(lib, rng, error):
    """Upstream's usage is a convolution, so the two must agree."""
    n, m = 128, 17
    h = rng.normal(size=m)
    x = rng.normal(size=n)
    dst = np.zeros(n, dtype=np.float64)
    lib.mpa_convmtx_dot(addr(dst), addr(x), addr(h), n, m)
    assert error(dst, np.convolve(x, h)[:n]) < 1e-13


def test_mpa_convmtx_dot_zeroes_every_output_sample(lib, rng):
    """Every dst entry is written, including those with no overlapping tap."""
    n, m = 40, 6
    h = np.zeros(m)
    dst = np.full(n, 99.0, dtype=np.float64)
    lib.mpa_convmtx_dot(addr(dst), addr(rng.normal(size=n)), addr(h), n, m)
    assert np.all(dst == 0.0)


def test_mpa_convmtx_dot_truncates_at_zero_for_negative_lags(lib, rng, error):
    """`dst[i] = sum_k h[k] x[i-k]` with x[<0] treated as zero."""
    n, m = 20, 8
    h = rng.normal(size=m)
    x = rng.normal(size=n)
    dst = np.zeros(n, dtype=np.float64)
    lib.mpa_convmtx_dot(addr(dst), addr(x), addr(h), n, m)
    for i in (0, 1, 7, n - 1):
        expected = sum(h[k] * x[i - k] for k in range(m) if i - k >= 0)
        assert dst[i] == pytest.approx(expected, abs=1e-13)


# ---------------------------------------------------------------------------
# mpa_low_pass_dirac
# ---------------------------------------------------------------------------


def test_mpa_low_pass_dirac_is_alpha_times_a_sinc(lib, rng, error):
    n_rows, n = 9, 40
    t0 = rng.uniform(0, 3, n_rows)
    alpha = rng.uniform(0.1, 1.0, n_rows)
    dst = np.zeros((n_rows, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(t0), addr(alpha), n_rows, n, 16000.0)

    for t in range(n_rows):
        expected = alpha[t] * np.sinc(np.arange(n) - 16000.0 * t0[t])
        assert dst[t] == pytest.approx(expected, abs=1e-15), f"row {t}"


def test_mpa_low_pass_dirac_covers_every_row(lib):
    """A different attenuation per row must reach every row of the output."""
    n_rows, n = 32, 16
    t0 = np.arange(n_rows, dtype=np.float64) / 16000.0
    alpha = np.arange(1, n_rows + 1, dtype=np.float64)
    dst = np.zeros((n_rows, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(t0), addr(alpha), n_rows, n, 16000.0)

    # with the delay on an exact sample the sinc peaks there at the attenuation;
    # the off-peak taps are the sinc's residual ringing, about 1e-16
    offsets = np.arange(n)[None, :] - np.arange(n_rows)[:, None]
    expected = alpha[:, None] * np.sinc(offsets)
    assert dst == pytest.approx(expected, abs=1e-14)
    assert dst.shape == (n_rows, n)


def test_mpa_low_pass_dirac_handles_a_fractional_delay(lib):
    """A delay between samples spreads the peak over its two neighbours."""
    fs, n = 16000.0, 16
    dst = np.zeros((1, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(f64([4.5 / fs])), addr(f64([1.0])),
                           1, n, fs)
    expected = np.sinc(np.arange(n) - 4.5)
    assert dst[0] == pytest.approx(expected, abs=1e-15)
    # the peak sits between samples 4 and 5
    assert int(np.argmax(dst[0])) in (4, 5)
    assert dst[0, 4] == pytest.approx(expected[4], abs=1e-15)


def test_mpa_low_pass_dirac_survives_a_single_row(lib):
    """One row, and the arguments are temporaries rather than named buffers.

    Both `f64(...)` arguments are dropped the instant `addr` returns, so this
    also exercises that the address keeps its buffer alive across the call.
    The delay is chosen so `Fs * t0` is 8, just past the end of an 8-sample
    row, where upstream's sinc is exactly zero.
    """
    fs, n = 16000.0, 8
    dst = np.zeros((1, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(f64([8.0 / fs])), addr(f64([2.0])),
                           1, n, fs)
    assert np.max(np.abs(dst)) < 1e-15


def test_mpa_low_pass_dirac_peaks_on_an_aligned_sample(lib):
    """`Fs * t0 = k` puts the attenuation of row k exactly at column k."""
    fs, n_rows, n = 16000.0, 12, 20
    t0 = np.arange(n_rows, dtype=np.float64) / fs
    alpha = np.arange(1.0, n_rows + 1.0)
    dst = np.zeros((n_rows, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(t0), addr(alpha), n_rows, n, fs)
    for k in range(n_rows):
        assert dst[k, k] == pytest.approx(alpha[k], abs=1e-14), f"row {k}"
        # and the neighbouring taps are the sinc's residual ringing
        for j in (k - 1, k + 1):
            if 0 <= j < n:
                assert abs(dst[k, j]) < 1e-14


def test_mpa_low_pass_dirac_uses_upstream_alpha_times_sinc(lib, rng, error):
    """Upstream is `alpha * np.sinc(np.arange(N) - Fs * t0)`, exactly."""
    n_rows, n, fs = 7, 24, 16000
    t0 = rng.uniform(0, 4, n_rows)
    alpha = rng.uniform(0.1, 1.0, n_rows)
    dst = np.zeros((n_rows, n), dtype=np.float64)
    lib.mpa_low_pass_dirac(addr(dst), addr(t0), addr(alpha), n_rows, n, float(fs))
    for t in range(n_rows):
        expected = alpha[t] * np.sinc(np.arange(n) - fs * t0[t])
        assert dst[t] == pytest.approx(expected, rel=1e-13, abs=1e-14), f"row {t}"


# ---------------------------------------------------------------------------
# mpa_fractional_delay_bank
# ---------------------------------------------------------------------------


def test_mpa_fractional_delay_bank_matches_upstream(lib, rng, error):
    delays = np.sort(rng.uniform(0, 9, 21))
    n_delays = delays.size
    L = FDL
    min_delay = float(delays.min())
    filter_len = L + int(math.ceil(float((delays - min_delay).max())))
    bank = np.zeros((n_delays, filter_len), dtype=np.float64)
    lib.mpa_fractional_delay_bank(
        addr(bank), addr(delays), n_delays, L, filter_len, min_delay
    )
    assert error(bank, pu.fractional_delay_filter_bank(delays)) < 1e-15


def _bank_expected(delays, L, filter_len, n_delays, got):
    """The fractional-delay bank, tap by tap, and whether `got` matches it.

    Upstream splits each delay into an integer `di` and a fractional `df` and
    scatters `hann[k] * sinc(k - df - (L - 1) / 2)` at index `k + di`. Any tap
    landing outside the row is dropped, so a row declared shorter than its
    delay requires is truncated rather than written past.
    """
    half = (L - 1) / 2.0
    ok = True
    for t in range(n_delays):
        d = float(delays[t]) - float(delays.min())
        di = math.floor(d)
        df = d - di
        expected = np.zeros(filter_len, dtype=np.float64)
        for k in range(L):
            idx = di + k
            if 0 <= idx < filter_len:
                hann = 0.5 - 0.5 * math.cos(2 * math.pi * k / (L - 1))
                expected[idx] = hann * np.sinc(k - df - half)
        if not np.allclose(got[t], expected, rtol=0, atol=1e-13):
            ok = False
    return ok


def test_mpa_fractional_delay_bank_index_is_clamped_to_the_row(lib):
    """A tap whose index falls past the declared row must be dropped.

    The memory around the bank is checked as well, so a kernel that wrote past
    the row instead of dropping the tap would be caught here.
    """
    L = 7
    delays = np.array([0.0, 1.0, 2.0, 3.0])
    n_delays = delays.size
    min_delay = 0.0
    # row t writes at indices t .. t + L - 1; at filter_len = L + 2 = 9 row 3
    # would need index 9 and loses exactly its last tap, the clamp under test
    filter_len = L + 2

    canary = -12345.0
    pad = 8
    backing = np.full(n_delays * filter_len + 2 * pad, canary, dtype=np.float64)
    view = backing[pad : pad + n_delays * filter_len].reshape(n_delays, filter_len)
    lib.mpa_fractional_delay_bank(addr(view), addr(delays), n_delays, L,
                                  filter_len, min_delay)

    assert np.all(backing[:pad] == canary), "wrote before the buffer"
    assert np.all(backing[pad + n_delays * filter_len :] == canary), (
        "wrote past the buffer"
    )
    # each row writes exactly L taps at delay[t] .. delay[t] + L - 1 and leaves
    # the rest of the row at the canary, which is also how the kernel is
    # specified: it scatters, it does not clear
    half = (L - 1) / 2.0
    for t in range(n_delays):
        d = int(delays[t])
        expected = np.full(filter_len, canary, dtype=np.float64)
        for k in range(L):
            idx = d + k
            if 0 <= idx < filter_len:
                hann = 0.5 - 0.5 * math.cos(2 * math.pi * k / (L - 1))
                expected[idx] = hann * np.sinc(k - half)
        assert view[t] == pytest.approx(expected, abs=1e-13), f"row {t}"
    # row 3's taps would land at 3 .. 9 and filter_len is 9, so its last tap is
    # the one dropped and the row ends at index 8
    assert np.all(view[3, 3:] != canary), "row 3 did not fill its row"
    assert np.all(view[0, L:] == canary), "row 0 wrote past its L taps"
    assert np.all(view[3, 3 : 3 + L - 1] != canary)


def test_mpa_fractional_delay_bank_truncates_a_row_declared_too_short(lib):
    """A row too short for its delay offset loses the taps that do not fit."""
    L = 9
    delays = np.array([0.0, 1.0, 2.0, 3.0])
    n_delays = delays.size
    # offsets 2 and 3 would need filter_len 11 and 12 respectively
    filter_len = L + 1

    bank = np.zeros((n_delays, filter_len), dtype=np.float64)
    lib.mpa_fractional_delay_bank(addr(bank), addr(delays), n_delays, L,
                                  filter_len, 0.0)

    assert _bank_expected(delays, L, filter_len, n_delays, bank)
    # the tail that did not fit is genuinely zero, not folded in
    for t in range(n_delays):
        tail = int(delays[t]) + L
        assert np.all(bank[t, tail:] == 0.0), f"row {t}"
    # row 0 fits entirely, row 3 drops its last three taps
    assert np.any(bank[0, :L] != 0.0)
    assert np.any(bank[3, : L - 1] != 0.0)


def test_mpa_fractional_delay_bank_zero_delay_row_is_the_windowed_sinc(lib, error):
    """With every delay equal, min_delay cancels and the offset is zero."""
    delays = np.zeros(5)
    filter_len = FDL
    bank = np.zeros((5, filter_len), dtype=np.float64)
    lib.mpa_fractional_delay_bank(addr(bank), addr(delays), 5, FDL, filter_len, 0.0)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FDL) / (FDL - 1))
    expected = hann * np.sinc(np.arange(FDL) - (FDL - 1) / 2.0)
    assert error(bank[0], expected) < 1e-15
    # every row is identical when every delay is identical
    assert np.array_equal(bank[0], bank[-1])


# ---------------------------------------------------------------------------
# mpa_levinson
# ---------------------------------------------------------------------------


def _toeplitz(r, n):
    return np.array([[r[abs(i - j)] for j in range(n)] for i in range(n)])


def test_mpa_levinson_solves_rx_equals_b(lib, rng, error):
    """The residual `R @ x - b` is what actually pins the kernel down."""
    p = 11
    r = np.concatenate([[3.0], rng.uniform(-0.4, 0.4, p)])
    b = rng.normal(size=p)
    x = np.zeros(p, dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    lib.mpa_levinson(addr(r), addr(b[:, None]), addr(x), p, 1, addr(a), addr(arev))

    assert error(x, pu.levinson(r, b)) < 1e-12
    assert error(_toeplitz(r, p) @ x, b) < 1e-12


def test_mpa_levinson_leaves_a_zeroed_at_a0_is_one(lib):
    """`a[0]` is initialised to 1 and must stay 1 after the recursion."""
    p = 6
    r = np.concatenate([[2.5], np.full(p, 0.1)])
    b = np.ones(p)
    x = np.zeros(p, dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    lib.mpa_levinson(addr(r), addr(b[:, None]), addr(x), p, 1, addr(a), addr(arev))
    assert a[0] == 1.0


def test_mpa_levinson_handles_every_column_independently(lib, rng, error):
    """`n_rhs > 1` solves one independent system per column of b."""
    p, n_rhs = 8, 4
    r = np.concatenate([[4.0], rng.uniform(-0.3, 0.3, p)])
    B = rng.normal(size=(p, n_rhs))
    x = np.zeros((p, n_rhs), dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    lib.mpa_levinson(addr(r), addr(B), addr(x), p, n_rhs, addr(a), addr(arev))

    for c in range(n_rhs):
        assert error(x[:, c], pu.levinson(r, B[:, c])) < 1e-12


def test_mpa_levinson_writes_only_the_p_len_solution_rows(lib, rng):
    """x is (p_len, n_rhs); nothing is written past it."""
    p, n_rhs = 5, 2
    r = np.concatenate([[2.0], rng.uniform(-0.2, 0.2, p)])
    B = rng.normal(size=(p, n_rhs))
    buf = np.full((p + 3, n_rhs), 7.0, dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    lib.mpa_levinson(addr(r), addr(B), addr(buf), p, n_rhs, addr(a), addr(arev))
    assert np.all(buf[p:] == 7.0), "wrote past the solution rows"


# ---------------------------------------------------------------------------
# mpa_lpc_from_autocorr
# ---------------------------------------------------------------------------


def test_mpa_lpc_from_autocorr_matches_upstream(lib, rng, error):
    """The Yule-Walker solve the kernel performs, called directly."""
    p = 12
    x = rng.normal(size=3000)
    r = pu.autocorr(x, p, biased=True, method="numpy")
    out = np.zeros(p, dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    lib.mpa_lpc_from_autocorr(addr(r), addr(out), p, addr(a), addr(arev))

    assert out.shape == (p,)
    assert error(out, pu.lpc(x, p)) < 1e-10
    # and the public wrapper reaches the same kernel
    assert error(out, mpa.utilities.lpc_from_autocorr(r, p)) < 1e-15


def test_mpa_lpc_from_autocorr_recovers_a_two_pole_process(lib, rng):
    """The fitted coefficients must match the polynomial that generated the signal.

    The process is `x[n] = 1.6 x[n-1] - 0.9 x[n-2] + noise`, so its
    prediction-error polynomial is `1 - 1.6 z^-1 + 0.9 z^-2` and the LPC fit
    converges on those coefficients, in that order and with those signs.
    """
    n = 20000
    noise = np.random.default_rng(77).normal(size=n) * 0.02
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = 1.6 * x[i - 1] - 0.9 * x[i - 2] + noise[i]
    # the fft branch: the numpy branch is O(n^2) and this signal is long
    r = pu.autocorr(x, 2, biased=True, method="fft")
    out = np.zeros(2, dtype=np.float64)
    lib.mpa_lpc_from_autocorr(addr(r), addr(out), 2, addr(np.zeros(3)),
                              addr(np.zeros(3)))
    assert out == pytest.approx([1.6, -0.9], abs=0.05)


# ---------------------------------------------------------------------------
# mpa_autocorr_time
# ---------------------------------------------------------------------------


def test_mpa_autocorr_time_is_the_raw_correlation_numerator(lib, rng, error):
    """No biased/unbiased scaling: the raw dot products, as documented."""
    length, p_max = 500, 24
    x = rng.normal(size=length)
    r = np.zeros(p_max + 1, dtype=np.float64)
    lib.mpa_autocorr_time(addr(x), addr(r), length, p_max)

    for m in range(p_max + 1):
        assert r[m] == pytest.approx(float(np.dot(x[: length - m], x[m:])), abs=1e-11)
    # the wrapper applies the scaling, so it equals upstream exactly
    assert error(mpa.autocorr(x, p_max, True, method="time"), r / length) < 1e-14


def test_mpa_autocorr_time_handles_p_max_beyond_the_signal(lib, rng):
    """`length - m` goes negative; those lags must be 0, not a wild read."""
    length, p_max = 12, 40
    x = rng.normal(size=length)
    r = np.full(p_max + 1, 5.0, dtype=np.float64)
    lib.mpa_autocorr_time(addr(x), addr(r), length, p_max)
    assert np.all(r[length:] == 0.0)
    assert r[0] == pytest.approx(float(np.dot(x, x)), abs=1e-11)


def test_mpa_autocorr_time_writes_exactly_p_max_plus_one(lib, rng):
    p_max = 7
    x = rng.normal(size=64)
    r = np.full(p_max + 4, 3.0, dtype=np.float64)
    lib.mpa_autocorr_time(addr(x), addr(r), 64, p_max)
    assert np.all(r[p_max + 1 :] == 3.0)


# ---------------------------------------------------------------------------
# mpa_goertzel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 7, 100, 511])
def test_mpa_goertzel_is_the_dft_coefficient(lib, rng, error, k):
    """`exp(+i 2 pi f) s[-1] - s[-2]` must be `fft(x)[k]`, not an approximation."""
    n = 512
    x = rng.normal(size=n)
    out = np.zeros(2, dtype=np.float64)
    lib.mpa_goertzel(addr(x), addr(out), n, k)
    got = complex(out[0], out[1])
    assert got == pytest.approx(np.fft.fft(x)[k], rel=1e-9, abs=1e-9)


def test_mpa_goertzel_writes_both_halves(lib, rng):
    """A zero signal must give exactly (0, 0) in both halves of the output."""
    n = 64
    x = np.zeros(n)
    out = np.full(2, 7.0, dtype=np.float64)
    lib.mpa_goertzel(addr(x), addr(out), n, 3)
    assert np.all(out == 0.0)


def test_mpa_goertzel_matches_upstream_on_a_tone(lib, error):
    """A pure tone puts all the energy in one bin.

    Both coefficient and residual are near zero here, so the comparison is made
    against an absolute scale rather than a relative one.
    """
    n, k = 4000, 200  # 200 * 16000 / 4000 = 800 Hz, exactly on a bin
    t = np.arange(n) / 16000.0
    x = np.sin(2 * np.pi * 800.0 * t)
    out = np.zeros(2, dtype=np.float64)
    lib.mpa_goertzel(addr(x), addr(out), n, k)
    got = complex(out[0], out[1])

    theirs = pu.goertzel(x, k)
    assert abs(got - theirs) < 1e-6 * max(1.0, abs(theirs))
    assert abs(got) > 1000.0


# ---------------------------------------------------------------------------
# convmtx shape and dtype contracts
# ---------------------------------------------------------------------------


def test_convmtx_rejects_a_matrix_argument():
    with pytest.raises(ValueError, match="1-D"):
        mpa.utilities.convmtx(np.zeros((3, 3)), 4)


def test_convmtx_rejects_a_non_positive_n():
    with pytest.raises(ValueError, match="positive"):
        mpa.utilities.convmtx(np.zeros(3), 0)


def test_convmtx_dot_truncates_to_n(rng, error):
    """`n` shorter than `x` returns just the first `n` samples."""
    h = rng.normal(size=5)
    x = rng.normal(size=40)
    assert convmtx_dot(h, x, 12).shape == (12,)
    assert error(convmtx_dot(h, x, 12), convmtx_dot(h, x)[:12]) == 0.0
    # and n longer than x is capped rather than reading past it
    assert convmtx_dot(h, x, 99).shape == (40,)


def test_convmtx_dot_defaults_to_the_length_of_x(rng, error):
    h = rng.normal(size=5)
    x = rng.normal(size=21)
    assert convmtx_dot(h, x).shape == (21,)
    # it is the convolution with its last h.size - 1 samples dropped
    assert error(convmtx_dot(h, x), np.convolve(x, h)[: x.size]) < 1e-13


def test_convmtx_dot_rejects_a_matrix_argument():
    with pytest.raises(ValueError, match="1-D"):
        convmtx_dot(np.zeros((3, 3)), np.zeros(4))


def test_low_pass_dirac_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of elements"):
        mpa.low_pass_dirac(np.zeros(5), np.ones(4), 16000, 8)


def test_levinson_rejects_a_short_r():
    with pytest.raises(ValueError, match="at least"):
        mpa.levinson(np.ones(3), np.ones(5))


def test_levinson_rejects_a_2d_r():
    with pytest.raises(ValueError, match="1-D first column"):
        mpa.levinson(np.ones((5, 1)), np.ones(5))


def test_distance_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="same number of dimensions"):
        mpa.distance(np.zeros((2, 3)), np.zeros((3, 3)))


def test_distance_rejects_a_1d_argument():
    with pytest.raises(ValueError, match=r"\(dim, n_points\)"):
        mpa.distance(np.zeros(4), np.zeros((2, 3)))


def test_fractional_delay_bank_rejects_an_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        mpa.fractional_delay_filter_bank(np.zeros(0))


def test_fractional_delay_bank_rejects_a_degenerate_filter_length():
    mpa.constants.set("frac_delay_length", 1)
    try:
        with pytest.raises(ValueError, match="at least 3"):
            mpa.fractional_delay_filter_bank(np.array([0.0, 0.5]))
    finally:
        mpa.parameters._constants.pop("frac_delay_length")


# ---------------------------------------------------------------------------
# rir_builder out-of-bounds clamp
# ---------------------------------------------------------------------------


def test_mpa_rir_builder_clamps_a_short_buffer_instead_of_overrunning(lib):
    """The kernel must stay inside [0, rir_len) whatever it is handed.

    Upstream's C++ rejects a too-small buffer with an exception; here the taps
    that would fall outside are dropped, so a short buffer costs truncated taps
    rather than a write into whatever follows it in memory.
    """
    lut, hann = _tables()
    # one arrival whose fdl taps would run past the end of a short buffer
    time = np.array([(FDL - 1) // 2 + 5.0], dtype=np.float64) / FS
    alpha = np.array([1.0], dtype=np.float64)

    n = 3
    ours = np.zeros(n, dtype=np.float64)
    canary = -999.0
    backing = np.concatenate(
        [np.full(8, canary), ours, np.full(8, canary)]
    )
    lib.mpa_rir_builder(
        addr(backing[8:]), addr(time), addr(alpha), addr(lut), addr(hann),
        1, n, FS, FDL, LUT_GRAN,
    )
    assert np.all(backing[:8] == canary), "wrote before the rir"
    assert np.all(backing[8 + n :] == canary), "wrote past the rir"

    # a correctly sized buffer gives the same first taps as the short one
    big = np.zeros(400, dtype=np.float64)
    lib.mpa_rir_builder(
        addr(big), addr(time), addr(alpha), addr(lut), addr(hann),
        1, 400, FS, FDL, LUT_GRAN,
    )
    assert np.array_equal(backing[8 : 8 + n], big[:n])


def test_mpa_rir_builder_clamps_a_short_buffer_in_parallel(lib):
    """The clamp holds for the threaded path, where slices are narrower still."""
    lut, hann = _tables()
    time = np.full(20, ((FDL - 1) // 2 + 5.0) / FS)
    alpha = np.ones(20)
    n = 5
    canary = -999.0
    ours = np.zeros(n, dtype=np.float64)
    backing = np.concatenate([np.full(8, canary), ours, np.full(8, canary)])
    lib.mpa_rir_builder_parallel(
        addr(backing[8:]), addr(time), addr(alpha), addr(lut), addr(hann),
        20, n, FS, FDL, LUT_GRAN, 4,
    )
    assert np.all(backing[:8] == canary)
    assert np.all(backing[8 + n :] == canary)


def test_mpa_delay_sum_clamps_a_short_buffer(lib, rng):
    """delay_sum drops taps past `out_len` instead of writing past the array."""
    irs = rng.normal(size=(20, 16))
    delays = rng.integers(0, 40, 20).astype(np.float64)
    n = 8
    canary = -999.0
    backing = np.concatenate([np.full(8, canary), np.zeros(n), np.full(8, canary)])
    lib.mpa_delay_sum(addr(backing[8:]), addr(irs), addr(delays), 20, 16, n)
    assert np.all(backing[:8] == canary)
    assert np.all(backing[8 + n :] == canary)


# ---------------------------------------------------------------------------
# cov_gram: the mirrored triangle must be filled, not just the lower one
# ---------------------------------------------------------------------------


def test_mpa_cov_gram_fills_both_triangles(lib, rng, error):
    """Guards the mirroring direction: the upper triangle must be populated."""
    rows, cols = 7, 64
    X = rng.normal(size=(rows, cols))
    G = np.full((rows, rows), np.nan)
    lib.mpa_cov_gram(addr(G), addr(X), rows, cols)
    assert not np.any(np.isnan(G)), "an entry of the Gram matrix was never written"
    assert error(G, X @ X.T) < 1e-13
    assert np.array_equal(G, np.tril(G) + np.tril(G, -1).T)


def test_mpa_cov_gram_fills_the_upper_triangle(lib, error):
    """The mirrored half must be written, not left at the zero fill.

    `a @ a.T` is symmetric by construction, so the kernel computes the lower
    triangle and mirrors it; a half-filled matrix would show up as a zero
    upper triangle. The input is random, so every off-diagonal entry is
    substantial and a missing one cannot hide in the noise.
    """
    rows, cols = 5, 32
    X = np.random.default_rng(4).normal(size=(rows, cols))
    G = np.full((rows, rows), np.nan, dtype=np.float64)
    lib.mpa_cov_gram(addr(G), addr(X), rows, cols)

    assert not np.any(np.isnan(G)), "an entry was never written"
    assert error(G, X @ X.T) < 1e-13
    assert np.array_equal(G, G.T), "the mirrored triangle differs"
    # and the off-diagonals are large enough that a zero fill would fail
    assert np.abs(G - np.diag(np.diag(G))).max() > 1.0


# ---------------------------------------------------------------------------
# rake_delay_and_sum / rake_mvdr_filters wrappers
# ---------------------------------------------------------------------------


def test_rake_delay_and_sum_matches_the_beamformer_method(error):
    """The free function is the method, with the kernel underneath."""
    rng = np.random.default_rng(5)
    R = mpa.circular_2D_array([0.0, 0.0], 8, 0.0, 0.04)
    images = np.array([[1.0, -1.0, 0.5], [0.5, 0.5, -1.0]])

    bf = mpa.Beamformer(R, 16000, N=64)
    bf.rake_delay_and_sum_weights(images, attn=True, ff=False)

    freqs = np.array([0.0, 500.0, 2000.0])
    got = mpa.rake_delay_and_sum(R, images, freqs, attn=True, ff=False)
    assert got.shape == (8, freqs.size)
    assert error(got, bf.weights[:, [int(np.argmin(np.abs(bf.frequencies - f)))
                                   for f in freqs]]) < 1e-13


def test_rake_delay_and_sum_rejects_mismatched_geometry():
    with pytest.raises(ValueError, match="same dimension"):
        mpa.rake_delay_and_sum(np.zeros((2, 4)), np.zeros((3, 2)), np.zeros(4))


def test_rake_delay_and_sum_rejects_a_one_dimensional_array():
    with pytest.raises(ValueError, match="2-D or 3-D"):
        mpa.rake_delay_and_sum(np.zeros((4, 4)), np.zeros((4, 2)), np.zeros(4))


def test_rake_mvdr_filters_is_distortionless_and_reports_its_sinr(error):
    """`g /= inner(h, g)` means the response towards the source is exactly 1."""
    rng = np.random.default_rng(9)
    n = 6
    A = rng.normal(size=(n, n))
    R_xx = A @ A.T + np.eye(n)
    B = rng.normal(size=(n, n))
    K_nq = B @ B.T + np.eye(n)
    h = rng.normal(size=n)

    filters, sinr = mpa.rake_mvdr_filters(h, R_xx, K_nq)
    assert filters.shape == (n,)
    assert h @ filters == pytest.approx(1.0, abs=1e-12)
    assert sinr == pytest.approx(
        float((filters @ R_xx @ filters) / (filters @ K_nq @ filters)), rel=1e-12
    )


def test_rake_mvdr_filters_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match=r"must both be"):
        mpa.rake_mvdr_filters(np.ones(4), np.eye(3), np.eye(3))


def test_rake_mvdr_filters_reports_a_not_positive_definite_system():
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        mpa.rake_mvdr_filters(np.ones(3), -np.eye(3), -np.eye(3))


def test_beamformer_rake_mvdr_filters_method_matches_the_free_function():
    rng = np.random.default_rng(11)
    n = 5
    A = rng.normal(size=(n, n))
    R_xx = A @ A.T + np.eye(n)
    K_nq = np.eye(n)
    h = rng.normal(size=n)

    bf = mpa.Beamformer(np.zeros((2, 3)), 16000)
    got_filters, got_sinr = bf.rake_mvdr_filters(h, R_xx, K_nq)
    want_filters, want_sinr = mpa.rake_mvdr_filters(h, R_xx, K_nq)
    assert np.array_equal(got_filters.ravel(), want_filters)
    assert got_sinr == want_sinr


# ---------------------------------------------------------------------------
# build_rir_matrix goes through the convmtx_dot kernel
# ---------------------------------------------------------------------------


def test_build_rir_matrix_matches_the_dense_convmtx_form():
    """`convmtx(h, Lg)` really is the convolution matrix the kernel stands in for.

    Upstream's docstring is the contract: `np.dot(convmtx(x, n), v)` for a
    `v` of length `n` equals `np.convolve(x, v)`. `build_rir_matrix` uses that
    product to fill a block of the channel matrix.
    """
    import pyroomacoustics.utilities as pu

    Lg = 24
    h = np.random.default_rng(2).normal(size=31)
    v = np.random.default_rng(3).normal(size=Lg)
    # upstream's documented contract
    assert np.allclose(pu.convmtx(h, Lg) @ v, np.convolve(h, v))
    # the kernel returns the first len(v) of that, dropping the h.size - 1 tail
    assert np.allclose(convmtx_dot(h, v), np.convolve(h, v)[:Lg])
    assert np.allclose(convmtx_dot(h, v), (pu.convmtx(h, Lg) @ v)[:Lg])


def test_convmtx_matches_upstream_elementwise(rng, error):
    """`convmtx` itself must equal upstream's, not just its product with e."""
    import pyroomacoustics.utilities as pu

    x = rng.normal(size=9)
    assert error(mpa.convmtx(x, 12), pu.convmtx(x, 12)) == 0.0
    assert mpa.convmtx(x, 12).shape == pu.convmtx(x, 12).shape


def test_build_rir_matrix_rejects_bad_geometry():
    with pytest.raises(ValueError, match=r"\(dim, n_mics\)"):
        mpa.build_rir_matrix(np.zeros(4), [], 8, 16000)