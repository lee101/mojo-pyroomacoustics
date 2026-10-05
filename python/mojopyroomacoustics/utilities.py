"""Utilities from `pyroomacoustics.utilities`, with upstream's names and
signatures. Every body is a call into the compiled Mojo library.
"""

from __future__ import annotations

import math

import numpy as np

from . import _lib
from .parameters import constants

__all__ = [
    "convmtx",
    "convmtx_dot",
    "low_pass_dirac",
    "fractional_delay",
    "fractional_delay_filter_bank",
    "levinson",
    "autocorr",
    "lpc",
    "goertzel",
]


def convmtx(x, n):
    """Create a convolution matrix H for the vector x of size len(x) times n.

    Returns the dense Toeplitz matrix. The Mojo kernel `mpa_convmtx_dot`
    evaluates that matrix against a vector without materialising it; this
    wrapper returns the matrix itself because that is what upstream returns.
    Callers that only ever multiply by it should use
    `mojopyroomacoustics.utilities.convmtx_dot`.
    """
    x = np.asarray(x)
    n = int(n)
    if n <= 0:
        raise ValueError(f"`n` must be positive, got {n}")
    if x.ndim > 1:
        raise ValueError(
            f"`x` must be 1-D, got shape {x.shape}; see upstream's note that the "
            f"argument is a vector"
        )
    c = np.concatenate((x, np.zeros(n - 1)))
    r = np.zeros(n)
    return _toeplitz(c, r)


def convmtx_dot(h, x, n=None):
    """`dst = convmtx(h, len(x)) @ x`, the matrix-free form of `convmtx`.

    Upstream's contract is that `np.dot(convmtx(h, n), v)` for a `v` of length
    `n` equals `np.convolve(h, v)`, so this returns that product directly
    without forming the Toeplitz matrix. `build_rir_matrix` uses it: it assigns
    `convmtx(h, Lg).T` into a block of the channel matrix, which is the same
    product read the other way round.

    Parameters
    ----------
    h: ndarray (m,)
        The impulse response.
    x: ndarray (n,)
        The signal to filter. Its length is the number of output samples.
    n: int, optional
        Return only the first `n` samples of the product. Defaults to all of
        them. The kernel reads `x[i-k]` for `i < n` and `k < len(h)`, so a
        smaller `n` is always safe; a larger one would read past `x`.

    Returns
    -------
    ndarray
        The product, with `n` samples if `n` was given.
    """
    h = _lib.f64(h)
    x = _lib.f64(x)
    if h.ndim != 1 or x.ndim != 1:
        raise ValueError(
            f"`h` and `x` must both be 1-D, got {h.shape} and {x.shape}"
        )
    m = h.size
    out_n = x.size if n is None else int(n)
    if out_n < 0:
        raise ValueError(f"`n` must not be negative, got {n}")
    # the kernel has no length argument for x, so cap at x's own length
    out_n = min(out_n, x.size)
    dst = np.zeros(out_n, dtype=np.float64)
    _lib.require().mpa_convmtx_dot(_lib.addr(dst), _lib.addr(x), _lib.addr(h), out_n, m)
    return dst


def _toeplitz(c, r):
    out = np.empty((len(c), len(r)), dtype=np.result_type(c, r))
    for i in range(len(c)):
        for j in range(len(r)):
            out[i, j] = c[i - j] if i >= j else r[j - i]
    return out


def low_pass_dirac(t0, alpha, Fs, N):
    """Creates a vector containing a lowpass Dirac of duration T sampled at Fs
    with delay t0 and attenuation alpha.

    If t0 and alpha are 2D column vectors of the same size, the result is a
    matrix with one row per (t0, alpha) pair.
    """
    t0 = np.atleast_1d(np.asarray(t0, dtype=np.float64))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    # one row per t0, and the kernel reads alpha[i] for the same i, so the two
    # inputs must be the same length or the kernel would read past alpha
    if t0.size != alpha.size:
        raise ValueError(
            f"t0 and alpha must have the same number of elements, got "
            f"{t0.size} and {alpha.size}"
        )
    n_rows = t0.size
    dst = np.zeros((n_rows, N), dtype=np.float64)
    _lib.require().mpa_low_pass_dirac(
        _lib.addr(dst), _lib.addr(_lib.f64(t0.ravel())),
        _lib.addr(_lib.f64(alpha.ravel())), n_rows, N, float(Fs),
    )
    return dst


def fractional_delay(t0):
    """Creates a fractional delay filter using a windowed sinc function.

    The length of the filter is fixed by the module wide constant
    `frac_delay_length` (default 81).
    """
    N = constants.get("frac_delay_length")
    t0 = np.asarray(t0, dtype=np.float64)
    if t0.ndim == 1:
        t0 = t0[:, None]
    return np.hanning(N)[None, :] * np.sinc(
        np.arange(N)[None, :] - (N - 1) / 2 - t0
    )


def fractional_delay_filter_bank(delays):
    """Creates a fractional delay filter bank of windowed sinc filters.

    Parameters
    ----------
    delays: 1d ndarray
        The delays corresponding to each filter in fractional samples.

    Returns
    -------
    ndarray (N, filter_length)
        Row `i` is the fractional delay filter for delay `i`.
    """
    delays = np.asarray(delays, dtype=np.float64).ravel()
    if delays.size == 0:
        raise ValueError("`delays` must not be empty")
    n_delays = delays.size
    L = int(constants.get("frac_delay_length"))
    if L < 3:
        # the window is 0.5 - 0.5 cos(2 pi k / (l - 1)) and divides by l - 1
        raise ValueError(f"frac_delay_length must be at least 3, got {L}")
    min_delay = float(delays.min())
    # upstream: filter_length = L + int(np.ceil(delays - min).max())
    filter_len = L + int(math.ceil(float((delays - min_delay).max())))
    bank = np.zeros((n_delays, filter_len), dtype=np.float64)
    _lib.require().mpa_fractional_delay_bank(
        _lib.addr(bank), _lib.addr(_lib.f64(delays)), n_delays, L, filter_len, min_delay,
    )
    return bank


def levinson(r, b):
    """Solve a system of the form Rx=b where R is a Hermitian Toeplitz matrix
    and b is any vector, using the generalized Levinson recursion.

    Parameters
    ----------
    r: ndarray
        First column of R, the Toeplitz Hermitian matrix.
    b: ndarray
        The right-hand argument. If b is a matrix, the system is solved for
        every column vector in b.

    Returns
    -------
    ndarray
        The solution of the linear system Rx = b.
    """
    r = np.asarray(r, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b2 = b[:, None]
        n_rhs = 1
    else:
        b2 = b
        n_rhs = b2.shape[1]
    p_len = b2.shape[0]
    if r.ndim != 1:
        raise ValueError(f"`r` must be the 1-D first column of R, got shape {r.shape}")
    if r.size < p_len:
        # upstream reads exactly r[0..p_len-1]; anything shorter would let the
        # kernel read past the buffer
        raise ValueError(
            f"`r` must have at least p_len = {p_len} entries, got {r.size}"
        )
    x = np.zeros((p_len, n_rhs), dtype=np.float64)
    # the recursion writes a[0..p_len] and arev[0..j-1] with j up to p_len - 1,
    # so both scratch buffers need p_len + 1 entries, not p_len
    a = np.zeros(p_len + 1, dtype=np.float64)
    arev = np.zeros(p_len + 1, dtype=np.float64)
    _lib.require().mpa_levinson(
        _lib.addr(_lib.f64(r)), _lib.addr(_lib.f64(b2)), _lib.addr(x),
        p_len, n_rhs, _lib.addr(a), _lib.addr(arev),
    )
    return x[:, 0] if b.ndim == 1 else x


def autocorr(x, p, biased=True, method="numpy"):
    """Compute the autocorrelation for real signal `x` up to lag `p`.

    `method` may be 'numpy', 'fft', 'time' or 'pra'. 'time' runs the Mojo
    kernel; the others delegate to the same NumPy expressions upstream uses, so
    they agree with upstream exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    if method == "time":
        r = np.zeros(p + 1, dtype=np.float64)
        _lib.require().mpa_autocorr_time(_lib.addr(_lib.f64(x)), _lib.addr(r), L, p)
    elif method == "fft":
        X = np.fft.rfft(np.r_[x, np.zeros_like(x)])
        r = np.fft.irfft(X * np.conj(X))[: p + 1]
    elif method == "numpy":
        r = np.correlate(x, x, "full")[L - 1 : L + p]
    elif method == "pra":
        from pyroomacoustics.utilities import autocorr as pra_autocorr

        return pra_autocorr(x, p, biased, "pra")
    else:
        raise ValueError(
            "Invalid `method` for computing autocorrelation"
            "choose one of: `fft`, `time`, `numpy`, `pra`."
        )

    if biased:
        return r / L
    return r / np.arange(L, L - p - 1, step=-1)


def lpc(x, p, biased=True):
    """Compute `p` LPC coefficients for a speech segment `x`.

    Returns
    -------
    ndarray
        `p` LPC coefficients.
    """
    r = autocorr(x, p, biased, method="time")
    return lpc_from_autocorr(r, p)


def lpc_from_autocorr(r, p):
    """Solve the Yule-Walker equations for `p` LPC coefficients from `r`.

    This is upstream's `levinson(r[:p], r[1:])`, reached without recomputing the
    autocorrelation; `lpc` uses it after calling `autocorr`.
    """
    r = np.asarray(r, dtype=np.float64)
    x = np.zeros(p, dtype=np.float64)
    a = np.zeros(p + 1, dtype=np.float64)
    arev = np.zeros(p + 1, dtype=np.float64)
    _lib.require().mpa_lpc_from_autocorr(
        _lib.addr(_lib.f64(r)), _lib.addr(x), p, _lib.addr(a), _lib.addr(arev),
    )
    return x


def goertzel(x, k):
    """Goertzel algorithm to compute DFT coefficient `k` of a real signal."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _lib.require().mpa_goertzel(_lib.addr(_lib.f64(x)), _lib.addr(out), len(x), int(k))
    return complex(out[0], out[1])