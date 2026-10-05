"""`pyroomacoustics.utilities` parity: the LPC, autocorrelation and
fractional-delay primitives.
"""

import numpy as np
import pytest

import mojopyroomacoustics as mpa
import pyroomacoustics.utilities as pu


# ---------------------------------------------------------- low_pass_dirac


def test_low_pass_dirac_matches_upstream(rng, error):
    t0 = rng.uniform(0, 5, 11)
    alpha = rng.uniform(0.1, 1.0, 11)
    ours = mpa.low_pass_dirac(t0[:, None], alpha[:, None], 16000, 64)
    theirs = pu.low_pass_dirac(t0[:, None], alpha[:, None], 16000, 64)
    assert ours.shape == theirs.shape
    assert error(ours, theirs) == 0.0


def test_low_pass_dirac_peaks_at_the_integer_delay(rng):
    """At t0 = k/Fs the low-pass Dirac peaks exactly at sample k."""
    n = 32
    ours = mpa.low_pass_dirac(np.array([[6 / 16000]]), np.array([[1.0]]), 16000, n)
    assert int(np.argmax(ours[0])) == 6
    assert ours[0, 6] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------- fractional_delay


def test_fractional_delay_matches_upstream(rng, error):
    d = rng.uniform(-2, 2, 13)
    assert error(mpa.fractional_delay(d), pu.fractional_delay(d)) == 0.0


def test_fractional_delay_uses_the_module_constant():
    L = mpa.constants.get("frac_delay_length")
    assert mpa.fractional_delay(np.zeros(1)).shape == (1, L)


# ------------------------------------------- fractional_delay_filter_bank


def test_fractional_delay_filter_bank_matches_upstream(rng, error):
    d = rng.uniform(0, 6, 17)
    assert error(mpa.fractional_delay_filter_bank(d),
                 pu.fractional_delay_filter_bank(d)) < 1e-15


def test_fractional_delay_filter_bank_shapes_and_length(rng):
    d = rng.uniform(0, 6, 17)
    bank = mpa.fractional_delay_filter_bank(d)
    L = mpa.constants.get("frac_delay_length")
    assert bank.shape == (17, L + int(np.ceil(d.max())))


# ------------------------------------------------------------- goertzel


@pytest.mark.parametrize("k", [0, 1, 100, 1000])
def test_goertzel_matches_upstream(rng, error, k):
    """Goertzel is a marginally stable recursion, so parity is to ~1e-9 not 1e-15.

    The residual is the algorithm's own conditioning: upstream runs the same
    second-order recursion through `lfilter`, and both accumulate the same
    slowly-growing error. What must match is the DFT coefficient.
    """
    x = rng.normal(size=4096)
    ours = mpa.goertzel(x, k)
    theirs = pu.goertzel(x, k)
    reference = np.fft.fft(x)[k]
    assert error([ours], [reference]) < 1e-9
    assert error([ours], [theirs]) < 1e-9


# ------------------------------------------------------------ autocorr


def test_autocorr_time_matches_upstream(rng, error):
    x = rng.normal(size=2000)
    for p in (16, 64):
        for biased in (True, False):
            reference = pu.autocorr(x, p, biased, "numpy")
            assert error(mpa.autocorr(x, p, biased, method="time"), reference) < 1e-13


def test_autocorr_zero_lag_is_the_mean_square(rng, error):
    x = rng.normal(size=500)
    r = mpa.autocorr(x, 8, biased=True, method="time")
    assert r[0] == pytest.approx(np.mean(x ** 2), rel=1e-14)


def test_autocorr_rejects_an_unknown_method(rng):
    with pytest.raises(ValueError, match="Invalid `method`"):
        mpa.autocorr(rng.normal(size=10), 2, method="nonsense")


# ------------------------------------------------------------ levinson


def _toeplitz(r, p):
    return np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])


@pytest.mark.parametrize("p", [2, 5, 12])
def test_levinson_solves_the_toeplitz_system(rng, error, p):
    r = np.concatenate([[5.0], rng.uniform(-0.3, 0.3, p)])
    b = rng.normal(size=p)
    ours = mpa.levinson(r, b)
    assert error(ours, pu.levinson(r, b)) < 1e-12
    R = _toeplitz(r, p)
    assert error(R @ ours, b) < 1e-12


@pytest.mark.parametrize("n_rhs", [1, 2, 5])
def test_levinson_multiple_right_hand_sides(rng, error, n_rhs):
    """Every column is an independent system; `np.dot` keeps delta per column."""
    p = 7
    r = np.concatenate([[4.0], rng.uniform(-0.3, 0.3, p)])
    B = rng.normal(size=(p, n_rhs))
    ours = mpa.levinson(r, B)
    assert error(ours, pu.levinson(r, B)) < 1e-12
    R = _toeplitz(r, p)
    assert error(R @ ours, B) < 1e-11


def test_levinson_reduces_to_the_scalar_solution(rng, error):
    p = 6
    r = np.concatenate([[6.0], rng.uniform(-0.2, 0.2, p)])
    B = rng.normal(size=(p, 3))
    as_matrix = mpa.levinson(r, B)
    for c in range(3):
        assert error(as_matrix[:, c], mpa.levinson(r, B[:, c])) < 1e-13


# ---------------------------------------------------------------- lpc


@pytest.mark.parametrize("p", [8, 20])
def test_lpc_matches_upstream(rng, error, p):
    x = rng.normal(size=3000)
    assert error(mpa.lpc(x, p), pu.lpc(x, p)) < 1e-10


def test_lpc_of_white_noise_is_near_zero(rng):
    """An all-pole fit of white noise has no poles to find, so a stays small."""
    x = rng.normal(size=8000)
    a = mpa.lpc(x, 16)
    assert np.abs(a).sum() < 0.2


def test_lpc_predicts_a_tone_well(rng):
    """A two-pole process should be recovered with a stable, resonant filter."""
    t = np.arange(6000) / 8000.0
    a_true = np.array([1.0, -1.6, 0.9])
    x = np.zeros_like(t)
    for i in range(2, len(t)):
        x[i] = 1.6 * x[i - 1] - 0.9 * x[i - 2] + 0.02 * rng.normal()
    a = mpa.lpc(x, 2)
    # signs alternate against the polynomial coefficients
    assert a[0] == pytest.approx(-a_true[1], abs=0.05)
    assert a[1] == pytest.approx(-a_true[2], abs=0.05)
    # and the prediction error is much smaller than the signal
    assert np.sqrt(np.mean((x[2:] - (a[0] * x[1:-1] + a[1] * x[:-2])) ** 2)) < 0.05


# ------------------------------------------------------------ convmtx


def test_convmtx_matches_upstream(rng, error):
    x = rng.normal(size=10)
    h = rng.normal(size=5)
    assert error(mpa.convmtx(h, 12), pu.convmtx(h, 12)) == 0.0


# ----------------------------------------------------------- constants


def test_constants_round_trip():
    """`set` overrides the module-wide default until it is removed again."""
    from mojopyroomacoustics import parameters

    mpa.constants.set("frac_delay_length", 65)
    assert mpa.constants.get("frac_delay_length") == 65
    parameters._constants.pop("frac_delay_length")
    assert mpa.constants.get("frac_delay_length") == 81


def test_speed_of_sound_default_matches_upstream():
    import pyroomacoustics.parameters as ppar

    for key in ("c", "ffdist", "frac_delay_length", "sinc_lut_granularity"):
        assert mpa.constants.get(key) == ppar.constants.get(key)


def test_unknown_constant_raises():
    with pytest.raises(NameError, match="no such constant"):
        mpa.constants.get("definitely_not_a_constant")