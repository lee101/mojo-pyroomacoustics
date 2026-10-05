"""The RIR kernels, checked against pyroomacoustics' own `libroom` C++.

`rir_builder`, `delay_sum` and `fractional_delay` are upstream's
`libroom_src/rir_builder.cpp`. Upstream dispatches on dtype, so passing float64
here makes it run the float64 instantiation and the comparison is exact rather
than float32-limited.
"""

import math

import numpy as np
import pytest

from mojopyroomacoustics import _lib
from conftest import addr, f64, rel_err

import pyroomacoustics.utilities as pu


FS = 16000
FDL = 81
LUT_GRAN = 20


def tables(fdl=FDL, lut_gran=LUT_GRAN):
    lut = np.zeros((fdl + 2) * lut_gran, dtype=np.float64)
    hann = np.zeros(fdl, dtype=np.float64)
    lib = _lib.require()
    lib.mpa_build_sinc_lut(addr(lut), fdl, lut_gran)
    lib.mpa_build_hann(addr(hann), fdl)
    return lut, hann


def rir_len_for(time, fdl=FDL):
    """Long enough for the last arrival's filter tail, as the C++ bounds check."""
    fdl2 = (fdl - 1) // 2
    return int(math.ceil(FS * float(time.max()))) + fdl2 + fdl + 2


# --------------------------------------------------------------- tables


def test_mpa_build_sinc_lut_matches_upstream_formula(lib, error):
    """The sinc LUT is the `sin(pi n)/(pi n)` walk from rir_builder.cpp."""
    lut = np.zeros((FDL + 2) * LUT_GRAN, dtype=np.float64)
    lib.mpa_build_sinc_lut(addr(lut), FDL, LUT_GRAN)

    fdl2 = (FDL - 1) // 2
    n = -fdl2 - 1.0
    expected = []
    for _ in range((FDL + 2) * LUT_GRAN):
        expected.append(1.0 if n == 0.0 else math.sin(math.pi * n) / (math.pi * n))
        n += 1.0 / LUT_GRAN

    assert error(lut, np.array(expected)) < 1e-15
    # the removable singularity is exactly 1, not the limit of the division
    assert lut[np.argmin(np.abs(np.array(expected) - 1.0))] == 1.0


def test_mpa_build_hann_matches_upstream_formula(lib, error):
    hann = np.zeros(FDL, dtype=np.float64)
    lib.mpa_build_hann(addr(hann), FDL)
    expected = [
        0.5 - 0.5 * math.cos((math.pi * 2 * idx) / (FDL - 1)) for idx in range(FDL)
    ]
    assert error(hann, np.array(expected)) < 1e-16
    # the symmetric window starts and ends at zero and peaks in the middle
    assert hann[0] == 0.0 and hann[-1] == 0.0
    assert hann[FDL // 2] == pytest.approx(1.0)


# ---------------------------------------------------------- rir_builder


@pytest.mark.parametrize("n_times", [1, 7, 5000])
def test_mpa_rir_builder_matches_libroom(lib, pra_libroom, rng, error, n_times):
    """Serial `rir_builder` reproduces upstream's C++ exactly."""
    lut, hann = tables()
    # upstream's C++ requires floor(fs * t_min) - fdl2 >= 0
    t_min = ((FDL - 1) // 2 + 1) / FS
    time = rng.uniform(t_min + 1.0 / FS, 0.05, n_times)
    alpha = rng.uniform(0.01, 1.0, n_times)
    n = rir_len_for(time)

    theirs = np.zeros(n, dtype=np.float64)
    pra_libroom.rir_builder(theirs, time, alpha, FS, FDL, LUT_GRAN, 1)

    ours = np.zeros(n, dtype=np.float64)
    lib.mpa_rir_builder(addr(ours), addr(time), addr(alpha), addr(lut), addr(hann),
                        n_times, n, FS, FDL, LUT_GRAN)

    assert error(ours, theirs) < 1e-14


def test_mpa_rir_builder_parallel_is_bit_identical_to_serial(lib, rng):
    """The threaded path partitions the output, so it cannot race.

    Each worker owns a disjoint slice of the RIR and walks every arrival, so the
    result must be bit-for-bit the serial one at any worker count.
    """
    lut, hann = tables()
    time = rng.uniform(0.005, 0.2, 60000)
    alpha = rng.uniform(0.1, 1.0, 60000)
    n = rir_len_for(time)

    serial = np.zeros(n, dtype=np.float64)
    lib.mpa_rir_builder(addr(serial), addr(time), addr(alpha), addr(lut), addr(hann),
                        time.size, n, FS, FDL, LUT_GRAN)

    for workers in (2, 3, 8):
        par = np.zeros(n, dtype=np.float64)
        lib.mpa_rir_builder_parallel(addr(par), addr(time), addr(alpha), addr(lut),
                                    addr(hann), time.size, n, FS, FDL, LUT_GRAN, workers)
        assert np.array_equal(par, serial), f"workers={workers}"


def test_mpa_rir_builder_single_arrival_is_a_windowed_sinc(lib, error):
    """A lone arrival at an exact sample produces the centred windowed sinc."""
    lut, hann = tables()
    time = np.array([100.0 / FS], dtype=np.float64)
    alpha = np.array([0.5], dtype=np.float64)
    n = 4 * FS // 100
    ours = np.zeros(n, dtype=np.float64)
    lib.mpa_rir_builder(addr(ours), addr(time), addr(alpha), addr(lut), addr(hann),
                        1, n, FS, FDL, LUT_GRAN)

    # the arrival lands at 100 - fdl2 and the peak tap is the window maximum
    peak = 100 - (FDL - 1) // 2 + FDL // 2
    assert ours[peak] == pytest.approx(0.5, rel=1e-12)
    assert np.argmax(ours) == peak


# ----------------------------------------------------------- delay_sum


def test_mpa_delay_sum_matches_libroom(lib, pra_libroom, rng, error):
    irs = rng.normal(size=(200, 32))
    delays = rng.integers(0, 5000, 200).astype(np.float64)
    n = 5200

    theirs = np.zeros(n, dtype=np.float64)
    pra_libroom.delay_sum(irs, delays, theirs, 1)

    ours = np.zeros(n, dtype=np.float64)
    lib.mpa_delay_sum(addr(ours), addr(irs), addr(delays), irs.shape[0], irs.shape[1], n)

    assert error(ours, theirs) < 1e-15


def test_mpa_delay_sum_parallel_is_bit_identical_to_serial(lib, rng):
    irs = rng.normal(size=(5000, 40))
    delays = rng.integers(0, 20000, 5000).astype(np.float64)
    n = 21000

    serial = np.zeros(n, dtype=np.float64)
    lib.mpa_delay_sum(addr(serial), addr(irs), addr(delays), irs.shape[0],
                      irs.shape[1], n)

    for workers in (2, 4, 8):
        par = np.zeros(n, dtype=np.float64)
        lib.mpa_delay_sum_parallel(addr(par), addr(irs), addr(delays), irs.shape[0],
                                  irs.shape[1], n, workers)
        assert np.array_equal(par, serial), f"workers={workers}"


def test_mpa_delay_sum_places_each_row_at_its_delay(lib, rng):
    """A single unit row must appear, unchanged, at its delay offset."""
    irs = np.zeros((1, 16), dtype=np.float64)
    irs[0, 0] = 1.0
    delays = np.array([137.0], dtype=np.float64)
    n = 256
    out = np.zeros(n, dtype=np.float64)
    lib.mpa_delay_sum(addr(out), addr(irs), addr(delays), 1, 16, n)
    assert np.flatnonzero(out)[0] == 137
    assert out[137] == 1.0
    assert out.sum() == 1.0


# ---------------------------------------------------- fractional_delay


def test_mpa_fractional_delay_matches_libroom(lib, pra_libroom, rng, error):
    lut, hann = tables()
    time = rng.uniform(0.0, 1.0, 64)

    theirs = np.zeros((64, FDL), dtype=np.float64)
    pra_libroom.fractional_delay(theirs, time, LUT_GRAN, 1)

    ours = np.zeros((64, FDL), dtype=np.float64)
    lib.mpa_fractional_delay(addr(ours), addr(time), addr(lut), addr(hann),
                             time.size, FDL, LUT_GRAN)

    assert error(ours, theirs) < 1e-15


def test_mpa_fractional_delay_is_windowed_and_peaks_in_the_middle(lib):
    """Each row is the Hann window times an interpolated sinc, so it is
    dominated by the centre tap and tapers symmetrically enough to sum small."""
    lut, hann = tables()
    time = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    ours = np.zeros((4, FDL), dtype=np.float64)
    lib.mpa_fractional_delay(addr(ours), addr(time), addr(lut), addr(hann), 4,
                             FDL, LUT_GRAN)

    # Reproduce upstream's LUT walk directly, tap by tap.
    fdl2 = (FDL - 1) // 2
    lut_gran_f = float(LUT_GRAN)
    for i, tf in enumerate(time):
        x_off_frac = (1.0 - tf) * lut_gran_f
        lut_gran_off = math.floor(x_off_frac)
        x_off = x_off_frac - lut_gran_off
        lut_pos = int(lut_gran_off)
        expected = np.empty(FDL, dtype=np.float64)
        for k in range(FDL):
            expected[k] = hann[k] * (
                lut[lut_pos] + x_off * (lut[lut_pos + 1] - lut[lut_pos])
            )
            lut_pos += LUT_GRAN
        assert ours[i] == pytest.approx(expected, rel=0, abs=1e-15)
        # the Hann window ends the filter at zero in every row
        assert ours[i][0] == pytest.approx(0.0, abs=1e-15)
        assert ours[i][-1] == pytest.approx(0.0, abs=1e-15)
    assert not np.allclose(ours[0], ours[2])


# ------------------------------------------------- end-to-end RIR path


def test_ism_rir_assemble_sums_bands_into_the_rir(lib, rng):
    """`mpa_ism_rir_assemble` accumulates one rir_builder pass per band."""
    lut, hann = tables()
    n_img, n_bands = 40, 3
    time = rng.uniform(0.005, 0.02, n_img)
    amp = rng.uniform(0.1, 1.0, (n_bands, n_img))
    n = rir_len_for(time)

    ours = np.zeros(n, dtype=np.float64)
    lib.mpa_ism_rir_assemble(addr(ours), addr(time), addr(amp), n_img, n_bands, FS,
                             FDL, addr(lut), addr(hann), LUT_GRAN, n)

    # the same sum, formed band by band with the serial kernel
    expected = np.zeros(n, dtype=np.float64)
    for b in range(n_bands):
        lib.mpa_rir_builder(addr(expected), addr(time), addr(amp[b]), addr(lut),
                            addr(hann), n_img, n, FS, FDL, LUT_GRAN)
    assert np.array_equal(ours, expected)


def test_ism_rir_length_follows_upstreams_formula(lib, rng):
    n_img = 25
    time = rng.uniform(0.002, 0.03, n_img)
    got = lib.mpa_ism_rir_length(addr(time), n_img, FS, FDL)
    fdl2 = FDL // 2
    expected = int(math.ceil(float(time.max()) * FS + fdl2 + 1)) + 1
    assert got == expected


def test_ism_image_amplitudes_matches_upstream_expression(lib, rng, error):
    """`compute_ism_rir`'s per-image time and amplitude."""
    n_img = 32
    images = rng.normal(size=(3, n_img))
    mic = rng.normal(size=3)
    damping = rng.uniform(0.5, 1.0, n_img)
    air = np.array([0.01, 0.02, 0.05])
    n_bands = air.size

    dst = np.zeros(n_bands * n_img + n_img, dtype=np.float64)
    lib.mpa_ism_image_amplitudes(addr(dst), addr(images), addr(mic), addr(damping),
                                 3, n_img, 343.0, addr(air), n_bands)

    dist = np.sqrt(((images - mic[:, None]) ** 2).sum(axis=0))
    expected_amp = (damping[None, :] / dist[None, :]
                    * np.exp(-0.5 * air[:, None] * dist[None, :]))
    assert error(dst[: n_bands * n_img].reshape(n_bands, n_img), expected_amp) < 1e-15
    assert error(dst[n_bands * n_img :], dist / 343.0) < 1e-15


def test_rir_builder_from_images_is_a_realistic_impulse_response(rng):
    """The public path produces a peak at the direct-path arrival."""
    import mojopyroomacoustics as mpa

    fs = 16000
    src = np.array([1.0, 2.0, 1.5])
    mic = np.array([2.0, 1.0, 1.5])
    # a few image sources at increasing distance
    offsets = np.array([[0.0, 0.5, -0.7, 1.1], [0.0, 0.3, 0.9, -0.4],
                        [0.0, 0.0, 0.0, 0.0]])
    images = src[:, None] + offsets
    damping = np.array([1.0, 0.5, 0.25, 0.125])

    rir = mpa.rir_builder_from_images(images, damping, mic, fs=fs)
    assert rir.dtype == np.float64
    assert rir.ndim == 1 and rir.size > 100

    direct = np.linalg.norm(mic - src) / 343.0 * fs
    # the peak sits within the fractional delay filter length of the arrival
    assert abs(int(np.argmax(np.abs(rir))) - direct) <= mpa.constants.get(
        "frac_delay_length"
    )


def test_autocorr_matches_upstream_all_methods(rng, error):
    import mojopyroomacoustics as mpa

    x = rng.normal(size=2000)
    p = 32
    for biased in (True, False):
        reference = pu.autocorr(x, p, biased, "numpy")
        assert error(mpa.autocorr(x, p, biased, method="time"), reference) < 1e-14
        # the fft branch of upstream must agree with its numpy branch
        assert error(mpa.autocorr(x, p, biased, method="fft"), reference) < 1e-12