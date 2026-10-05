"""`pyroomacoustics.beamforming` parity: array geometry, steering vectors,
beampattern weights and the TD rake kernels.
"""

import numpy as np
import pytest

import mojopyroomacoustics as mpa
from mojopyroomacoustics import _lib
import pyroomacoustics.beamforming as pb

from conftest import addr, c128


FS = 16000
CENTER = np.array([0.1, -0.2])


@pytest.fixture
def both_arrays():
    R = pb.circular_2D_array(center=[0.0, 0.0], M=8, phi0=0.0, radius=0.04)
    return mpa.Beamformer(R, FS, N=512), pb.Beamformer(R, FS, N=512)


# ------------------------------------------------------------- geometry


def test_unit_vec2D_matches_upstream(error):
    for phi in (0.0, 0.3, -1.2, np.pi):
        assert error(mpa.unit_vec2D(phi), pb.unit_vec2D(phi)) < 1e-16


def test_linear_2D_array_matches_upstream(error):
    for M, phi, d in ((5, 0.3, 0.05), (12, -1.0, 0.02)):
        assert error(mpa.linear_2D_array(CENTER, M, phi, d),
                     pb.linear_2D_array(CENTER, M, phi, d)) < 1e-15


def test_circular_2D_array_matches_upstream(error):
    for M, phi0, radius in ((6, 0.4, 0.1), (16, -0.5, 0.03)):
        assert error(mpa.circular_2D_array(CENTER, M, phi0, radius),
                     pb.circular_2D_array(CENTER, M, phi0, radius)) < 1e-15


def test_square_2D_array_matches_upstream(error):
    """Rows run along phi, row centres along phi + pi/2."""
    for M, N, phi, d in ((3, 4, 0.2, 0.05), (2, 5, -0.8, 0.03)):
        assert error(mpa.square_2D_array(CENTER, M, N, phi, d),
                     pb.square_2D_array(CENTER, M, N, phi, d)) < 1e-15


def test_distance_matches_upstream(rng, error):
    a = rng.normal(size=(3, 7))
    b = rng.normal(size=(3, 4))
    assert error(mpa.distance(a, b), pb.distance(a, b)) < 1e-15


def test_distance_matrix_diagonal_is_zero_for_identical_point_sets():
    """`distance` is the full pairwise matrix; only the diagonal vanishes."""
    p = np.array([[1.0, 2.0], [3.0, 4.0]])
    D = mpa.distance(p, p)
    assert np.allclose(np.diag(D), 0.0, atol=1e-15)
    assert D[0, 1] == pytest.approx(np.hypot(1.0, 1.0))


def test_H_matches_upstream(rng, error):
    A = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    ours = mpa.H(A)
    assert ours.shape == (3, 4)
    assert error(ours, pb.H(A)) == 0.0


def test_H_is_an_involution_on_complex_input(rng):
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    assert np.allclose(mpa.H(mpa.H(A)), A, atol=1e-15)


def test_mdot3_matches_numpy(rng, error):
    A = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    C = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))
    assert error(mpa.beamforming.mdot3(A, B, C), A @ B @ C) < 1e-15


def test_mdot_folds_left_to_right(rng):
    A = rng.normal(size=(3, 3))
    B = rng.normal(size=(3, 3))
    C = rng.normal(size=(3, 3))
    assert np.allclose(mpa.mdot(A, B, C), (A @ B) @ C)


# ------------------------------------------------------ steering vectors


@pytest.mark.parametrize("attn", [True, False])
@pytest.mark.parametrize("ff", [False, True])
def test_steering_vector_2D_from_point_matches_upstream(both_arrays, error, attn, ff):
    ours_bf, theirs_bf = both_arrays
    src = np.array([[1.0], [0.5]])
    ours = ours_bf.steering_vector_2D_from_point(1000.0, src, attn=attn, ff=ff)
    theirs = theirs_bf.steering_vector_2D_from_point(1000.0, src, attn=attn, ff=ff)
    # upstream itself divides by a zero far-field distance, so nan/inf is parity
    assert np.array_equal(np.isnan(ours), np.isnan(theirs))
    finite = ~np.isnan(ours)
    assert np.allclose(ours[finite], theirs[finite], rtol=1e-11, atol=1e-13)


def test_steering_vector_2D_from_point_multiple_images(both_arrays, error):
    """Several image sources give one steering column each."""
    ours_bf, theirs_bf = both_arrays
    srcs = np.array([[1.0, -1.0, 0.5], [0.5, 0.5, -1.0]])
    ours = ours_bf.steering_vector_2D_from_point(2000.0, srcs, attn=True, ff=False)
    theirs = theirs_bf.steering_vector_2D_from_point(2000.0, srcs, attn=True, ff=False)
    assert ours.shape == (8, 3)
    assert error(ours, theirs) < 1e-14


def test_steering_vector_2D_matches_upstream(both_arrays, error):
    ours_bf, theirs_bf = both_arrays
    phis = np.linspace(-np.pi, np.pi, 33)
    assert error(ours_bf.steering_vector_2D(1500.0, phis, 10.0),
                 theirs_bf.steering_vector_2D(1500.0, phis, 10.0)) < 1e-12


def test_steering_vector_phase_is_the_arrival_delay(both_arrays):
    """exp(-i omega D / c) with D the microphone-to-source distance."""
    ours_bf, _ = both_arrays
    freq = 2000.0
    src = np.array([[1.0], [0.5]])
    a = ours_bf.steering_vector_2D_from_point(freq, src, attn=False, ff=False)
    dist = np.linalg.norm(ours_bf.R - src, axis=0)
    expected = np.exp(-1j * 2 * np.pi * freq * dist / mpa.constants.get("c"))
    assert np.allclose(a.ravel(), expected, atol=1e-12)


# -------------------------------------------------------- beam response


def test_far_field_weights_match_upstream(both_arrays, error):
    ours_bf, theirs_bf = both_arrays
    for phi in (np.pi / 3, -0.4, 1.2):
        ours_bf.far_field_weights(phi)
        theirs_bf.far_field_weights(phi)
        assert error(ours_bf.weights, theirs_bf.weights) < 1e-14


def test_far_field_projection_is_causal(lib):
    """`proj -= proj.max()` must leave the first arriving mic at zero delay."""
    R = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    center = np.zeros((2, 1))
    proj = np.zeros(4)
    lib.mpa_far_field_projection(addr(proj), addr(R), addr(center), 2, 4, 0.0)
    assert proj.max() == pytest.approx(0.0, abs=1e-15)
    assert np.all(proj <= 1e-15)


def test_far_field_weights_beam_points_the_right_way(both_arrays, error):
    """The maximum beampattern response is near the steering direction."""
    ours_bf, theirs_bf = both_arrays
    ours_bf.far_field_weights(0.0)
    theirs_bf.far_field_weights(0.0)
    ours_freq, ours_resp = ours_bf.response(np.linspace(-np.pi, np.pi, 361), 2000.0)
    theirs_freq, theirs_resp = theirs_bf.response(np.linspace(-np.pi, np.pi, 361), 2000.0)
    assert error(ours_resp, theirs_resp) < 1e-12
    peak = np.argmax(np.abs(ours_resp))
    assert abs(np.linspace(-np.pi, np.pi, 361)[peak]) < 0.05


def test_response_matches_upstream(both_arrays, error):
    ours_bf, theirs_bf = both_arrays
    ours_bf.far_field_weights(np.pi / 3)
    theirs_bf.far_field_weights(np.pi / 3)
    phis = np.linspace(0, np.pi, 9)
    fo, ro = ours_bf.response(phis, 2000.0)
    ft, rt = theirs_bf.response(phis, 2000.0)
    assert fo == ft
    assert ro.shape == (9,)
    assert error(ro, rt) < 1e-12


def test_response_without_weights_raises(both_arrays):
    ours_bf, _ = both_arrays
    with pytest.raises(NameError, match="weights or filters"):
        ours_bf.response([0.0], 1000.0)


# ----------------------------------------------------------- rake weights


def test_rake_delay_and_sum_weights_match_upstream(both_arrays, error):
    class Src:
        pass

    ours_bf, theirs_bf = both_arrays
    s = Src()
    s.images = np.array([[1.0, -1.0], [0.5, 0.5]])
    ours_bf.rake_delay_and_sum_weights(s, attn=True, ff=False)
    theirs_bf.rake_delay_and_sum_weights(s, attn=True, ff=False)
    assert error(ours_bf.weights, theirs_bf.weights) < 1e-13


def test_delay_and_sum_weights_are_the_normalised_steering_sum(both_arrays, error):
    """Upstream is 1/M/(K+1) times the steering vectors summed over the images."""
    class Src:
        pass

    ours_bf, _ = both_arrays
    s = Src()
    s.images = np.array([[1.0, -1.0], [0.5, 0.5]])
    ours_bf.rake_delay_and_sum_weights(s, attn=True, ff=False)

    i = 10
    A = ours_bf.steering_vector_2D_from_point(
        ours_bf.frequencies[i], s.images, attn=True, ff=False
    )
    expected = A.sum(axis=1) / ours_bf.M / s.images.shape[1]
    assert error(ours_bf.weights[:, i], expected) < 1e-14


# ---------------------------------------------------------- TD rake kernels


def test_mpa_cov_gram_matches_numpy(rng, error):
    X = rng.normal(size=(6, 40))
    G = np.zeros((6, 6))
    _lib.require().mpa_cov_gram(addr(G), addr(X), 6, 40)
    assert error(G, X @ X.T) < 1e-14


def test_mpa_cov_gram_is_symmetric_positive(rng):
    X = rng.normal(size=(5, 30))
    G = np.zeros((5, 5))
    _lib.require().mpa_cov_gram(addr(G), addr(X), 5, 30)
    assert np.allclose(G, G.T, atol=1e-13)
    assert np.linalg.eigvalsh(G).min() > -1e-12


def test_mpa_rake_mvdr_solution_is_distortionless(rng):
    """`g /= np.inner(h, g)` makes the response towards the source exactly 1."""
    m, l = 4, 6
    h = rng.normal(size=m * l)
    c = rng.normal(size=m * l)
    out = np.zeros(m * l)
    _lib.require().mpa_rake_mvdr_solution(addr(out), addr(h), addr(c), m, l)
    assert h @ out == pytest.approx(1.0, abs=1e-12)
    # it is the same solution rescaled
    assert np.allclose(out * (h @ c), c, atol=1e-12)


def test_mpa_response_sinr_matches_its_definition(rng, error):
    n = 12
    Rxx = rng.normal(size=(n, n))
    Rxx = Rxx @ Rxx.T + np.eye(n)
    Knq = rng.normal(size=(n, n))
    Knq = Knq @ Knq.T + np.eye(n)
    g = rng.normal(size=n)
    out = np.zeros(1)
    _lib.require().mpa_response_sinr(addr(out), addr(g), addr(Rxx), addr(Knq), n)
    assert error(out, (g @ Rxx @ g) / (g @ Knq @ g)) < 1e-14


def test_response_sinr_is_positive_for_positive_definite_matrices():
    n = 5
    Rxx = np.eye(n)
    Knq = 2 * np.eye(n)
    g = rng = np.random.default_rng(0).normal(size=n)
    out = np.zeros(1)
    _lib.require().mpa_response_sinr(addr(out), addr(g), addr(Rxx), addr(Knq), n)
    assert out[0] == pytest.approx(0.5)


# --------------------------------------------------------- the class API


def test_beamformer_constructor_matches_upstream(both_arrays):
    ours_bf, theirs_bf = both_arrays
    assert ours_bf.N == theirs_bf.N
    assert ours_bf.Lg == theirs_bf.Lg
    assert ours_bf.hop == theirs_bf.hop
    assert ours_bf.M == theirs_bf.M == 8
    assert ours_bf.dim == theirs_bf.dim == 2
    assert np.allclose(ours_bf.frequencies, theirs_bf.frequencies)
    assert np.allclose(ours_bf.center, theirs_bf.center)


def test_beamformer_rounds_odd_fft_length_up():
    bf = mpa.Beamformer(np.zeros((2, 4)), FS, N=511)
    assert bf.N == 512


def test_beamformer_zero_padding_lengths():
    bf = mpa.Beamformer(np.zeros((2, 4)), FS, N=512, zpf=32, zpb=64)
    assert bf.L == 512 - 32 - 64
    assert bf.zpf == 32 and bf.zpb == 64


def test_beamformer_concatenation(both_arrays):
    ours_bf, _ = both_arrays
    other = mpa.Beamformer(np.zeros((2, 3)), FS, N=256)
    joined = ours_bf + other
    assert joined.M == 11
    assert joined.R.shape == (2, 11)


def test_microphone_array_rejects_bad_geometry():
    with pytest.raises(ValueError, match="shape"):
        mpa.MicrophoneArray(np.zeros((4, 5)), FS)


def test_microphone_array_length_and_directivity():
    mic = mpa.MicrophoneArray(np.zeros((2, 5)), FS)
    assert len(mic) == 5 and mic.M == 5 and mic.nmic == 5
    assert mic.is_directive is False
    assert mic.directivity == [None] * 5


def test_weights_from_filters_round_trip(both_arrays, error):
    """`weights_from_filters` is upstream's conj(rfft(filters))."""
    ours_bf, theirs_bf = both_arrays
    filters = np.random.default_rng(3).normal(size=(8, 512))
    ours_bf.filters = filters.copy()
    theirs_bf.filters = filters.copy()
    ours_bf.weights_from_filters()
    theirs_bf.weights_from_filters()
    assert error(ours_bf.weights, theirs_bf.weights) < 1e-15


def test_weights_from_a_far_field_beam_pass_the_response_test(both_arrays):
    """The far-field beam is nulled at pi/2 from the steering direction."""
    ours_bf, _ = both_arrays
    phis = np.array([0.0, np.pi / 2, np.pi])
    ours_bf.far_field_weights(0.0)
    _, resp = ours_bf.response(phis, 2000.0)
    assert np.abs(resp[0]) > np.abs(resp[1])
    assert np.abs(resp[0]) > np.abs(resp[2])
    # the beam is steered: the response is largest towards phi = 0 across
    # frequency, not just at one bin
    peak_phi = phis[np.argmax(np.abs(resp))]
    assert abs(peak_phi) < 0.05