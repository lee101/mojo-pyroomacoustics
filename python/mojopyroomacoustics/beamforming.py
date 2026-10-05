"""Beamforming from `pyroomacoustics.beamforming`, with upstream's names.

    The free functions (`distance`, `unit_vec2D`, `linear_2D_array`,
    `circular_2D_array`, `square_2D_array`, `H`, `mdot3`, `rake_delay_and_sum`,
    `rake_mvdr_filters`) and the steering-vector and weight methods of
    `Beamformer` run in the compiled Mojo library. `Beamformer` itself keeps
    upstream's constructor signature and its `steering_vector_2D`,
    `steering_vector_2D_from_point`, `response`, `far_field_weights`,
    `rake_delay_and_sum_weights` and `rake_mvdr_filters` contracts.
    """

from __future__ import annotations

import numpy as np

from . import _lib
from .parameters import constants

__all__ = [
    "H",
    "mdot",
    "mdot3",
    "distance",
    "unit_vec2D",
    "linear_2D_array",
    "circular_2D_array",
    "square_2D_array",
    "spiral_2D_array",
    "poisson_2D_array",
    "build_rir_matrix",
    "rake_delay_and_sum",
    "rake_mvdr_filters",
    "MicrophoneArray",
    "Beamformer",
]


# =========================================================================
# Free (non-class-member) functions related to beamformer design
# =========================================================================


def H(A, **kwargs):
    """Returns the conjugate (Hermitian) transpose of a matrix."""
    A = np.ascontiguousarray(A, dtype=np.complex128)
    rows, cols = A.shape
    # complex is interleaved (re, im) float64 pairs, so the kernel buffer is
    # twice the element count
    buf = np.empty(2 * rows * cols, dtype=np.float64)
    _lib.require().mpa_H(_lib.addr(buf), _lib.addr(_lib.c128(A)), rows, cols)
    return buf.view(np.complex128).reshape(cols, rows)


def mdot(*args):
    """Left-to-right associative matrix multiplication of multiple 2D ndarrays."""
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


def mdot3(a, b, c):
    """`a @ b @ c` for 2D complex arrays, evaluated left to right in Mojo.

    Upstream's `mdot` folds over `np.dot`; with three operands this is the same
    expression, and the intermediate is formed in the same association so the
    rounding matches.
    """
    A = np.ascontiguousarray(a, dtype=np.complex128)
    B = np.ascontiguousarray(b, dtype=np.complex128)
    C = np.ascontiguousarray(c, dtype=np.complex128)
    n, m = A.shape
    k = B.shape[1]
    l = C.shape[1]
    work = np.empty(2 * n * k, dtype=np.float64)
    dst = np.empty(2 * n * l, dtype=np.float64)
    _lib.require().mpa_mdot3(
        _lib.addr(_lib.c128(A)), _lib.addr(_lib.c128(B)), _lib.addr(_lib.c128(C)),
        _lib.addr(dst), _lib.addr(work), n, m, k, l,
    )
    return dst.view(np.complex128).reshape(n, l)


def distance(x, y):
    """Computes the distance matrix E.
    E[i,j] = sqrt(sum((x[:,i]-y[:,j])**2)).
    x and y are DxN ndarray containing N D-dimensional vectors.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"both arguments must be (dim, n_points), got {x.shape} and {y.shape}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"both arguments must have the same number of dimensions, got "
            f"{x.shape[0]} and {y.shape[0]}"
        )
    dim = x.shape[0]
    dst = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
    # `r` is an argument the kernel never reads, but a pointer is non-nullable
    # on this ABI, so it is given a real address rather than 0
    scratch = np.empty(1, dtype=np.float64)
    _lib.require().mpa_distance(
        _lib.addr(scratch), _lib.addr(x), _lib.addr(y), _lib.addr(dst),
        dim, x.shape[1], y.shape[1],
    )
    return dst


def unit_vec2D(phi):
    """The unit vector at angle `phi`, as a (2, 1) column."""
    dst = np.zeros(2, dtype=np.float64)
    _lib.require().mpa_unit_vec2D(_lib.addr(dst), float(phi))
    return dst[:, None]


def linear_2D_array(center, M, phi, d):
    """Creates an array of uniformly spaced linear points in 2D.

    Returns
    -------
    ndarray (2, M)
        The array of points.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    dst = np.empty(2 * M, dtype=np.float64)
    u = np.zeros(2, dtype=np.float64)
    _lib.require().mpa_linear_2D_array(
        _lib.addr(dst), _lib.addr(_lib.f64(center)), M, float(phi), float(d),
        _lib.addr(u),
    )
    return dst.reshape(2, M)


def circular_2D_array(center, M, phi0, radius):
    """Create an array of uniformly spaced circular points in 2D.

    Returns
    -------
    ndarray (2, M)
        The array of points.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    dst = np.empty(2 * M, dtype=np.float64)
    _lib.require().mpa_circular_2D_array(
        _lib.addr(dst), _lib.addr(_lib.f64(center)), M, float(phi0), float(radius),
    )
    return dst.reshape(2, M)


def square_2D_array(center, M, N, phi, d):
    """Creates an array of uniformly spaced grid points in 2D.

    Returns
    -------
    ndarray (2, M * N)
        The array of points.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    dst = np.empty(2 * M * N, dtype=np.float64)
    row = np.zeros(2 * M, dtype=np.float64)
    u = np.zeros(2, dtype=np.float64)
    _lib.require().mpa_square_2D_array(
        _lib.addr(dst), _lib.addr(_lib.f64(center)), M, N, float(phi), float(d),
        _lib.addr(row), _lib.addr(u),
    )
    return dst.reshape(2, M * N)


def poisson_2D_array(center, M, d):
    """Create array of 2D positions drawn from a Poisson process.

    Stochastic and seeded from NumPy, so this stays in Python; it is not one of
    the compute-bound kernels.
    """
    from numpy.random import randint, standard_exponential

    center = np.asarray(center, dtype=np.float64).ravel()
    R = d * standard_exponential((2, M)) * (2 * randint(0, 2, (2, M)) - 1)
    R = R.cumsum(axis=1)
    R -= R.mean(axis=1)[:, np.newaxis]
    R += np.array([center]).T
    return R


def spiral_2D_array(center, M, radius=1.0, divi=3, angle=None):
    """Generate an array of points placed on a spiral."""
    num_seg = int(np.ceil(M / divi))
    pos_array_norm = np.linspace(0, radius, num=M, endpoint=False)
    pos_array_angle = (
        np.reshape(
            np.tile(np.pi * 2 * np.arange(divi) / divi, num_seg), (divi, -1), order="F"
        )
        + np.linspace(0, 2 * np.pi / divi, num=num_seg, endpoint=False)[np.newaxis, :]
    )
    pos_array_angle = np.insert(pos_array_angle.flatten("F")[: M - 1], 0, 0)
    if angle is None:
        pos_array_angle += np.random.rand() * np.pi / divi
    else:
        pos_array_angle += angle
    return np.array([pos_array_norm * np.cos(pos_array_angle),
                     pos_array_norm * np.sin(pos_array_angle)])


# =========================================================================
# Classes (microphone array and beamformer related)
# =========================================================================


def rake_delay_and_sum(mics, images, frequencies, fs=16000, attn=True, ff=False,
                       c=None):
    """Delay-and-sum frequency-domain weights over a source's image sources.

    Upstream's `rake_delay_and_sum_weights` normalises by `1/M/(K+1)`, where K
    is `images.shape[1] - 1`, and the `mpa_rake_delay_and_sum_weights` kernel
    reproduces that sum over the per-image steering vectors.

    Parameters
    ----------
    mics: ndarray (dim, M)
        Microphone positions.
    images: ndarray (dim, n_src)
        Image-source positions of the source to focus on.
    frequencies: ndarray (n_freq,)
        The frequency bins the weights are wanted at.
    fs: int
        Sampling frequency. Sets `Beamformer.N` when it is the default.
    attn: bool
        Include the `1/(4 pi d)` attenuation in the steering vectors.
    ff: bool
        Use far-field distances.
    c: float, optional
        Speed of sound. Defaults to `constants.get("c")`.

    Returns
    -------
    ndarray (M, n_freq)
    """
    R = np.ascontiguousarray(mics, dtype=np.float64)
    images = np.ascontiguousarray(images, dtype=np.float64)
    frequencies = np.ascontiguousarray(frequencies, dtype=np.float64).ravel()
    if R.ndim != 2 or images.ndim != 2:
        raise ValueError(
            f"`mics` and `images` must both be (dim, n), got {R.shape} and {images.shape}"
        )
    if R.shape[0] != images.shape[0]:
        raise ValueError(
            f"`mics` and `images` must have the same dimension, got "
            f"{R.shape[0]} and {images.shape[0]}"
        )
    if R.shape[0] not in (2, 3):
        raise ValueError(
            f"the array must be 2-D or 3-D, got {R.shape[0]} dimensions"
        )
    if c is None:
        c = float(constants.get("c"))

    M, n_src, n_freq = R.shape[1], images.shape[1], frequencies.size
    center = R.mean(axis=1, keepdims=True)

    steer = np.empty(2 * n_freq * M * n_src, dtype=np.float64)
    work = np.empty(R.shape[0] * n_src + M * n_src, dtype=np.float64)
    for i, f in enumerate(frequencies):
        block = np.empty(2 * M * n_src, dtype=np.float64)
        _lib.require().mpa_steering_vector_2D_from_point(
            _lib.addr(R), _lib.addr(center.ravel()), _lib.addr(images),
            _lib.addr(block), R.shape[0], M, n_src, float(f), float(c),
            1 if attn else 0, 1 if ff else 0, _lib.addr(work),
        )
        steer[i * 2 * M * n_src : (i + 1) * 2 * M * n_src] = block

    dst = np.empty(2 * M * n_freq, dtype=np.float64)
    _lib.require().mpa_rake_delay_and_sum_weights(
        _lib.addr(dst), _lib.addr(steer), M, n_freq, n_src,
    )
    return dst.view(np.complex128).reshape(M, n_freq)


def rake_mvdr_filters(h, R_xx, K_nq):
    """MVDR time-domain filters from a factored covariance system.

    Upstream's `rake_mvdr_filters` builds `R_xx + K_nq` from a channel matrix,
    Cholesky-factors it, solves for `g`, rescales so the response towards the
    source is exactly one, and returns the SINR. The Cholesky and the triangular
    solves stay in NumPy: they are BLAS-shaped and not one of this port's
    kernels. The two kernels this function does own are
    `mpa_rake_mvdr_solution`, which normalises the solved vector, and
    `mpa_response_sinr`, which reports `num / denom`.

    Parameters
    ----------
    h: ndarray (n,)
        The constraint vector: the channel-matrix column at the beamformer
        delay. Its length is the filter length.
    R_xx: ndarray (n, n)
        Signal covariance.
    K_nq: ndarray (n, n)
        Noise-plus-interference covariance.

    Returns
    -------
    filters: ndarray (n,)
    sinr: float
    """
    h = np.ascontiguousarray(h, dtype=np.float64).ravel()
    R_xx = np.ascontiguousarray(R_xx, dtype=np.float64)
    K_nq = np.ascontiguousarray(K_nq, dtype=np.float64)
    n = h.size
    if R_xx.shape != (n, n) or K_nq.shape != (n, n):
        raise ValueError(
            f"`R_xx` and `K_nq` must both be ({n}, {n}) to match the constraint "
            f"vector, got {R_xx.shape} and {K_nq.shape}"
        )

    # upstream: C = cho_factor(R_xx + K_nq); g = cho_solve(C, h)
    try:
        chol = np.linalg.cholesky(R_xx + K_nq)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "R_xx + K_nq must be positive definite for the MVDR filters"
        ) from exc
    solved = np.linalg.solve(chol.T, np.linalg.solve(chol, h))

    filters = np.zeros(n, dtype=np.float64)
    _lib.require().mpa_rake_mvdr_solution(
        _lib.addr(filters), _lib.addr(h), _lib.addr(_lib.f64(solved.ravel())), 1, n
    )
    sinr = np.zeros(1, dtype=np.float64)
    _lib.require().mpa_response_sinr(
        _lib.addr(sinr), _lib.addr(filters), _lib.addr(R_xx), _lib.addr(K_nq), n
    )
    return filters, float(sinr[0])


class MicrophoneArray(object):
    """Microphone array class."""

    def __init__(self, R, fs, directivity=None):
        self.R = np.array(R)  # array geometry

        if self.dim not in (2, 3):
            dim_mismatch = True
        else:
            dim_mismatch = False

        if self.R.ndim != 2 or dim_mismatch:
            raise ValueError(
                "The location of microphones should be described by an array_like "
                "object with 2 dimensions of shape `(2 or 3, n_mics)` "
                "where `n_mics` is the number of microphones. Each column contains "
                "the location of a microphone."
            )

        self.fs = fs
        self.set_directivity(directivity)
        self.signals = None
        self.center = np.mean(R, axis=1, keepdims=True)

    @property
    def dim(self):
        return self.R.shape[0]

    def __len__(self):
        return self.R.shape[1]

    @property
    def nmic(self):
        """The number of microphones of the array."""
        return len(self)

    @property
    def M(self):
        """The number of microphones of the array."""
        return len(self)

    @property
    def is_directive(self):
        return any(d is not None for d in self.directivity)

    def set_directivity(self, directivities):
        """Sets `self.directivity` as a list with `n_mics` entries."""
        if directivities is None:
            self.directivity = [None] * len(self)
        elif not isinstance(directivities, (list, tuple)):
            self.directivity = [directivities] * len(self)
        elif len(directivities) != len(self):
            raise ValueError("directivity list should be of length `n_mics`")
        else:
            self.directivity = list(directivities)


class Beamformer(MicrophoneArray):
    """At some point, in some nice way, the design methods should also go here.

    Parameters
    ----------
    R: numpy.ndarray
        Mics positions
    fs: int
        Sampling frequency
    N: int, optional
        Length of FFT, i.e. number of FD beamforming weights, equally spaced.
        Defaults to 1024.
    Lg: int, optional
        Length of time-domain filters. Default to N.
    hop: int, optional
        Hop length for frequency domain processing. Default to N/2.
    zpf: int, optional
        Front zero padding length for frequency domain processing. Default is 0.
    zpb: int, optional
        Zero padding length for frequency domain processing. Default is 0.
    """

    def __init__(self, R, fs, N=1024, Lg=None, hop=None, zpf=0, zpb=0):
        MicrophoneArray.__init__(self, R, fs)

        # only support even length (in freq)
        if N % 2 == 1:
            N += 1

        self.N = int(N)

        if Lg is None:
            self.Lg = N
        else:
            self.Lg = int(Lg)

        self.zpf = int(zpf)
        self.zpb = int(zpb)
        self.L = self.N - self.zpf - self.zpb
        if hop is None:
            self.hop = self.L // 2
        else:
            self.hop = hop

        self.frequencies = np.arange(0, self.N // 2 + 1) / self.N * float(self.fs)

        self.weights = None
        self.filters = None

    def __add__(self, y):
        """Concatenates two beamformers together."""
        newR = np.concatenate((self.R, y.R), axis=1)
        return Beamformer(
            newR, self.fs, self.Lg, self.N, hop=self.hop, zpf=self.zpf, zpb=self.zpb
        )

    # ------------------------------------------------------------- steering

    def steering_vector_2D(self, frequency, phi, dist, attn=False):
        """Creates a steering vector for a 2D direction at a given distance."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64)).ravel()
        n_phi = phi.size
        dst = np.empty(2 * self.M * n_phi, dtype=np.float64)
        work = np.empty(2 * n_phi, dtype=np.float64)
        _lib.require().mpa_steering_vector_2D(
            _lib.addr(_lib.f64(self.R)), _lib.addr(_lib.f64(self.center.ravel())),
            _lib.addr(dst), self.M, n_phi, float(frequency), float(dist),
            float(constants.get("c")), 1 if attn else 0,
            _lib.addr(_lib.f64(phi)), _lib.addr(work),
        )
        return dst.view(np.complex128).reshape(self.M, n_phi)

    def steering_vector_2D_from_point(self, frequency, source, attn=True, ff=False):
        """Creates a steering vector for a particular frequency and source.

        Args:
            frequency: the frequency in Hz.
            source: location in cartesian coordinates, (dim,) or (dim, n_src).
            attn: include attenuation factor if True.
            ff: uses far-field distance if true.

        Returns:
            An (M, n_src) ndarray of steering vectors.
        """
        X = np.asarray(source, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        n_src = X.shape[1]
        dim = X.shape[0]
        dst = np.empty(2 * self.M * n_src, dtype=np.float64)
        # scratch: dim*n_src for the source offsets, then n*n_src for the
        # far-field projections
        work = np.empty(dim * n_src + self.M * n_src, dtype=np.float64)
        _lib.require().mpa_steering_vector_2D_from_point(
            _lib.addr(_lib.f64(self.R)), _lib.addr(_lib.f64(self.center.ravel())),
            _lib.addr(_lib.f64(X)), _lib.addr(dst), dim, self.M, n_src,
            float(frequency), float(constants.get("c")),
            1 if attn else 0, 1 if ff else 0, _lib.addr(work),
        )
        return dst.view(np.complex128).reshape(self.M, n_src)

    def response(self, phi_list, frequency):
        """Beampattern response at `frequency` for the directions `phi_list`.

        Returns ``(frequency, response)`` where response has one entry per
        direction, matching upstream's ``np.dot(H(w), A)``.
        """
        phi_list = np.atleast_1d(np.asarray(phi_list, dtype=np.float64)).ravel()
        n_phi = phi_list.size
        i_freq = int(np.argmin(np.abs(self.frequencies - frequency)))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed first."
            )

        A = self.steering_vector_2D(
            self.frequencies[i_freq], phi_list, constants.get("ffdist")
        )
        w = np.ascontiguousarray(self.weights[:, i_freq])

        buf = np.empty(2 * n_phi, dtype=np.float64)
        _lib.require().mpa_response(_lib.addr(buf), _lib.addr(_lib.c128(w)),
                                    _lib.addr(_lib.c128(A)), self.M, n_phi)
        return self.frequencies[i_freq], buf.view(np.complex128)

    # -------------------------------------------------------------- weights

    def far_field_weights(self, phi):
        """This method computes weight for a far field at infinity.

        phi: direction of the beam, in radians.
        """
        dst = np.empty(2 * self.M * self.frequencies.size, dtype=np.float64)
        proj = np.empty(self.M, dtype=np.float64)
        _lib.require().mpa_far_field_projection(
            _lib.addr(proj), _lib.addr(_lib.f64(self.R)),
            _lib.addr(_lib.f64(self.center.ravel())), self.dim, self.M, float(phi),
        )
        _lib.require().mpa_far_field_weights(
            _lib.addr(dst), _lib.addr(proj),
            _lib.addr(_lib.f64(self.frequencies)), self.M,
            self.frequencies.size, float(constants.get("c")),
        )
        self.weights = dst.view(np.complex128).reshape(
            self.M, self.frequencies.size
        )

    def weights_from_filters(self):
        """The frequency-domain weights are the conjugate rFFT of the filters."""
        self.weights = np.zeros((self.M, self.N // 2 + 1), dtype=np.complex128)
        for m in range(self.M):
            self.weights[m] = np.conj(np.fft.rfft(self.filters[m], n=self.N))

    def rake_delay_and_sum_weights(self, source, interferer=None, R_n=None, attn=True, ff=False):
        """Delay-and-sum beamformer weights over the source's image sources."""
        images = np.asarray(getattr(source, "images", source), dtype=np.float64)
        if images.ndim == 1:
            images = images[:, np.newaxis]
        n_src = images.shape[1]
        n_freq = self.frequencies.size

        steer = np.empty(2 * n_freq * self.M * n_src, dtype=np.float64)
        for i, f in enumerate(self.frequencies):
            block = np.empty(2 * self.M * n_src, dtype=np.float64)
            work = np.empty(images.shape[0] * n_src + self.M * n_src, dtype=np.float64)
            _lib.require().mpa_steering_vector_2D_from_point(
                _lib.addr(_lib.f64(self.R)),
                _lib.addr(_lib.f64(self.center.ravel())),
                _lib.addr(_lib.f64(images)), _lib.addr(block), images.shape[0],
                self.M, n_src, float(f), float(constants.get("c")),
                1 if attn else 0, 1 if ff else 0, _lib.addr(work),
            )
            steer[i * 2 * self.M * n_src : (i + 1) * 2 * self.M * n_src] = block

        dst = np.empty(2 * self.M * n_freq, dtype=np.float64)
        _lib.require().mpa_rake_delay_and_sum_weights(
            _lib.addr(dst), _lib.addr(steer), self.M, n_freq, n_src,
        )
        self.weights = dst.view(np.complex128).reshape(self.M, n_freq)

    def rake_mvdr_filters(self, h, R_xx, K_nq):
        """MVDR time-domain filters; see the module-level `rake_mvdr_filters`.

        Kept as a method so `Beamformer.rake_mvdr_filters` exists with the name
        upstream gives it.
        """
        filters, sinr = rake_mvdr_filters(h, R_xx, K_nq)
        self.filters = filters.reshape(1, filters.size)
        return self.filters, sinr


def build_rir_matrix(mics, sources, Lg, Fs, epsilon=5e-3, unit_damping=False):
    """Build the channel matrix for many sources and microphones.

    Parameters
    ----------
    mics: ndarray
        A dim-by-M array where each column is a microphone position.
    sources: list of pyroomacoustics.SoundSource
        The sources to build the matrix for.
    Lg: int
        The length of the beamforming filters.
    Fs: int
        The sampling frequency.
    epsilon: float, optional
        Minimum decay of the sinc before truncation.
    unit_damping: bool, optional
        Whether the wall damping parameters are used.

    Returns
    -------
    ndarray (M*Lg, S*(Lg+Lh-1))
        The channel matrix H.
    """
    from .utilities import convmtx_dot, low_pass_dirac

    mics = np.ascontiguousarray(mics, dtype=np.float64)
    if mics.ndim != 2:
        raise ValueError(f"`mics` must be (dim, n_mics), got shape {mics.shape}")
    c = float(constants.get("c"))

    for s in sources:
        if s.damping.shape[0] > 1:
            raise NotImplementedError("Multiple frequency bands not supported yet")

    d_min = np.inf
    d_max = 0.0
    dmp_max = 0.0
    for s in range(len(sources)):
        dist_mat = distance(mics, sources[s].images)
        if unit_damping is True:
            dmp_max = np.maximum((1.0 / (4 * np.pi * dist_mat)).max(), dmp_max)
        else:
            dmp_max = np.maximum(
                (sources[s].damping[0, np.newaxis, :] / (4 * np.pi * dist_mat)).max(),
                dmp_max,
            )
        d_min = np.minimum(dist_mat.min(), d_min)
        d_max = np.maximum(dist_mat.max(), d_max)

    t_max = d_max / c
    t_min = d_min / c
    offset = dmp_max / (np.pi * Fs * epsilon)
    Lh = int((t_max - t_min + 2 * offset) * float(Fs))

    L = Lg + Lh - 1
    H = np.zeros((Lg * mics.shape[1], len(sources) * L))

    # unit-length signal for the matrix-free convolution: `convmtx(h, Lg).T` is
    # exactly `convmtx_dot(h, e)[:Lg]` for the unit vector e, so the Toeplitz
    # matrix is never formed.
    unit = np.zeros(Lg, dtype=np.float64)
    unit[0] = 1.0

    for s in range(len(sources)):
        for r in np.arange(mics.shape[1]):
            dist = sources[s].distance(mics[:, r])
            time = dist / c - t_min + offset
            if unit_damping is True:
                dmp = 1.0 / (4 * np.pi * dist)
            else:
                dmp = sources[s].damping[0, :] / (4 * np.pi * dist)
            h = low_pass_dirac(time[:, np.newaxis], dmp[:, np.newaxis], Fs, Lh).sum(
                axis=0
            )
            H[r * Lg : (r + 1) * Lg, s * L : (s + 1) * L] = convmtx_dot(h, unit)

    return H