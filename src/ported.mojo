"""mojo-pyroomacoustics: the compute-bound kernels of pyroomacoustics.

Functions are emitted in upstream's source order and grouped by upstream's
module, so this file reads top-to-bottom alongside the Python package it came
from:

    pyroomacoustics/utilities.py         low_pass_dirac, fractional_delay,
                                          fractional_delay_filter_bank,
                                          levinson, autocorr, lpc, goertzel,
                                          convmtx
    pyroomacoustics/beamforming.py       distance, unit_vec2D,
                                          linear_2D_array, circular_2D_array,
                                          square_2D_array, H, mdot,
                                          steering_vector_2D(_from_point),
                                          response, far_field_weights,
                                          rake_delay_and_sum_weights,
                                          rake_mvdr_filters,
                                          rake_distortionless_filters
    pyroomacoustics/libroom_src/         rir_builder, delay_sum,
    rir_builder.cpp                         fractional_delay
    pyroomacoustics/simulation/ism.py    image amplitudes and the RIR assembly

Three conventions hold throughout, all of them forced by the compiler:

- **Complex values are interleaved (re, im) float64 pairs.** `Pointer` to
  `ComplexFloat64` is a `Pointer[ComplexSIMD[.float64, 1]]` and Mojo's load and
  store are SIMD-typed, so they reject it. Complex arrays therefore cross the
  ABI as plain float64 and are viewed as `numpy.complex128` with no copy.
- **Nothing allocates.** Every routine takes its scratch from the caller, so
  lifetimes are Python's problem and there is nothing to leak.
- **Buffers cross as `Int` addresses.** `@export` refuses parametric functions
  and a pointer with an inferred origin is parametric, so `p(addr)` rebuilds the
  pointer inside the wrapper.

Parallelism: `rir_builder` and `delay_sum` scatter into overlapping output, so
partitioning the *inputs* would race. Both partition the *output* instead --
every worker walks every arrival but only writes the slice of the RIR it owns,
which needs no private buffers and no reduce pass. Verified to reproduce the
serial result bit for bit in `tests/test_rir_kernels.py`.
"""

from max.algorithm import parallelize
from std.math import cos, exp, floor, pi, sin, sqrt
from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()

# Below this many scatter-taps the thread pool costs more than it saves.
comptime PAR_THRESHOLD = 1 << 16


def p(addr: Int) -> Ptr:
    """Rebuild a mutable float64 pointer from an ABI address."""
    return Ptr(unsafe_from_address=addr)


def sinc(x: Float64) -> Float64:
    """`np.sinc`: sin(pi x) / (pi x), with the removable singularity at 0 filled."""
    if x == 0.0:
        return 1.0
    var pix = pi * x
    return sin(pix) / pix


# =========================================================================
# pyroomacoustics/utilities.py
# =========================================================================


def convmtx_dot(dst: Ptr, x: Ptr, h: Ptr, n: Int, m: Int):
    """`dst[:n] = convmtx(h, m) @ x`, without materialising the Toeplitz matrix.

    Upstream's docstring is the contract: `np.dot(convmtx(x, n), v)` for a `v`
    of length `n` equals `np.convolve(x, v)`. `convmtx(h, m)` is
    `(m + n - 1, n)`, so a full-length product needs an `x` of `m + n - 1`.

    That contract is what `build_rir_matrix` leans on: it assigns
    `convmtx(h, Lg).T` into a block of the channel matrix, which is this same
    product read the other way round.

    `x` MUST be at least `n` long: the kernel indexes `x[i - k]` for
    `0 <= i < n` and `0 <= k < m`, so `x[i - k]` reaches as high as `n - 1`
    and as low as `-(m - 1)`, and negative indices are treated as zero. There
    is no length argument for `x` here, so a caller that passes a shorter `x`
    reads past it; the Python wrapper `convmtx_dot` checks.
    """
    for i in range(n):
        var acc = 0.0
        for k in range(m):
            var idx = i - k
            if idx >= 0:
                acc += h.unsafe_load(k) * x.unsafe_load(idx)
        dst.unsafe_store(i, acc)


def low_pass_dirac(dst: Ptr, t0: Ptr, alpha: Ptr, n_rows: Int, n: Int, fs: Float64):
    """Creates a vector containing a lowpass Dirac of duration T sampled at Fs
    with delay t0 and attenuation alpha.

    Upstream is `alpha * np.sinc(np.arange(N) - Fs * t0)`, vectorised over the
    leading axis of `t0`/`alpha`. The elementwise loop is kept as it is upstream.
    """
    for t in range(n_rows):
        var a = alpha.unsafe_load(t)
        var tau = fs * t0.unsafe_load(t)
        for i in range(n):
            dst.unsafe_store(t * n + i, a * sinc(Float64(i) - tau))


def fractional_delay_bank(
    dst: Ptr, delays: Ptr, n_delays: Int, l: Int, filter_len: Int, min_delay: Float64
):
    """Creates a fractional delay filter bank of windowed sinc filters.

    Upstream subtracts the minimum delay so every delay is positive, splits into
    integer `di` and fractional `df` parts, and scatters
    `hanning(L)[k] * sinc(k - df - (L-1)/2)` into row `t` at index `k + di`.
    The scatter and the arithmetic are both reproduced; the only change is that
    the index is clamped to the row, because upstream relies on its `indices`
    array being in range by construction.
    """
    var half = Float64(l - 1) / 2.0
    for t in range(n_delays):
        var d = delays.unsafe_load(t) - min_delay
        var di = floor(d)
        var df = d - di
        for k in range(l):
            var w = 0.5 - 0.5 * cos(2.0 * pi * Float64(k) / Float64(l - 1))
            var idx = k + Int(di)
            if 0 <= idx < filter_len:
                dst.unsafe_store(t * filter_len + idx, w * sinc(Float64(k) - df - half))


def levinson(r: Ptr, b: Ptr, x: Ptr, p_len: Int, n_rhs: Int, a: Ptr, arev: Ptr):
    """Solve Rx = b where R is a Hermitian Toeplitz matrix, using the generalized
    Levinson recursion as described in M.H. Hayes, Statistical Signal Processing
    and Modelling, p. 268.

    `r` is the first column of R (length p_len+1), `b` is p_len x n_rhs, `x`
    receives the solution. `a` and `arev` are scratch of p_len+1 float64 each,
    supplied by the caller because nothing here allocates.

    Upstream is written with `np.flipud(a[::-1])` and `np.conj(a[::-1])`; for the
    real autocorrelation problem that `levinson` is actually called with, the
    conjugate-reverse is the plain reverse, so the two are one array here. The
    loop order and every division are unchanged.
    """
    # a = [1]
    a.unsafe_store(0, 1.0)
    for k in range(1, p_len + 1):
        a.unsafe_store(k, 0.0)

    # x = b[0, :] / r[0]
    for c in range(n_rhs):
        x.unsafe_store(c, b.unsafe_load(c) / r.unsafe_load(0))

    var epsilon = r.unsafe_load(0)

    var j = 1
    while j < p_len:
        # a holds the predictor of order j-1, in a[0..j-1].
        # g = sum(conj(r[1:j+1]) * a[::-1]) = sum_{k=1..j} r[k] * a[j-k]
        var g = 0.0
        for k in range(1, j + 1):
            g += r.unsafe_load(k) * a.unsafe_load(j - k)

        var gamma = -g / epsilon

        # a_new = concat((a, 0)) + gamma * concat((0, conj(a[::-1])))
        # so a_new[0] = a[0], a_new[k] = a[k] + gamma * a[j-k] for 1 <= k < j,
        # and a_new[j] = gamma * a[0]. `arev` holds the gamma*a[j-k] tail so the
        # update reads the old `a` rather than the half-written new one.
        arev.unsafe_store(0, 0.0)
        for k in range(1, j):
            arev.unsafe_store(k, gamma * a.unsafe_load(j - k))
        for k in range(j):
            a.unsafe_store(k, a.unsafe_load(k) + arev.unsafe_load(k))
        a.unsafe_store(j, gamma * a.unsafe_load(0))

        epsilon = epsilon * (1.0 - gamma * gamma)

        # delta = dot(conj(r[1:j+1]), flipud(x)) sums over the lag axis, giving one
        # value per right-hand side: delta[c] = sum_{k=1..j} r[k] * x[j-k, c].
        # x currently holds j rows, x[0..j-1].
        for c in range(n_rhs):
            var delta = 0.0
            for k in range(1, j + 1):
                delta += r.unsafe_load(k) * x.unsafe_load((j - k) * n_rhs + c)

            # x_new = concat((x, 0)) + q * conj(a_new[::-1]) for
            # q = (b[j, c] - delta[c]) / epsilon; a_new[::-1][k] = a_new[j - k].
            var q = (b.unsafe_load(j * n_rhs + c) - delta) / epsilon
            for k in range(j + 1):
                var base = 0.0
                if k < j:
                    base = x.unsafe_load(k * n_rhs + c)
                x.unsafe_store(k * n_rhs + c, base + q * a.unsafe_load(j - k))

        j += 1


def autocorr_time(x: Ptr, r: Ptr, length: Int, p_max: Int):
    """Autocorrelation of a real signal up to lag `p_max`, time domain.

    Upstream `autocorr(..., method="time")` is `r[m] = np.dot(x[:L-m], x[m:])`.
    The biased/unbiased scaling is applied by the caller, so this writes the raw
    numerator, matching the `numpy` and `fft` branches exactly.
    """
    for m in range(p_max + 1):
        var acc = 0.0
        for i in range(length - m):
            acc += x.unsafe_load(i) * x.unsafe_load(i + m)
        r.unsafe_store(m, acc)


def lpc_from_autocorr(r: Ptr, x: Ptr, p_len: Int, a: Ptr, arev: Ptr):
    """`lpc`: solve the Yule-Walker equations for the LPC coefficients.

    Upstream is `levinson(r[:p], r[1:])`: the Toeplitz matrix is built from the
    first `p+1` autocorrelation values and the right-hand side is `r[1:p+1]`.
    """
    levinson(r, r + 1, x, p_len, 1, a, arev)


def goertzel(x: Ptr, dst_o: Ptr, length: Int, k: Int):
    """Goertzel algorithm to compute DFT coefficient `k` of a real signal.

    Upstream runs `scipy.signal.lfilter(b, a, x)` with
    `a = [1, -2 cos(2 pi f), 1]`, which is this second-order recursion written
    out in the time domain; the final combination is unchanged.
    """
    var f = Float64(k) / Float64(length)
    # lfilter solves sum_k a[k] s[n-k] = x[n], so with
    # a = [1, -2 cos(2 pi f), 1] the recursion is s[n] = x[n] + 2 cos s[n-1] - s[n-2].
    var coeff = 2.0 * cos(2.0 * pi * f)
    var s1 = 0.0
    var s2 = 0.0
    for i in range(length):
        var s = x.unsafe_load(i) + coeff * s1 - s2
        s2 = s1
        s1 = s
    # y = exp(2 pi i f) s[-1] - s[-2]
    dst_o.unsafe_store(0, cos(2.0 * pi * f) * s1 - s2)
    dst_o.unsafe_store(1, sin(2.0 * pi * f) * s1)


# =========================================================================
# pyroomacoustics/beamforming.py -- free functions
# =========================================================================


def distance(r: Ptr, x: Ptr, y: Ptr, dst: Ptr, dim: Int, nx: Int, ny: Int):
    """Computes the distance matrix E.
    E[i,j] = sqrt(sum((x[:,i]-y[:,j])**2)).
    x and y are DxN ndarray containing N D-dimensional vectors.

    Upstream builds the difference with a broadcast and reduces over `axis=0`;
    the same sum is walked explicitly over the coordinate axis here.
    """
    for i in range(nx):
        for j in range(ny):
            var acc = 0.0
            for d in range(dim):
                var diff = x.unsafe_load(d * nx + i) - y.unsafe_load(d * ny + j)
                acc += diff * diff
            dst.unsafe_store(i * ny + j, sqrt(acc))


def unit_vec2D(dst: Ptr, phi: Float64):
    """`unit_vec2D(phi) = [[cos(phi), sin(phi)]].T`, shape (2, 1)."""
    dst.unsafe_store(0, cos(phi))
    dst.unsafe_store(1, sin(phi))


def linear_2D_array(dst: Ptr, center: Ptr, m: Int, phi: Float64, d: Float64, u: Ptr):
    """Creates an array of uniformly spaced linear points in 2D, shape (2, M).

    Upstream is `center[:, None] + d * (arange(M) - (M-1)/2) * u` with
    `u = unit_vec2D(phi)`; the arithmetic order is kept.
    """
    unit_vec2D(u, phi)
    var u0 = u.unsafe_load(0)
    var u1 = u.unsafe_load(1)
    for i in range(m):
        var t = d * (Float64(i) - (Float64(m) - 1.0) / 2.0)
        dst.unsafe_store(i, center.unsafe_load(0) + t * u0)
        dst.unsafe_store(m + i, center.unsafe_load(1) + t * u1)


def circular_2D_array(dst: Ptr, center: Ptr, m: Int, phi0: Float64, radius: Float64):
    """Create an array of uniformly spaced circular points in 2D, shape (2, M)."""
    for i in range(m):
        var phi = Float64(i) * 2.0 * pi / Float64(m)
        dst.unsafe_store(i, center.unsafe_load(0) + radius * cos(phi + phi0))
        dst.unsafe_store(m + i, center.unsafe_load(1) + radius * sin(phi + phi0))


def square_2D_array(
    dst: Ptr, center: Ptr, m: Int, n: Int, phi: Float64, d: Float64, row: Ptr, u: Ptr
):
    """Creates an array of uniformly spaced grid points in 2D, shape (2, M*N).

    Upstream lays out grid row `i` by calling `linear_2D_array` on the `i`-th
    point of a perpendicular linear array; `row` is that array's scratch and `u`
    the two scratch unit vectors.
    """
    # the row centres run along phi + pi/2 ...
    linear_2D_array(row, center, m, phi + pi / 2.0, d, u)
    # ... and each row itself runs along phi, starting at its row centre
    unit_vec2D(u, phi)
    var u0 = u.unsafe_load(0)
    var u1 = u.unsafe_load(1)
    for i in range(m):
        var cx = row.unsafe_load(i)
        var cy = row.unsafe_load(m + i)
        for j in range(n):
            var t = d * (Float64(j) - (Float64(n) - 1.0) / 2.0)
            dst.unsafe_store(i * n + j, cx + t * u0)
            dst.unsafe_store(m * n + i * n + j, cy + t * u1)


def H(dst: Ptr, a: Ptr, rows: Int, cols: Int):
    """Returns the conjugate (Hermitian) transpose of a matrix.

    `a` is rows x cols interleaved complex; `dst` is cols x rows. A conjugate is
    only meaningful for complex input, but the operation is the same either way
    and matches `np.transpose(A).conj()`.
    """
    for i in range(rows):
        for j in range(cols):
            var v = a.unsafe_load(2 * (i * cols + j))
            dst.unsafe_store(2 * (j * rows + i), v)
            dst.unsafe_store(2 * (j * rows + i) + 1, -a.unsafe_load(2 * (i * cols + j) + 1))


def mdot3(a: Ptr, b: Ptr, c: Ptr, dst: Ptr, work: Ptr, n: Int, m: Int, k: Int, l: Int):
    """`dst[n, l] = a[n, m] @ b[m, k] @ c[k, l]`, all complex interleaved.

    Upstream `mdot(H(w), R_nq_inv, A_s)` chains `np.dot` left to right, so the
    intermediate a @ b is formed first and the association is kept here for the
    same rounding. `work` is caller-supplied scratch of n*k interleaved complex,
    which is what `np.dot` allocates between the two calls upstream.
    """
    for i in range(n):
        for j in range(k):
            var sr = 0.0
            var si = 0.0
            for t in range(m):
                var ar = a.unsafe_load(2 * (i * m + t))
                var ai = a.unsafe_load(2 * (i * m + t) + 1)
                var br = b.unsafe_load(2 * (t * k + j))
                var bi = b.unsafe_load(2 * (t * k + j) + 1)
                sr += ar * br - ai * bi
                si += ar * bi + ai * br
            work.unsafe_store(2 * (i * k + j), sr)
            work.unsafe_store(2 * (i * k + j) + 1, si)

    for i in range(n):
        for j in range(l):
            var sr = 0.0
            var si = 0.0
            for t in range(k):
                var ar = work.unsafe_load(2 * (i * k + t))
                var ai = work.unsafe_load(2 * (i * k + t) + 1)
                var br = c.unsafe_load(2 * (t * l + j))
                var bi = c.unsafe_load(2 * (t * l + j) + 1)
                sr += ar * br - ai * bi
                si += ar * bi + ai * br
            dst.unsafe_store(2 * (i * l + j), sr)
            dst.unsafe_store(2 * (i * l + j) + 1, si)


# =========================================================================
# pyroomacoustics/beamforming.py -- steering vectors and weights
# =========================================================================


def steering_vector_2D_from_point(
    r: Ptr,
    center: Ptr,
    x: Ptr,
    dst: Ptr,
    dim: Int,
    n: Int,
    n_src: Int,
    freq: Float64,
    c: Float64,
    attn: Bool,
    ff: Bool,
    work: Ptr,
):
    """Creates a steering vector for a particular frequency and source.

    `x` is the source position(s) as (dim, n_src); `dst` is the steering matrix
    (n, n_src) interleaved. `work` is scratch of dim*n_src + n_src float64.

    Upstream normalises `p = X - center`, projects `D = -R.T @ p` and subtracts
    the per-column minimum in the far-field branch, and uses
    `distance(R, X)` otherwise. Then `phase = exp(-i omega D / c)` and, with
    attenuation, `1/(4 pi) / D * phase`.
    """
    var omega = 2.0 * pi * freq

    if ff:
        # unit vectors pointing towards sources: p = X - center, then divided by
        # a single norm over the whole array (upstream's np.linalg.norm(p))
        var norm2 = 0.0
        for d in range(dim):
            for s in range(n_src):
                var pd = x.unsafe_load(d * n_src + s) - center.unsafe_load(d)
                work.unsafe_store(d * n_src + s, pd)
                norm2 += pd * pd
        var norm = sqrt(norm2)

        # The projected microphone distances on the unit vectors
        var proj = work + dim * n_src
        for m in range(n):
            for s in range(n_src):
                var acc = 0.0
                for d in range(dim):
                    acc -= r.unsafe_load(d * n + m) * work.unsafe_load(d * n_src + s)
                proj.unsafe_store(m * n_src + s, acc / norm)

        # subtract the minimum; upstream writes "in each column" but takes
        # np.min over the whole array, and the whole array is what it subtracts
        var dmin = proj.unsafe_load(0)
        for i in range(1, n * n_src):
            dmin = min(dmin, proj.unsafe_load(i))

        for i in range(n * n_src):
            var dd = proj.unsafe_load(i) - dmin
            var arg = -omega * dd / c
            if attn:
                var g = 1.0 / (4.0 * pi) / dd
                dst.unsafe_store(2 * i, g * cos(arg))
                dst.unsafe_store(2 * i + 1, g * sin(arg))
            else:
                dst.unsafe_store(2 * i, cos(arg))
                dst.unsafe_store(2 * i + 1, sin(arg))
    else:
        for m in range(n):
            for s in range(n_src):
                var acc = 0.0
                for dd in range(dim):
                    var diff = r.unsafe_load(dd * n + m) - x.unsafe_load(dd * n_src + s)
                    acc += diff * diff
                var dm = sqrt(acc)
                var arg = -omega * dm / c
                if attn:
                    var g = 1.0 / (4.0 * pi) / dm
                    dst.unsafe_store(2 * (m * n_src + s), g * cos(arg))
                    dst.unsafe_store(2 * (m * n_src + s) + 1, g * sin(arg))
                else:
                    dst.unsafe_store(2 * (m * n_src + s), cos(arg))
                    dst.unsafe_store(2 * (m * n_src + s) + 1, sin(arg))


def steering_vector_2D(
    r: Ptr,
    center: Ptr,
    dst: Ptr,
    n: Int,
    n_phi: Int,
    freq: Float64,
    dist: Float64,
    c: Float64,
    attn: Bool,
    phis: Ptr,
    work: Ptr,
):
    """`steering_vector_2D(frequency, phi_list, dist)`.

    Upstream builds `X = dist * [cos(phi), sin(phi)] + center` for every angle and
    then evaluates the same `distance(R, X)` sum as the from-point variant, so
    this fills `X` and runs the identical loop rather than duplicating it.
    `phis` is the caller's angle list, `work` is scratch of 2*n_phi.
    """
    # X is (2, n_phi): the cos row then the sin row, both offset by the centre.
    for i in range(n_phi):
        work.unsafe_store(i, dist * cos(phis.unsafe_load(i)) + center.unsafe_load(0))
        work.unsafe_store(n_phi + i, dist * sin(phis.unsafe_load(i)) + center.unsafe_load(1))

    var omega = 2.0 * pi * freq
    for m in range(n):
        for i in range(n_phi):
            var acc = 0.0
            for dd in range(2):
                var diff = r.unsafe_load(dd * n + m) - work.unsafe_load(dd * n_phi + i)
                acc += diff * diff
            var dm = sqrt(acc)
            var arg = -omega * dm / c
            if attn:
                var g = 1.0 / (4.0 * pi) / dm
                dst.unsafe_store(2 * (m * n_phi + i), g * cos(arg))
                dst.unsafe_store(2 * (m * n_phi + i) + 1, g * sin(arg))
            else:
                dst.unsafe_store(2 * (m * n_phi + i), cos(arg))
                dst.unsafe_store(2 * (m * n_phi + i) + 1, sin(arg))


def response(dst: Ptr, w: Ptr, a: Ptr, n: Int, n_phi: Int):
    """`np.dot(H(weights[:, i]), steering_vector)`.

    Upstream takes `H(weights[:, i])`, which collapses the weight column to a
    (M,) vector, and dots it with the (M, n_phi) steering matrix, so the result
    has one beampattern response per direction. `w` is the (M,) weight column and
    `a` the (M, n_phi) steering matrix, both interleaved complex.
    """
    for p in range(n_phi):
        var sr = 0.0
        var si = 0.0
        for i in range(n):
            var wr = w.unsafe_load(2 * i)
            var wi = -w.unsafe_load(2 * i + 1)
            var ar = a.unsafe_load(2 * (i * n_phi + p))
            var ai = a.unsafe_load(2 * (i * n_phi + p) + 1)
            sr += wr * ar - wi * ai
            si += wr * ai + wi * ar
        dst.unsafe_store(2 * p, sr)
        dst.unsafe_store(2 * p + 1, si)


def far_field_weights(dst: Ptr, proj: Ptr, frequencies: Ptr, m: Int, n_freq: Int, c: Float64):
    """Computes weights for a far field at infinity.

    Upstream projects the array onto the unit beam direction,
    `proj = (u.T @ (R - center))[0]`, then `proj -= proj.max()` so the filter is
    causal, and sets `weights[m, i] = exp(2 i pi f_i proj[m] / c)`. The caller
    passes `proj` already normalised and shifted, since that is the part that
    needs the array geometry. `dst` is (m, n_freq) interleaved.
    """
    for i in range(n_freq):
        var f = frequencies.unsafe_load(i)
        for j in range(m):
            var arg = 2.0 * pi * f * proj.unsafe_load(j) / c
            dst.unsafe_store(2 * (j * n_freq + i), cos(arg))
            dst.unsafe_store(2 * (j * n_freq + i) + 1, sin(arg))


def far_field_projection(dst: Ptr, r: Ptr, center: Ptr, dim: Int, m: Int, phi: Float64):
    """`proj = (unit_vec2D(phi).T @ (R - center))[0]`, shifted to be causal.

    This is the geometry half of `far_field_weights`; it is split out because it
    depends on the array and the kernel above does not.
    """
    var u0 = cos(phi)
    var u1 = sin(phi)
    var pmax = -1e300
    for j in range(m):
        var acc = u0 * (r.unsafe_load(j) - center.unsafe_load(0))
        if dim > 1:
            acc += u1 * (r.unsafe_load(m + j) - center.unsafe_load(1))
        dst.unsafe_store(j, acc)
        pmax = max(pmax, acc)
    # normalize the first arriving signal to ensure a causal filter
    for j in range(m):
        dst.unsafe_store(j, dst.unsafe_load(j) - pmax)


def rake_delay_and_sum_weights(dst: Ptr, steer: Ptr, m: Int, n_freq: Int, n_src: Int):
    """`rake_delay_and_sum_weights`: w[:, i] = 1/M/(K+1) * sum_k W[:, k].

    Upstream sums the steering vectors over the image sources and normalises by
    `1/M/(K+1)`. `steer` holds them as (n_freq, m, n_src) interleaved, `dst` is
    (m, n_freq).
    """
    for i in range(n_freq):
        for j in range(m):
            var mr = 0.0
            var mi = 0.0
            for k in range(n_src):
                mr += steer.unsafe_load(2 * ((i * m + j) * n_src + k))
                mi += steer.unsafe_load(2 * ((i * m + j) * n_src + k) + 1)
            var g = 1.0 / Float64(m) / Float64(n_src)
            dst.unsafe_store(2 * (j * n_freq + i), g * mr)
            dst.unsafe_store(2 * (j * n_freq + i) + 1, g * mi)


def cov_gram(dst: Ptr, a: Ptr, rows: Int, cols: Int):
    """`dst[rows, rows] = a @ a.H` for a real (rows, cols) matrix.

    Upstream writes these as `np.dot(Hpr, Hpr.T)` and `np.dot(Hnc, Hnc.T)` in
    `rake_mvdr_filters` and `rake_distortionless_filters`; the pair is spelled as
    one symmetric Gram kernel here, so only the lower triangle is accumulated
    and then mirrored. Computing the whole rows x rows block and discarding the
    upper half would double the memory traffic for an identical result.
    """
    for i in range(rows * rows):
        dst.unsafe_store(i, 0.0)
    for r in range(rows):
        var row = a + r * cols
        var i = 0
        while i <= r:
            var rowi = a + i * cols
            var acc = 0.0
            var k = 0
            var vacc = SIMD[DType.float64, W](0.0)
            while k + W <= cols:
                vacc += row.unsafe_load[width=W](k) * rowi.unsafe_load[width=W](k)
                k += W
            acc = vacc.reduce_add()
            while k < cols:
                acc += row.unsafe_load(k) * rowi.unsafe_load(k)
                k += 1
            dst.unsafe_store(r * rows + i, acc)
            i += 1
    for i in range(rows):
        for j in range(i + 1, rows):
            dst.unsafe_store(i * rows + j, dst.unsafe_load(j * rows + i))


def rake_mvdr_solution(dst: Ptr, h: Ptr, c: Ptr, m: Int, l: Int):
    """Back-substitute the MVDR system and normalise for a distortionless
    response towards the source, as `rake_mvdr_filters` does.

    Upstream computes `g = cho_solve(C, h)` with `C = cho_factor(R_xx + K_nq)`,
    then `g /= np.inner(h, g)`. Here `c` is the already-factored, already-solved
    `L^-1 h`, so the caller's two triangular solves stand in for the `cho_solve`.
    """
    var num = 0.0
    for i in range(m * l):
        num += h.unsafe_load(i) * c.unsafe_load(i)
    for i in range(m * l):
        dst.unsafe_store(i, c.unsafe_load(i) / num)


def response_sinr(dst: Ptr, g: Ptr, r_xx: Ptr, k_nq: Ptr, n: Int):
    """`num / denom` of `rake_mvdr_filters`, the SINR of the computed filters.

    Upstream returns `np.inner(g.T, R_xx g) / np.inner(g.T K_nq, g)`.
    """
    var num = 0.0
    var den = 0.0
    for i in range(n):
        var gr = g.unsafe_load(i)
        var rg = 0.0
        var kg = 0.0
        for t in range(n):
            rg += r_xx.unsafe_load(i * n + t) * g.unsafe_load(t)
            kg += k_nq.unsafe_load(i * n + t) * g.unsafe_load(t)
        num += gr * rg
        den += gr * kg
    dst.unsafe_store(0, num / den)


# =========================================================================
# pyroomacoustics/libroom_src/rir_builder.cpp -- the C++ RIR kernels
# =========================================================================


def build_sinc_lut(lut: Ptr, fdl: Int, lut_gran: Int):
    """Sinc look-up table of size (fdl + 2) * lut_gran.

    Upstream walks `n` from `-fdl2 - 1` in steps of `1/lut_gran` and stores
    `sin(pi n) / (pi n)`, forcing the removable singularity at exactly zero to
    1.0. The same walk is reproduced so the table matches to float64.
    """
    var fdl2 = Float64((fdl - 1) // 2)
    var lut_size = (fdl + 2) * lut_gran
    var lut_delta = 1.0 / Float64(lut_gran)
    var n = -fdl2 - 1.0
    for idx in range(lut_size):
        if n == 0.0:
            lut.unsafe_store(idx, 1.0)
        else:
            var pix = pi * n
            lut.unsafe_store(idx, sin(pix) / pix)
        n += lut_delta


def build_hann(hann: Ptr, fdl: Int):
    """Hann window of length `fdl`, matching `rir_builder.cpp` exactly.

    Upstream is `0.5 - 0.5 cos(pi * 2 * idx / (fdl - 1))`, the symmetric window.
    """
    for idx in range(fdl):
        hann.unsafe_store(idx, 0.5 - 0.5 * cos((pi * 2.0 * Float64(idx)) / Float64(fdl - 1)))


def rir_builder(
    rir: Ptr,
    time: Ptr,
    alpha: Ptr,
    lut: Ptr,
    hann: Ptr,
    n_times: Int,
    rir_len: Int,
    fs: Int,
    fdl: Int,
    lut_gran: Int,
    n_workers: Int,
):
    """Build the RIR from image-source times and attenuations.

    Upstream decomposes each arrival into integer and fractional sample, looks
    the fractional part up in a precomputed sinc LUT with linear interpolation
    between `lut_gran_off` and `lut_gran_off + 1`, multiplies by the Hann window
    and the image attenuation, and accumulates into `rir[f + k]` where
    `f = floor(fs * t) - fdl2`.

    The C++ does this in two threadpool phases -- scatter into per-thread
    buffers, then reduce. That is dropped here: the output is partitioned across
    workers instead, so each worker owns a disjoint slice of `rir` and no
    private buffer or reduce pass is needed. The per-arrival arithmetic and loop
    order are unchanged, and the serial path is bit-identical to upstream.

    `rir_len` is the validation buffer and the whole of [0, rir_len) is the
    slice this call owns; the caller must zero `rir` first.
    """
    var fdl2 = (fdl - 1) // 2
    _rir_scatter(
        rir, time, alpha, lut, hann, n_times, rir_len, fs, fdl, fdl2, lut_gran, 0, rir_len
    )


def _rir_scatter(
    rir: Ptr,
    time: Ptr,
    alpha: Ptr,
    lut: Ptr,
    hann: Ptr,
    n_times: Int,
    rir_len: Int,
    fs: Int,
    fdl: Int,
    fdl2: Int,
    lut_gran: Int,
    lo: Int,
    hi: Int,
):
    """One contiguous output slice [lo, hi) of the RIR. Never called reentrantly."""
    var lut_gran_f = Float64(lut_gran)

    var i = 0
    while i < n_times:
        # decompose integral/fractional parts
        var sample_frac = Float64(fs) * time.unsafe_load(i)
        var time_ip = floor(sample_frac)
        var time_fp = sample_frac - time_ip

        # LUT interpolation
        var x_off_frac = (1.0 - time_fp) * lut_gran_f
        var lut_gran_off = floor(x_off_frac)
        var x_off = x_off_frac - lut_gran_off

        var lut_pos = Int(lut_gran_off)
        var f = Int(time_ip) - fdl2
        var a = alpha.unsafe_load(i)

        # Upstream's C++ requires floor(fs * t_min) - fdl2 >= 0 and
        # ceil(fs * t_max) + fdl2 < rir_len, so an arrival whose fdl taps would
        # run past the end of the buffer cannot occur for a caller that sized it
        # the way upstream requires. The clamp below keeps this kernel in bounds
        # whatever it is handed -- a short buffer costs the truncated taps, not a
        # silent write into whatever follows `rir` in memory.
        var k0 = 0
        var k1 = fdl
        if f + k0 < lo:
            k0 = lo - f
        if f + k1 > hi:
            k1 = hi - f
        if k0 < 0:
            k0 = 0

        var k = k0
        while k < k1:
            var lp = lut_pos + k * lut_gran
            var sv = lut.unsafe_load(lp) + x_off * (lut.unsafe_load(lp + 1) - lut.unsafe_load(lp))
            rir.unsafe_store(f + k, rir.unsafe_load(f + k) + a * hann.unsafe_load(k) * sv)
            k += 1

        i += 1


def rir_builder_parallel(
    rir: Ptr,
    time: Ptr,
    alpha: Ptr,
    lut: Ptr,
    hann: Ptr,
    n_times: Int,
    rir_len: Int,
    fs: Int,
    fdl: Int,
    lut_gran: Int,
    n_workers: Int,
):
    """`rir_builder` with the output partitioned across the thread pool.

    Each worker walks every arrival but only accumulates into the slice of `rir`
    it owns, so no two workers ever write the same sample and no reduce is
    needed. Reproduces the serial result bit for bit.
    """
    var fdl2 = (fdl - 1) // 2

    def worker(w: Int) {imm}:
        var lo = (rir_len * w) // n_workers
        var hi = (rir_len * (w + 1)) // n_workers
        _rir_scatter(rir, time, alpha, lut, hann, n_times, rir_len, fs, fdl, fdl2, lut_gran, lo, hi)

    parallelize(worker, n_workers, n_workers)


def delay_sum(
    dst_o: Ptr, irs: Ptr, delays: Ptr, n_irs: Int, n_taps: Int, out_len: Int, n_workers: Int
):
    """Overlap-add `n_irs` short impulse responses of `n_taps` at integer delays.

    Upstream (`delay_sum.cpp`) writes `out[delay + k] += irs[idx, k]` and then
    sums the per-thread buffers; as in `rir_builder` the private-buffer reduce is
    replaced by partitioning the output. `out_len` is the validation length and
    the whole of [0, out_len) is the slice this call owns.
    """
    _delay_sum_serial(dst_o, irs, delays, n_irs, n_taps, 0, out_len)


def _delay_sum_serial(dst_o: Ptr, irs: Ptr, delays: Ptr, n_irs: Int, n_taps: Int, lo: Int, hi: Int):
    for idx in range(n_irs):
        var d = delays.unsafe_load(idx)
        for k in range(n_taps):
            var pos = Int(d) + k
            if lo <= pos < hi:
                dst_o.unsafe_store(pos, dst_o.unsafe_load(pos) + irs.unsafe_load(idx * n_taps + k))


def delay_sum_parallel(
    dst_o: Ptr, irs: Ptr, delays: Ptr, n_irs: Int, n_taps: Int, out_len: Int, n_workers: Int
):
    """`delay_sum` with the output partitioned across the thread pool."""
    def worker(w: Int) {imm}:
        var lo = (out_len * w) // n_workers
        var hi = (out_len * (w + 1)) // n_workers
        _delay_sum_serial(dst_o, irs, delays, n_irs, n_taps, lo, hi)

    parallelize(worker, n_workers, n_workers)


def fractional_delay(
    dst_o: Ptr, time: Ptr, lut: Ptr, hann: Ptr, n_times: Int, fdl: Int, lut_gran: Int
):
    """Build fractional-delay filters for each fractional arrival time.

    Upstream (`fractional_delay.cpp`) interpolates the sinc LUT at
    `(1 - time_fp) * lut_gran` and windows each tap, writing one row of length
    `fdl` per arrival. Every row is independent, so this vectorises.
    """
    var lut_gran_f = Float64(lut_gran)

    for idx in range(n_times):
        var x_off_frac = (1.0 - time.unsafe_load(idx)) * lut_gran_f
        var lut_gran_off = floor(x_off_frac)
        var x_off = x_off_frac - lut_gran_off

        var lut_pos = Int(lut_gran_off)
        var row = dst_o + idx * fdl
        var k = 0
        while k < fdl:
            row.unsafe_store(k, hann.unsafe_load(k) * (lut.unsafe_load(lut_pos) + x_off * (lut.unsafe_load(lut_pos + 1) - lut.unsafe_load(lut_pos))))
            lut_pos += lut_gran
            k += 1


# =========================================================================
# pyroomacoustics/simulation/ism.py -- the ISM scalar assembly
# =========================================================================


def ism_image_amplitudes(
    dst: Ptr,
    images: Ptr,
    mic: Ptr,
    damping: Ptr,
    dim: Int,
    n_img: Int,
    c: Float64,
    air_abs: Ptr,
    n_bands: Int,
):
    """Per-image arrival time and amplitude, mirroring `compute_ism_rir`.

    Upstream computes `dist = |images - mic|`, `time = dist / c`,
    `oct_band_amplitude = att / dist` and then applies
    `exp(-0.5 * air_abs_coeffs[:, None] * dist)` when air absorption is on.

    `dst` receives the amplitudes as (n_bands, n_img) followed by the n_img
    arrival times, which is what the RIR assembly below consumes.
    """
    for i in range(n_img):
        var acc = 0.0
        for d in range(dim):
            var diff = images.unsafe_load(d * n_img + i) - mic.unsafe_load(d)
            acc += diff * diff
        var dist = sqrt(acc)

        # time = dist / c
        dst.unsafe_store(n_img * n_bands + i, dist / c)

        # oct_band_amplitude = att / dist, then air absorption
        var att = damping.unsafe_load(i)
        for b in range(n_bands):
            dst.unsafe_store(b * n_img + i, att / dist * exp(-0.5 * air_abs.unsafe_load(b) * dist))


def ism_rir_length(time: Ptr, n_img: Int, fs: Int, fdl: Int) -> Int:
    """Length of the RIR buffer `compute_ism_rir` allocates.

    Upstream computes `N = int(math.ceil(t_max * fs + fdl2 + 1)) + 1` where
    `t_max` is the largest arrival time and `fdl2 = fdl // 2`. The trailing `+1`
    is upstream's own allowance for rare cases where numerical error pushes the
    last sample past the end. The time buffer is expected to already carry the
    `fdl2 / fs` delay that upstream adds to every arrival.
    """
    var fdl2 = fdl // 2
    var t_max = 0.0
    for i in range(n_img):
        t_max = max(t_max, time.unsafe_load(i))
    var target = t_max * Float64(fs) + Float64(fdl2) + 1.0
    var icell = Int(target)
    if Float64(icell) < target:
        icell += 1
    return icell + 1


def ism_rir_assemble(
    rir: Ptr,
    time: Ptr,
    amplitudes: Ptr,
    n_img: Int,
    n_bands: Int,
    fs: Int,
    fdl: Int,
    lut: Ptr,
    hann: Ptr,
    lut_gran: Int,
    rir_len: Int,
):
    """Assemble the per-octave-band image contributions into the final RIR.

    Upstream `compute_ism_rir` calls `rir_builder` once for every octave band
    into a per-band buffer and adds each into `rir`; since `rir_builder`
    accumulates, running it per band directly into `rir` is that same sum. The
    caller must have zeroed `rir` and set the times to carry the `fdl2 / fs`
    delay, exactly as upstream does before the loop.
    """
    for b in range(n_bands):
        rir_builder(rir, time, amplitudes + b * n_img, lut, hann, n_img, rir_len, fs, fdl, lut_gran, 1)

# =========================================================================


@export("mpa_build_sinc_lut")
def mpa_build_sinc_lut(lut: Int, fdl: Int, lut_gran: Int) abi("C"):
    build_sinc_lut(p(lut), fdl, lut_gran)


@export("mpa_build_hann")
def mpa_build_hann(hann: Int, fdl: Int) abi("C"):
    build_hann(p(hann), fdl)


@export("mpa_convmtx_dot")
def mpa_convmtx_dot(dst: Int, x: Int, h: Int, n: Int, m: Int) abi("C"):
    convmtx_dot(p(dst), p(x), p(h), n, m)


@export("mpa_low_pass_dirac")
def mpa_low_pass_dirac(dst: Int, t0: Int, alpha: Int, n_rows: Int, n: Int, fs: Float64) abi("C"):
    low_pass_dirac(p(dst), p(t0), p(alpha), n_rows, n, fs)


@export("mpa_fractional_delay_bank")
def mpa_fractional_delay_bank(
    dst: Int, delays: Int, n_delays: Int, l: Int, filter_len: Int, min_delay: Float64
) abi("C"):
    fractional_delay_bank(p(dst), p(delays), n_delays, l, filter_len, min_delay)


@export("mpa_levinson")
def mpa_levinson(
    r: Int, b: Int, x: Int, p_len: Int, n_rhs: Int, a: Int, arev: Int
) abi("C"):
    levinson(p(r), p(b), p(x), p_len, n_rhs, p(a), p(arev))


@export("mpa_lpc_from_autocorr")
def mpa_lpc_from_autocorr(r: Int, x: Int, p_len: Int, a: Int, arev: Int) abi("C"):
    lpc_from_autocorr(p(r), p(x), p_len, p(a), p(arev))


@export("mpa_autocorr_time")
def mpa_autocorr_time(x: Int, r: Int, length: Int, p_max: Int) abi("C"):
    autocorr_time(p(x), p(r), length, p_max)


@export("mpa_goertzel")
def mpa_goertzel(x: Int, dst_o: Int, length: Int, k: Int) abi("C"):
    goertzel(p(x), p(dst_o), length, k)


@export("mpa_distance")
def mpa_distance(r: Int, x: Int, y: Int, dst: Int, dim: Int, nx: Int, ny: Int) abi("C"):
    distance(p(r), p(x), p(y), p(dst), dim, nx, ny)


@export("mpa_unit_vec2D")
def mpa_unit_vec2D(dst: Int, phi: Float64) abi("C"):
    unit_vec2D(p(dst), phi)


@export("mpa_linear_2D_array")
def mpa_linear_2D_array(
    dst: Int, center: Int, m: Int, phi: Float64, d: Float64, u: Int
) abi("C"):
    linear_2D_array(p(dst), p(center), m, phi, d, p(u))


@export("mpa_circular_2D_array")
def mpa_circular_2D_array(
    dst: Int, center: Int, m: Int, phi0: Float64, radius: Float64
) abi("C"):
    circular_2D_array(p(dst), p(center), m, phi0, radius)


@export("mpa_square_2D_array")
def mpa_square_2D_array(
    dst: Int, center: Int, m: Int, n: Int, phi: Float64, d: Float64, row: Int, u: Int
) abi("C"):
    square_2D_array(p(dst), p(center), m, n, phi, d, p(row), p(u))


@export("mpa_H")
def mpa_H(dst: Int, a: Int, rows: Int, cols: Int) abi("C"):
    H(p(dst), p(a), rows, cols)


@export("mpa_mdot3")
def mpa_mdot3(
    a: Int, b: Int, c: Int, dst: Int, work: Int, n: Int, m: Int, k: Int, l: Int
) abi("C"):
    mdot3(p(a), p(b), p(c), p(dst), p(work), n, m, k, l)


@export("mpa_steering_vector_2D_from_point")
def mpa_steering_vector_2D_from_point(
    r: Int,
    center: Int,
    x: Int,
    dst: Int,
    dim: Int,
    n: Int,
    n_src: Int,
    freq: Float64,
    c: Float64,
    attn: Int,
    ff: Int,
    work: Int,
) abi("C"):
    steering_vector_2D_from_point(
        p(r), p(center), p(x), p(dst), dim, n, n_src, freq, c, attn != 0, ff != 0, p(work)
    )


@export("mpa_steering_vector_2D")
def mpa_steering_vector_2D(
    r: Int,
    center: Int,
    dst: Int,
    n: Int,
    n_phi: Int,
    freq: Float64,
    dist: Float64,
    c: Float64,
    attn: Int,
    phis: Int,
    work: Int,
) abi("C"):
    steering_vector_2D(p(r), p(center), p(dst), n, n_phi, freq, dist, c, attn != 0, p(phis), p(work))


@export("mpa_response")
def mpa_response(dst: Int, w: Int, a: Int, n: Int, n_phi: Int) abi("C"):
    response(p(dst), p(w), p(a), n, n_phi)


@export("mpa_far_field_weights")
def mpa_far_field_weights(
    dst: Int, proj: Int, frequencies: Int, m: Int, n_freq: Int, c: Float64
) abi("C"):
    far_field_weights(p(dst), p(proj), p(frequencies), m, n_freq, c)


@export("mpa_far_field_projection")
def mpa_far_field_projection(
    dst: Int, r: Int, center: Int, dim: Int, m: Int, phi: Float64
) abi("C"):
    far_field_projection(p(dst), p(r), p(center), dim, m, phi)


@export("mpa_rake_delay_and_sum_weights")
def mpa_rake_delay_and_sum_weights(
    dst: Int, steer: Int, m: Int, n_freq: Int, n_src: Int
) abi("C"):
    rake_delay_and_sum_weights(p(dst), p(steer), m, n_freq, n_src)


@export("mpa_cov_gram")
def mpa_cov_gram(dst: Int, a: Int, rows: Int, cols: Int) abi("C"):
    cov_gram(p(dst), p(a), rows, cols)


@export("mpa_rake_mvdr_solution")
def mpa_rake_mvdr_solution(dst: Int, h: Int, c: Int, m: Int, l: Int) abi("C"):
    rake_mvdr_solution(p(dst), p(h), p(c), m, l)


@export("mpa_response_sinr")
def mpa_response_sinr(dst: Int, g: Int, r_xx: Int, k_nq: Int, n: Int) abi("C"):
    response_sinr(p(dst), p(g), p(r_xx), p(k_nq), n)


@export("mpa_rir_builder")
def mpa_rir_builder(
    rir: Int,
    time: Int,
    alpha: Int,
    lut: Int,
    hann: Int,
    n_times: Int,
    rir_len: Int,
    fs: Int,
    fdl: Int,
    lut_gran: Int,
) abi("C"):
    rir_builder(p(rir), p(time), p(alpha), p(lut), p(hann), n_times, rir_len, fs, fdl, lut_gran, 1)


@export("mpa_rir_builder_parallel")
def mpa_rir_builder_parallel(
    rir: Int,
    time: Int,
    alpha: Int,
    lut: Int,
    hann: Int,
    n_times: Int,
    rir_len: Int,
    fs: Int,
    fdl: Int,
    lut_gran: Int,
    n_workers: Int,
) abi("C"):
    rir_builder_parallel(p(rir), p(time), p(alpha), p(lut), p(hann), n_times, rir_len, fs, fdl, lut_gran, n_workers)


@export("mpa_delay_sum")
def mpa_delay_sum(
    dst_o: Int, irs: Int, delays: Int, n_irs: Int, n_taps: Int, out_len: Int
) abi("C"):
    delay_sum(p(dst_o), p(irs), p(delays), n_irs, n_taps, out_len, 1)


@export("mpa_delay_sum_parallel")
def mpa_delay_sum_parallel(
    dst_o: Int, irs: Int, delays: Int, n_irs: Int, n_taps: Int, out_len: Int, n_workers: Int
) abi("C"):
    delay_sum_parallel(p(dst_o), p(irs), p(delays), n_irs, n_taps, out_len, n_workers)


@export("mpa_fractional_delay")
def mpa_fractional_delay(
    dst_o: Int, time: Int, lut: Int, hann: Int, n_times: Int, fdl: Int, lut_gran: Int
) abi("C"):
    fractional_delay(p(dst_o), p(time), p(lut), p(hann), n_times, fdl, lut_gran)


@export("mpa_ism_image_amplitudes")
def mpa_ism_image_amplitudes(
    dst: Int,
    images: Int,
    mic: Int,
    damping: Int,
    dim: Int,
    n_img: Int,
    c: Float64,
    air_abs: Int,
    n_bands: Int,
) abi("C"):
    ism_image_amplitudes(p(dst), p(images), p(mic), p(damping), dim, n_img, c, p(air_abs), n_bands)

@export("mpa_ism_rir_length")
def mpa_ism_rir_length(time: Int, n_img: Int, fs: Int, fdl: Int) abi("C") -> Int:
    return ism_rir_length(p(time), n_img, fs, fdl)


@export("mpa_ism_rir_assemble")
def mpa_ism_rir_assemble(
    rir: Int,
    time: Int,
    amplitudes: Int,
    n_img: Int,
    n_bands: Int,
    fs: Int,
    fdl: Int,
    lut: Int,
    hann: Int,
    lut_gran: Int,
    rir_len: Int,
) abi("C"):
    ism_rir_assemble(
        p(rir), p(time), p(amplitudes), n_img, n_bands, fs, fdl, p(lut), p(hann),
        lut_gran, rir_len,
    )
