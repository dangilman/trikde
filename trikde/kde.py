import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage


def effective_sample_size_max(weights):
    """
    ESS defined as sum(w / max(w)).  This is trikde's historical definition.
    It is always <= the Kish ESS, so it is conservative, but it depends
    entirely on a single order statistic (max(w)) and is therefore noisy.
    """
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights / np.max(weights)))

def effective_sample_size_kish(weights):
    """
    Kish's effective sample size, (sum w)^2 / sum(w^2).  Equals n for uniform
    weights and is the standard choice for importance-weighted samples.
    """
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights) ** 2 / np.sum(weights ** 2))


def weighted_covariance(data, weights=None):
    """
    Covariance of (n_samples, n_dim) data, honouring importance weights.
    """
    data = np.atleast_2d(data)
    if data.shape[0] == 1:
        data = data.T
    if weights is None:
        return np.cov(data.T)
    weights = np.asarray(weights, dtype=float)
    if np.all(weights == weights[0]):
        return np.cov(data.T)
    return np.cov(data.T, aweights=weights)


class PointInterp(object):

    def __init__(self, nbins):
        """
        :param nbins: number of bins along each dimension
        """
        self._nbins = nbins

    def get_bins(self, ranges, num_bins):
        histbins = []
        histcens = []
        for i in range(0, len(ranges)):
            t_bin_edges = np.linspace(ranges[i][0], ranges[i][-1], num_bins + 1)
            t_bin_cens = (t_bin_edges[1:] + t_bin_edges[:-1]) / 2
            histbins.append(t_bin_edges)
            histcens.append(t_bin_cens)
        return histbins, histcens

    @staticmethod
    def _get_coordinates(ranges, num_bins):
        """
        Bin *centres* along each dimension.

        FIX 3: this previously returned np.linspace(min, max, num_bins), i.e.
        the grid endpoints, whose spacing is (max-min)/(num_bins-1).  The
        histogram lives on bin centres with spacing (max-min)/num_bins.  The
        mismatch inflated the effective bandwidth by num_bins/(num_bins-1) and
        put the density on a grid half a bin away from where
        InterpolatedLikelihood assumed it was.
        """
        points = []
        for i in range(0, len(ranges)):
            edges = np.linspace(ranges[i][0], ranges[i][-1], num_bins + 1)
            points.append(0.5 * (edges[1:] + edges[:-1]))
        return points

    def NDhistogram(self, data, weights, ranges, nbins=None):
        """
        :param data: data to make the histogram. Shape (nsamples, ndim)
        :param weights: importance weights
        :param ranges: parameter ranges corresponding to columns in data
        :return: histogram, histogram bin edges, histogram bin centers

        Axes are in parameter order: axis i corresponds to ranges[i].
        """
        if nbins is None:
            nbins = self._nbins
        bin_edges, bin_centers = self.get_bins(ranges, nbins)
        H, _ = np.histogramdd(data, range=ranges, bins=bin_edges, weights=weights)
        return H, bin_edges, bin_centers


class KDE(PointInterp):
    """
    Gaussian kernel density estimator in arbitrary dimensions with boundary
    correction.  The returned density has axes in parameter order:
    density[i0, i1, ...] corresponds to ranges[0][i0], ranges[1][i1], ...
    """

    def __init__(self, bandwidth_scale=1, nbins=None, boundary_order=1, force_bandwidth=None,
                 use_cov=True, second_order_correction_floor=1e-10, weighted_covariance=True,
                 ess_definition='max'):
        """
        :param bandwidth_scale: scales the bandwidth relative to Silverman's value
        :param nbins: number of bins for output pdf
        :param boundary_order: 2 (second order), 1 (first order) or 0 (no correction)
        :param force_bandwidth: optionally set the bandwidth; may be a callable f(n, d)
        :param use_cov: use the sample covariance matrix to shape the kernels
        :param second_order_correction_floor: tolerance for the 2nd order correction
        :param weighted_covariance: FIX 2 -- use the importance weights when computing
        the covariance / standard deviation that shapes the kernel.  Set False to
        reproduce the old (unweighted) behaviour.
        :param ess_definition: 'max' for sum(w/max(w)) (historical trikde behaviour)
        or 'kish' for (sum w)^2 / sum(w^2).  Only affects the bandwidth, and only
        weakly, since h ~ ESS^(-1/(d+4)).
        """
        self.bandwidth_scale = bandwidth_scale
        self._boundary_order = boundary_order
        self._use_cov = use_cov
        self._force_bandwidth = force_bandwidth
        self._kde_bandwidth = None
        self._second_order_correction_floor = second_order_correction_floor
        self._weighted_covariance = weighted_covariance
        if ess_definition not in ('max', 'kish'):
            raise ValueError("ess_definition must be 'max' or 'kish'")
        self._ess_definition = ess_definition
        self._effective_sample_size = None
        super(KDE, self).__init__(nbins)

    @property
    def kde_bandwidth(self):
        """Return the kde bandwidth, if it has been evaluated"""
        return self._kde_bandwidth

    @property
    def effective_sample_size(self):
        """Return the ESS actually used to set the bandwidth"""
        return self._effective_sample_size

    def _scotts_bandwidth(self, n, d):
        return 1.05 * n ** (-1. / (d + 4))

    def _silverman_bandwidth(self, n, d):
        return (n * (d + 2) / 4.) ** (-1. / (d + 4))

    @staticmethod
    def _gaussian_kernel(inverse_cov_matrix, coords_centered, kernel_shape):
        coords_centered = np.atleast_2d(coords_centered)
        inverse_cov_matrix = np.atleast_2d(inverse_cov_matrix)
        quad = np.einsum('ij,jk,ik->i', coords_centered, inverse_cov_matrix, coords_centered)
        return np.reshape(np.exp(-0.5 * quad), kernel_shape)

    def _kernel_offsets(self, ranges, nbins, dimension, covariance=None, n_sigma=4.0):
        """
        Coordinate offsets of the kernel grid, truncated at n_sigma.
        Returns (offsets, kernel_shape). Kernel spans only where the Gaussian
        is non-negligible instead of the full nbins per axis, which avoids
        fftconvolve padding to (2*nbins-1)**dimension.
        """
        dx = np.array([(ranges[i][-1] - ranges[i][0]) / float(nbins)
                       for i in range(dimension)])
        half_max = (nbins - 1) // 2
        if covariance is None:
            hw = np.full(dimension, half_max, dtype=int)
        else:
            sig = np.sqrt(np.diag(np.atleast_2d(covariance)))
            hw = np.ceil(n_sigma * sig / dx).astype(int)
            hw = np.clip(hw, 1, half_max)
        axes = [np.arange(-hw[i], hw[i] + 1) * dx[i] for i in range(dimension)]
        grids = np.meshgrid(*axes, indexing='ij')
        kernel_shape = tuple(2 * int(hw[i]) + 1 for i in range(dimension))
        return np.vstack([g.ravel() for g in grids]).T, kernel_shape

    def __call__(self, data, ranges, weights):
        """
        :param data: shape (n_observations, ndim)
        :param ranges: list of [min, max] per dimension
        :param weights: importance weights for each observation (or None)
        :return: KDE estimate, axes in parameter order
        """
        data = np.asarray(data)
        try:
            dimension = int(np.shape(data)[1])
        except IndexError:
            dimension = 1
            data = data.reshape(-1, 1)

        nbins = self._nbins

        # ---- histogram, axes in parameter order -------------------------------
        H, _, _ = self.NDhistogram(data, weights, ranges)

        # ---- bandwidth --------------------------------------------------------
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if self._ess_definition == 'kish':
                effective_sample_size = effective_sample_size_kish(weights)
            else:
                effective_sample_size = effective_sample_size_max(weights)
        else:
            effective_sample_size = data.shape[0]
        self._effective_sample_size = effective_sample_size

        if self._force_bandwidth is None:
            bandwidth = self.bandwidth_scale * self._silverman_bandwidth(effective_sample_size, dimension)
        elif callable(self._force_bandwidth):
            bandwidth = self._force_bandwidth(effective_sample_size, dimension)
        else:
            bandwidth = float(self._force_bandwidth)
        self._kde_bandwidth = bandwidth

        # ---- kernel covariance ------------------------------------------------
        # FIX 2: honour the importance weights
        cov_weights = weights if self._weighted_covariance else None
        if self._use_cov is False:
            if cov_weights is None:
                var = np.std(data, axis=0) ** 2
            else:
                mu = np.average(data, axis=0, weights=cov_weights)
                var = np.average((data - mu) ** 2, axis=0, weights=cov_weights)
            covariance = np.eye(dimension) * bandwidth ** 2 * var
        else:
            covariance = bandwidth ** 2 * np.atleast_2d(weighted_covariance(data, cov_weights))

        if dimension > 1:
            c_inv = np.linalg.inv(covariance)
        else:
            c_inv = np.atleast_2d(1. / covariance)

        # ---- convolve ---------------------------------------------------------
        offsets, kernel_shape = self._kernel_offsets(ranges, nbins, dimension, covariance)
        gaussian_kernel = self._gaussian_kernel(c_inv, offsets, kernel_shape)

        density = fftconvolve(H, gaussian_kernel, mode='same')

        bc = BoundaryCorrection(gaussian_kernel, H.shape, self._second_order_correction_floor)

        if self._boundary_order == 0:
            pass
        elif self._boundary_order == 1:
            density = bc.first_order(density)
        elif self._boundary_order == 2:
            density = bc.second_order(density, H, gaussian_kernel)
        else:
            raise ValueError('boundary_order must be 0, 1 or 2')

        # FIX 1: no transpose -- already in parameter order
        return density


class BoundaryCorrection(object):

    def __init__(self, pdf, domain_shape, tol_second_order=1e-10):
        self._pdf = pdf
        self._tol_second_order = tol_second_order
        self._boundary_kernel = np.ones(domain_shape)

    def _renormalization(self):
        boundary_normalization = fftconvolve(self._boundary_kernel, self._pdf, mode='same')
        total_mass = np.sum(self._pdf)
        return boundary_normalization / total_mass

    def first_order(self, density):
        """
        Divide by the fraction of kernel mass inside the parameter space.
        """
        return density * self._renormalization() ** -1

    def second_order(self, density, H, gaussian_kernel):
        """
        Multiplicative bias correction (Jones et al. 1995; Lewis 2019):
            f_hat = g * (K_h * (H / g))
        with g the first-order corrected pilot.  O(h^4) bias vs O(h^2).
        """
        renormalization = self._renormalization()
        g = density * renormalization ** -1
        mbc_mask = g > self._tol_second_order * np.max(g)
        H_flattened = np.where(mbc_mask, H / np.where(mbc_mask, g, 1.0), H)
        density_corrected = fftconvolve(H_flattened, gaussian_kernel, mode='same')
        density_corrected *= renormalization ** -1
        return np.where(mbc_mask, g * density_corrected, g)
