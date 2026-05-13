import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import numpy as np


class PointInterp(object):

    def __init__(self, nbins):
        """

        :param nbins:
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
        Builds an array of coordinate values from the specified parameter ranges
        :param ranges: parameter ranges
        :return: array of coordinate values
        """
        points = []
        for i in range(0, len(ranges)):
            points.append(np.linspace(ranges[i][0], ranges[i][1], num_bins))
        return points

    def NDhistogram(self, data, weights, ranges, nbins=None):

        """

        :param data: data to make the histogram. Shape (nsamples, ndim)
        :param coordinates: np.linspace(min, max, nbins) for each dimension
        :param ranges: parameter ranges corresponding to columns in data
        :param weights: param weights
        :return: histogram, histogram bin edges, histogram bin centers
        """
        if nbins is None:
            nbins = self._nbins
        #coordinates = self._get_coordinates(ranges, nbins)
        #histbins = []
        #for i, coord in enumerate(coordinates):
        #    histbins.append(np.linspace(ranges[i][0], ranges[i][-1], len(coord) + 1))
        bin_edges, bin_centers = self.get_bins(ranges, nbins)

        H, _ = np.histogramdd(data, range=ranges, bins=bin_edges, weights=weights)

        ##update this to no longer return the transpose, since this is not needed for default methods.
        return H, bin_edges, bin_centers

class KDE(PointInterp):
    """
    This class implements a Gaussian kernel density estimator in arbitrary dimensions with a first order renormalization
    at the boundary of parameter space
    """

    def __init__(self, bandwidth_scale=1, nbins=None, boundary_order=1, force_bandwidth=None, use_cov=True,
                 second_order_correction_floor=1e-10):

        """
        Gaussian kernel density estimator with first-order boundary correction
        :param bandwidth_scale: scales the bandwidth of the kernel density estimator relative to Scott's factor value
        :param nbins: number of bins for output pdf
        :param boundary_order: 2 (second order), 1 (first order) or 0 (no correction)
        :param force_bandwidth: optionally set the bandwidth along each axis; otherwise, it is chosen based on
        Silvermanns's rule. force_bandwidth can be a function
        :param use_cov: bool; use the covariance matrix of samples to define the Gaussian kernels
        :param second_order_correction_floor: numerical tolerance when implementing multiplicative bias 2nd order boundary
        correction
        """
        self.bandwidth_scale = bandwidth_scale
        self._boundary_order = boundary_order
        self._use_cov = use_cov
        self._force_bandwidth = force_bandwidth
        self._kde_bandwidth = None
        self._second_order_correction_floor = second_order_correction_floor
        super(KDE, self).__init__(nbins)

    @property
    def kde_bandwidth(self):
        """
        Return the kde bandwidth, if it has been evaluated
        :return:
        """
        return self._kde_bandwidth

    def _scotts_bandwidth(self, n, d):

        """
        Implements the kernel bandwidth using Scotts factor
        :param n: number of data points
        :param d: number of dimensions
        :return: kernel bandwidth
        """
        return 1.05 * n ** (-1. / (d + 4))

    def _silverman_bandwidth(self, n, d):

        """
        Implements the kernel bandwidth using Silverman's factor
        :param n: number of data points
        :param d: number of dimensions
        :return: kernel bandwidth
        """
        return (n * (d + 2) / 4.) ** (-1. / (d + 4))

    @staticmethod
    def _gaussian_kernel(inverse_cov_matrix, coords_centered, dimension, n_reshape):

        """
        Computes the multivariate gaussian KDE from the covariance matrix and observation array
        :param inverse_cov_matrix: inverse of the covariance matrix estimated from observations
        :param coords_centered: array of observations transformed into pixel space
        :param dimension: number of dimensions
        :param n_reshape: shape of output
        :return: gaussian KDE evalauted at coords
        """
        def _gauss(_x):
            return np.exp(-0.5 * np.dot(np.dot(_x, inverse_cov_matrix), _x))
        z = [_gauss(coord) for coord in coords_centered]
        return np.reshape(z, tuple([n_reshape] * dimension))

    def __call__(self, data, ranges, weights):

        """

        :param data: data to make the histogram, shape = (n_observations, ndim)
        :param ranges: a list of parameter ranges corresponding to each dimension
        :param weights: importance weights for each observation
        :return: the KDE estimate of the data
        """

        # compute coordinate arrays for each parameter
        coordinates = self._get_coordinates(ranges, self._nbins)

        # shift coordinate arrays so that the center is at (0, 0)
        X = np.meshgrid(*coordinates)
        cc_center = np.vstack([X[i].ravel() - np.mean(ranges[i]) for i in range(len(X))]).T

        try:
            dimension = int(np.shape(data)[1])
        except:
            dimension = 1

        histbins = []
        for i, coord in enumerate(coordinates):
            histbins.append(np.linspace(ranges[i][0], ranges[i][-1], len(coord) + 1))

        # Compute the N-dimensional histogram
        ##This now needs to have the transpose since we are not returning the transpose by default
        H_T, t_bins, t_cens = self.NDhistogram(data, weights, ranges)
        H = H_T.T
        # compute the covariance, scale KDE kernel size by the bandwidth_scale parameter
        if weights is not None:
            normed_weights = weights / np.max(weights)
            effective_sample_size = np.sum(normed_weights)
        else:
            effective_sample_size = data.shape[0]
        if self._force_bandwidth is None:
            bandwidth = self.bandwidth_scale * self._silverman_bandwidth(effective_sample_size, dimension)
        else:
            if callable(self._force_bandwidth):
                bandwidth = self._force_bandwidth(effective_sample_size, dimension)
            else:
                bandwidth = float(self._force_bandwidth)
        self._kde_bandwidth = bandwidth
        if self._use_cov is False:
            covariance = np.eye(dimension) * bandwidth ** 2 * np.std(data, axis=0) ** 2
        else:
            covariance = bandwidth ** 2 * np.cov(data.T)

        # invert covariance matrix
        if dimension > 1:
            c_inv = np.linalg.inv(covariance)
        else:
            c_inv = 1 / covariance

        n = len(coordinates[0])
        gaussian_kernel = self._gaussian_kernel(c_inv, cc_center, dimension, n)

        # now compute the guassian KDE
        density = fftconvolve(H, gaussian_kernel, mode='same')

        bc = BoundaryCorrection(gaussian_kernel, self._second_order_correction_floor)
        if self._boundary_order == 0:
            pass  # no correction
        elif self._boundary_order == 1:
            density = bc.first_order(density)
        elif self._boundary_order == 2:
            density = bc.second_order(density, H, gaussian_kernel)

        ##take the transpose of the output so that it's the original coordinates.
        return density.T


class BoundaryCorrection(object):

    def __init__(self, pdf, tol_second_order=1e-10):
        """

        :param pdf: probability density function
        :param tol_second_order: the threshold for masking of the second order correction
        """
        self._pdf = pdf
        self._tol_second_order = tol_second_order
        self._boundary_kernel = np.ones(np.shape(pdf))

    def _renormalization(self):
        """
        Compute the first-order boundary renormalization factor.
        This is the fraction of kernel mass inside the parameter space at each point,
        normalized so that the interior equals 1.
        """
        boundary_normalization = fftconvolve(self._pdf, self._boundary_kernel, mode='same')
        total_mass = np.sum(self._pdf)
        return boundary_normalization / total_mass

    def first_order(self, density):
        """
        First-order boundary correction: divide by the fraction of kernel mass
        inside the parameter space at each point, correcting for kernel leakage
        near boundaries.
        :param density: raw convolved density
        :return: boundary-corrected density
        """
        return density * self._renormalization() ** -1

    def second_order(self, density, H, gaussian_kernel):
        """
        Second-order boundary correction via multiplicative bias correction
        (Lewis 2019, Jones et al. 1995):
            f_hat = g * (K_h * (H / g))
        where g is the first-order corrected pilot density.
        Achieves O(h^4) bias vs O(h^2) for first-order correction.
        In regions where g is effectively zero, falls back to the first-order estimate.
        :param density: raw convolved density
        :param H: N-dimensional histogram
        :param gaussian_kernel: Gaussian kernel array
        :return: boundary-corrected density
        """
        renormalization = self._renormalization()

        # Step 1: first-order corrected pilot g
        g = density * renormalization ** -1
        mbc_mask = g > self._tol_second_order * np.max(g)

        # Step 2: flatten H by pilot where g is nonzero, else leave H unchanged
        # This avoids dividing by near-zero values in sparsely sampled regions
        H_flattened = np.where(mbc_mask, H / np.where(mbc_mask, g, 1.0), H)
        density_corrected = fftconvolve(H_flattened, gaussian_kernel, mode='same')
        density_corrected *= renormalization ** -1

        # Step 3: multiply back by pilot where g is nonzero, else use first-order estimate
        return np.where(mbc_mask, g * density_corrected, g)
