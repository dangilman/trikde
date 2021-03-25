import numpy as np
from scipy.signal import fftconvolve


class KDE(object):
    """
    This class implements a Gaussian kernel density estimator in arbitrary dimensions with a first order renormalization
    at the boundary of parameter space
    """

    def __init__(self, bandwidth_scale=1, nbins=None):

        """

        :param bandwidth_scale: scales the kernel bandwidth, or the variance of each Gaussian
        :param nbins: number of bins in the KDE
        """

        self.bandwidth_scale = bandwidth_scale
        self._nbins = nbins

    def _scotts_factor(self, n, d):

        """
        Implements the kernel bandwidth using Scotts factor
        :param n: number of data points
        :param d: number of dimensions
        :return: kernel bandwidth
        """
        return 1.05 * n ** (-1. / (d + 4))

    def _gaussian_kernel(self, inverse_cov_matrix, coords_centered, dimension, n_reshape):

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

    def _get_coordinates(self, ranges):

        """
        Builds an array of coordinate values from the specified parameter ranges
        :param ranges: parameter ranges
        :return: array of coordinate values
        """
        points = []

        for i in range(0, len(ranges)):
            points.append(np.linspace(ranges[i][0], ranges[i][1], self._nbins))
        return points

    def NDhistogram(self, data, weights, ranges):

        """

        :param data: data to make the histogram. Shape (nsamples, ndim)
        :param coordinates: np.linspace(min, max, nbins) for each dimension
        :param ranges: parameter ranges corresponding to columns in data
        :param weights: param weights
        :return: histogram
        """

        coordinates = self._get_coordinates(ranges)

        histbins = []
        for i, coord in enumerate(coordinates):
            histbins.append(np.linspace(ranges[i][0], ranges[i][-1], len(coord) + 1))

        H, _ = np.histogramdd(data, range=ranges, bins=histbins, weights=weights)

        return H.T

    def __call__(self, data, ranges, weights, boundary_order=1):

        """

        :param data: data to make the histogram, shape = (n_observations, ndim)
        :param ranges: a list of parameter ranges corresponding to each dimension
        :param weights: importance weights for each observation
        :return: the KDE estimate of the data
        """

        # compute coordinate arrays for each parameter
        coordinates = self._get_coordinates(ranges)

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
        H = self.NDhistogram(data, weights, ranges)

        # compute the covariance, scale by the bandwidth
        bandwidth = self.bandwidth_scale * self._scotts_factor(data.shape[0], dimension)
        covariance = bandwidth * np.cov(data.T)

        # invert covariance matrix
        if dimension > 1:
            c_inv = np.linalg.inv(covariance)
        else:
            c_inv = 1 / covariance

        n = len(coordinates[0])
        gaussian_kernel = self._gaussian_kernel(c_inv, cc_center, dimension, n)

        # now compute the guassian KDE
        density = fftconvolve(H, gaussian_kernel, mode='same')

        # renormalize the boundary to remove bias
        if boundary_order == 1:
            boundary_kernel = np.ones(np.shape(H))
            boundary_normalization = fftconvolve(gaussian_kernel, boundary_kernel, mode='same')

            density *= boundary_normalization ** -1

        return density