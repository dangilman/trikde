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


    def sharing_is_caring(self, counts, factor=3):
        """
        shamelessly copied from chatgpt output...used more awesome function name.
        ### use odd factors so that the original bin center is associated with a final bin.
        Uniformly split every coarse histogram bin into ``factor`` sub‑bins
        along each axis, conserving the total counts.

        Parameters
        ----------
        counts : ndarray
            The original N‑D histogram array (integer or float).
        factor : int, optional
            Integer zoom factor per axis (same for all axes).  Default is 3.

        Returns
        -------
        ndarray
            An array with shape ``tuple(s * factor for s in counts.shape)``.
            The sum over any coarse‑bin–sized block equals the original
            value of that coarse bin, so the global sum is preserved.
        """
        factor = int(factor)
        # 1) replicate each bin `factor` times along every axis
        up = counts.astype(float)
        for ax in range(up.ndim):
            up = np.repeat(up, factor, axis=ax)

        # 2) divide so that the replicated block sums to the original value
        return up / (factor ** up.ndim)


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


class LinearKDE(PointInterp):

    def __init__(self, nbins, nbins_eval, resampling=True, n_resample=100000, sharing_interp = False):
        """

        :param nbins:
        :param nbins_eval:
        :param resampling:
        :param n_resample:
        """
        self._nbins_eval = nbins_eval
        self._resampling = resampling
        self._n_resample = n_resample
        self._sharing_interp = sharing_interp
        super(LinearKDE, self).__init__(nbins)

    def __call__(self, data, ranges, weights, n_cpu=1):

        """

        :param data: data to make the histogram, shape = (n_observations, ndim)
        :param ranges: a list of parameter ranges corresponding to each dimension
        :param weights: importance weights for each observation
        :return: the KDE estimate of the data
        """

        # number of dimensions
        ndim = len(ranges)
        #no longer use coordinates function
        ## compute coordinate arrays for each parameter
        ##coordinates = self._get_coordinates(ranges, self._nbins)

        # Compute the N-dimensional histogram
        values, bin_edges, bin_centers = self.NDhistogram(data, weights, ranges)
        ##can use this for just multiplying histogram centers together if you want.
        if self._nbins_eval == self._nbins:
            return values

        if self._resampling:
            ## use regular grid interpolator since the output coordinates from re-sampling random
            ## allow extrapolation beyond the bin center edges.
            interp = RegularGridInterpolator(bin_centers, values,
                                                                             method='linear',
                                                                             bounds_error=False,
                                                                             fill_value=None)
            #weights = np.empty(self._n_resample)
            points = np.empty((self._n_resample, ndim))
            for i in range(0, ndim):
                points[:, i] = np.random.uniform(ranges[i][0], ranges[i][1], self._n_resample)
            weights = interp(points)
            ## leaving this here in case we do need the transpose, but ND histogram is no longer returning the transpose.
            #density = self.NDhistogram(points, weights, ranges, nbins=self._nbins_eval).T
            density, bin_edges, bin_centers = self.NDhistogram(points, weights, ranges, nbins=self._nbins_eval)

        else:

            '''coordinates = self._get_coordinates(ranges, self._nbins_eval)
            coords = np.meshgrid(*coordinates)
            n_resample = self._nbins_eval ** ndim
            points = np.empty((n_resample, ndim))
            for i in range(0, ndim):
                points[:,i] = coords[i].ravel()
            weights = interp(points)
            ##no longer take the transpose of ND histogram
            #density = self.NDhistogram(points, weights, ranges, nbins=self._nbins_eval).T
            '''
            '''
            for the non-random re-sampling supposedly map_coordinates is much faster than regular grid interpolator. 
            I think it's what's used in lenstronomy a lot.
            there are 2 options, 
            i) One keeps the total probability conserved-> it says that the probability density in a bin is
            subdivided evenly among all of the sub bins -> i.e. if there are 9 total counts then the 3x3 bin each get 1 of the counts "sharing_interp"
            this will most accurately recover the probability distribution from an operation of sub-sampling then re-binning we can look at both.
            ii) One approximates the density field with the linear approximation, allows the probability density to vary across each bin.
            this is more correct if there are high bin-to-bin gradients.
            '''
            factor = self._nbins_eval/self._nbins
            if self._sharing_interp:
                density = self.sharing_is_caring(values, factor)
            else:
                coords_1d = []
                for i in range(0, ndim):
                    ##starting image has 5 bins- so pixel values are 0, 1, 2, 3, 4, 5
                    coord = np.arange(self._nbins_eval, dtype=float) / factor
                    ## final image has 15 bins- so pixel values are (0, 1, 2, 3, 4, ...15)/3 ->still goes from 1 to 5 but now has intermediate values
                    coords_1d.append(coord)

                coords_nd = np.meshgrid(*coords_1d, indexing="ij")  # list of ND arrays

                density = ndimage.map_coordinates(
                    values,
                    coords_nd,  # list: one array per axis
                    order=1,  # 0 = nearest, 1 = linear, 3 = cubic …
                    mode="reflect"  # mirror outside the edge
                )
        return density

class KDE(PointInterp):
    """
    This class implements a Gaussian kernel density estimator in arbitrary dimensions with a first order renormalization
    at the boundary of parameter space
    """

    def __init__(self, bandwidth_scale=1, nbins=None, boundary_order=1, use_cov=True):

        """
        Gaussian kernel density estimator with first-order boundary correction
        :param bandwidth_scale: scales the bandwidth of the kernel density estimator relative to Scott's factor value
        :param nbins: number of bins for output pdf
        :param boundary_order: 1 (first order) or 0 (no correction)
        :param use_cov: bool; use the covariance matrix of samples to define the Gaussian kernels
        """
        self.bandwidth_scale = bandwidth_scale
        self._boundary_order = boundary_order
        self._use_cov = use_cov
        super(KDE, self).__init__(nbins)

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
        bandwidth = self.bandwidth_scale * self._scotts_bandwidth(effective_sample_size, dimension)
        covariance = bandwidth ** 2 * np.cov(data.T)
        if self._use_cov is False:
            covariance *= np.eye(dimension)
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
        if self._boundary_order == 1:

            #boundary_kernel = np.ones(np.shape(H))
            #boundary_normalization = fftconvolve(gaussian_kernel, boundary_kernel, mode='same')
            #density *= boundary_normalization ** -1
            edge_kernel = BoundaryCorrection(gaussian_kernel)
            renormalization = edge_kernel.first_order_correction
            density *= renormalization ** -1
        ##take the transpose of the output so that it's the original coordinates.
        return density.T

class BoundaryCorrection(object):

    """
    Basically the same as a Gaussian KDE, except the convolution is with a prior instead of a Gaussian
    """
    def __init__(self, pdf):

        self._pdf = pdf
        self._boundary_kernel = np.ones(np.shape(pdf))

    @property
    def first_order_correction(self):

        boundary_normalization = fftconvolve(self._pdf, self._boundary_kernel, mode='same')
        return boundary_normalization
