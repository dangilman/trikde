import numpy as np
from trikde.kde import KDE

class GaussianWeight(object):

    def __init__(self, mean, sigma):

        self.mean, self.sigma = mean, sigma

    def __call__(self, values):

        dx = (self.mean - values)/self.sigma
        exponent = -0.5 * dx ** 2
        w = np.exp(exponent)
        norm = np.max(w)
        return w/norm

class IndepdendentLikelihoods(object):

    def __init__(self, density_samples_list):

        """
        :param density_samples_list: a list of DensitySamples instances,
        will be multiplied together to obtain the final distribution
        """
        if not isinstance(density_samples_list, list):
            raise Exception('must pass in a list of DensitySamples instances.')
        self.densities = density_samples_list

        self.param_names = density_samples_list[0].param_names
        self.param_ranges = density_samples_list[0].param_ranges

    @property
    def density(self):

        if not hasattr(self, '_product'):
            prod = 1
            for di in self.densities:

                prod *= di.averaged
            self._product = prod
        return self._product

    def projection_1D(self, pname):

        """
        Returns the 1D marginal pdf of the parameter 'pname'
        :param pname: parameter name
        :return: 1D pdf
        """

        sum_inds = []

        if pname not in self.param_names:
            raise Exception('no param named ' + pname)
        for i, name in enumerate(self.param_names):
            if pname != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        projection = np.sum(self.density, tuple(sum_inds))

        return projection

    def projection_2D(self, p1, p2):

        """
        Returns the 2D marginal pdf of the parameters 'p1', 'p2'
        :param p1: parameter name 1
        :param p2: parameter name 2
        :return: 2D pdf
        """

        if p1 not in self.param_names or p2 not in self.param_names:
            raise Exception(p1 + ' or ' + p2 + ' not in ' + str(self.param_names))
        sum_inds = []
        for i, name in enumerate(self.param_names):
            if p1 != name and p2 != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        tpose = False
        for name in self.param_names:
            if name == p1:
                break
            elif name == p2:
                tpose = True
                break

        projection = np.sum(self.density, tuple(sum_inds))
        if tpose:
            projection = projection.T

        return projection

    def projection_1Dold(self, pname):

        proj = 1
        for den in self.densities:
            proj *= den.projection_1D(pname)

        return proj * np.max(proj) ** -1

    def projection_2Dold(self, p1, p2):

        proj = 1
        for den in self.densities:
            proj *= den.projection_2D(p1, p2)
        return proj * np.max(proj) ** -1

class SingleDensity(object):

    """
    This class stores an N-dimensional probability density and optionally a KDE estimate of it
    """

    def __init__(self, data, param_names, param_ranges, weights, bandwidth_scale, nbins, kde):

        """

        :param data: numpy array of observations, shape = (n_observations, n_dimensions)
        :param param_names: parameter names for each dimension
        :param param_ranges: a list of parameter (min, max) values, e.g.
        [[param1_min, param1_max], [param2_min, param2_max], ...]
        :param weights: importance weights for each observation
        :param bandwidth_scale: bandwidth scale, only applicable if using a Gaussian KDE
        :param nbins: number of bins for the KDE or histogram
        :param kde: bool; whether or not to use a Gaussian KDE estimator for the data
        """

        self.param_names = param_names
        self.param_range_list = param_ranges

        estimator = KDE(bandwidth_scale, nbins)

        if kde:
            self.density = estimator(data, param_ranges, weights)
        else:
            self.density = estimator.NDhistogram(data, weights, param_ranges)

    def projection_1D(self, pname):

        """
        Returns the 1D marginal pdf of the parameter 'pname'
        :param pname: parameter name
        :return: 1D pdf
        """

        sum_inds = []

        if pname not in self.param_names:
            raise Exception('no param named '+pname)
        for i, name in enumerate(self.param_names):
            if pname != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        projection = np.sum(self.density, tuple(sum_inds))

        return projection

    def projection_2D(self, p1, p2):

        """
        Returns the 2D marginal pdf of the parameters 'p1', 'p2'
        :param p1: parameter name 1
        :param p2: parameter name 2
        :return: 2D pdf
        """

        if p1 not in self.param_names or p2 not in self.param_names:
            raise Exception(p1 + ' or '+ p2 + ' not in '+str(self.param_names))
        sum_inds = []
        for i, name in enumerate(self.param_names):
            if p1 != name and p2 != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        tpose = False
        for name in self.param_names:
            if name == p1:
                break
            elif name == p2:
                tpose = True
                break

        projection = np.sum(self.density, tuple(sum_inds))
        if tpose:
            projection = projection.T

        return projection

class MultivariateNormalPriorHyperCube(object):

    def __init__(self, means, sigmas, param_names, param_ranges, nbins):

        """
        Used to multiply by a new prior
        :param param_names: the list of parameter names in the joint inference
        :param mean_list: the list of means for the Gaussian priors
        example: [None, None, 5, None]
        will have a mean of 5 for the third parameter, and the other three parameters will have no new prior
        associated with them.
        :param sigma_list: same as mean_list, except specifies the variance, e.g. [None, None, 0.5, None] will assign
        a variance of 0.5 to the Gaussian prior on the third parameter
        :param param_ranges: the sampling ranges for each parameter
        :param nbins: the number of histogram/kde bins
        """
        N = 1000000 * len(param_names)
        shape = (N, len(param_names))
        samples = np.empty(shape)
        weights = 1.
        for i, param in enumerate(param_names):

            samples[:, i] = np.random.uniform(param_ranges[i][0], param_ranges[i][1], N)
            if means[i] is not None:
                weights *= np.exp(-0.5 * (samples[:, i] - means[i]) ** 2/sigmas[i]**2)

        self.param_names = param_names
        self.param_ranges = param_ranges
        self.density = np.histogramdd(samples, range=param_ranges, bins=nbins, weights=weights)[0].T

    @property
    def averaged(self):
        return self.density

class CustomPriorHyperCube(object):

    def __init__(self, chi_square_function, param_names, param_ranges, nbins, kwargs_weight_function={}):

        """
        Used to multiply by a new prior
        :param chi_square_function: a function that computes the chi_square value from an array of samples
        :param param_names: the parameter names
        :param param_ranges: the sampling ranges for each parameter
        :param nbins: the number of histogram/kde bins
        :param kwargs_weight_function: keyword arguments for the weight function
        """
        N = 3000000 * len(param_names)
        shape = (N, len(param_names))
        samples = np.empty(shape)
        for i, param in enumerate(param_names):
            samples[:, i] = np.random.uniform(param_ranges[i][0], param_ranges[i][1], N)
        weights = np.exp(-0.5 * chi_square_function(samples, **kwargs_weight_function))
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.density = np.histogramdd(samples, range=param_ranges, bins=nbins, weights=weights)[0].T

    @property
    def averaged(self):
        return self.density

class DensitySamples(object):

    """
    This class combins several instances of SingleDensity, that are combined by averaging
    """

    def __init__(self, data_list, param_names, weight_list, param_ranges=None, bandwidth_scale=0.6,
                 nbins=12, use_kde=False, samples_width_scale=3):

        """

        :param data_list: a list of observations, each with shape = (n_observations, n_dimensions)
        :param param_names: a list of parameter names
        :param weight_list: a list of importance weights for each element of data_list, or "None"
        :param param_ranges: a list of parameter ranges; if not specified, the parameter ranges will be estimated from the first
        dataset in data_list
        :param bandwidth_scale: rescales the kernel bandwidth, only applicable for Gaussian KDE
        :param nbins: number of bins for histogram/KDE
        :param use_kde: bool; whether or not to use a Gaussian KDE
        :param samples_width_scale: used to estimate the parameter ranges, determines parameter ranges using
        samples_width_scale standard deviations of the 1st dataset; only relevant if param_ranges is not specified
        """

        if not isinstance(data_list, list):
            data_list = [data_list]
        if weight_list is not None:
            if not isinstance(weight_list, list):
                weight_list = [weight_list]
            assert len(weight_list) == len(data_list), 'length of importance sampling weights must equal ' \
                                                       'length of data list'

        self.single_densities = []

        self._n = 0

        if weight_list is None:
            weight_list = [None] * len(data_list)

        self._first_data = data_list[0]
        if self._first_data.shape[1] > self._first_data.shape[0]:
            raise Exception('you have specified samples that have more dimensions that data points, '
                            'this is probably not what you want to do. data_list should be a list of datasets with'
                            'size (n_observations, n_dimensions)')
        self._width_scale = samples_width_scale

        if param_ranges is None:
            means = [np.mean(self._first_data[:, i]) for i in range(0, self._first_data.shape[1])]
            widths = [np.std(self._first_data[:, i]) for i in range(0, self._first_data.shape[1])]
            self.param_ranges = [[mu - samples_width_scale * s, mu + samples_width_scale * s] for mu, s in zip(means, widths)]
        else:
            self.param_ranges = param_ranges

        self.param_names = param_names

        for j, (data, weights) in enumerate(zip(data_list, weight_list)):
            self._n += 1
            self.single_densities.append(SingleDensity(data, param_names, self.param_ranges, weights,
                                                       bandwidth_scale, nbins, use_kde))

    def projection_1D(self, pname):

        """
        Returns the 1D marginal pdf of the parameter 'pname' by averaging over each observation in data list
        :param pname: parameter name
        :return: 1D averaged pdf
        """
        proj = 0
        for den in self.single_densities:
            proj += den.projection_1D(pname)
        return proj * np.max(proj) ** -1

    def projection_2D(self, p1, p2):

        """
        Returns the 2D marginal pdf of the parameters 'p1', 'p2' averaged over each observation in data list
        :param p1: parameter name 1
        :param p2: parameter name 2
        :return: 2D averaged pdf
        """
        proj = 0
        for den in self.single_densities:
            proj += den.projection_2D(p1, p2)
        return proj

    @property
    def averaged(self):

        """
        Computes the average of each PDF in N-dimensions
        :return: an array with shape (nbins, nbins)
        """
        avg = 0
        for di in self.single_densities:
            avg += di.density
        return avg
