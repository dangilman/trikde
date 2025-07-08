import numpy as np
from trikde.kde import KDE, LinearKDE
from trikde.kde import BoundaryCorrection
from scipy.interpolate import RegularGridInterpolator, interp1d
import numpy as np
from multiprocessing.pool import Pool

class InterpolatedLikelihood(object):
    """
    This class interpolates a likelihood, and returns a probability given a point in parameter space
    """
    def __init__(self, independent_densities, param_names, param_ranges, extrapolate=False):

        self.density = independent_densities.density.T
        self.param_names, self.param_ranges = param_names, param_ranges
        nbins = np.shape(self.density)[0]
        points = []
        for ran in param_ranges:
            points.append(np.linspace(ran[0], ran[-1], nbins))
        if extrapolate:
            kwargs_interpolator = {'bounds_error': False, 'fill_value': None}
        else:
            kwargs_interpolator = {}

        self._extrapolate = extrapolate

        self.interp = RegularGridInterpolator(points, self.density, **kwargs_interpolator)

    def sample(self, n, nparams=None, pranges=None, print_progress=True):

        """
        Generates n samples of the parameters in param_names from the likelihood through rejection sampling

        if param_names is None, will generate samples of all param_names used to initialize the class
        """

        if nparams is None:
            nparams = len(self.param_names)
        if pranges is None:
            pranges = self.param_ranges

        shape = (n, nparams)
        samples = np.empty(shape)
        count = 0
        last_ratio = 0

        readout = n / 10

        while True:
            proposal = []
            for ran in pranges:
                proposal.append(np.random.uniform(ran[0], ran[1]))
            proposal = tuple(proposal)
            like = self(proposal)
            u = np.random.uniform(0, 1)
            if like >= u:
                samples[count, :] = proposal
                count += 1
            if count == n:
                break

            current_ratio = count/n

            if print_progress:
                if count%readout == 0 and last_ratio != current_ratio:
                    print('sampling.... '+str(np.round(100*count/n, 2))+'%')
                    last_ratio = count/n

        return samples

    def __call__(self, point):

        """
        Evaluates the liklihood at a point in parameter space
        :param point: a tuple with length equal to len(param_names) that contains a point in parameter space, param ordering
        between param_names, param_ranges, and in point must be the same

        Returns the likelihood
        """
        point = np.array(point)
        for i, value in enumerate(point):
            if value is None:
                new_point = np.random.uniform(self.param_ranges[i][0], self.param_ranges[i][1])
                point[i] = new_point
            elif value < self.param_ranges[i][0] or value > self.param_ranges[i][1]:
                if self._extrapolate is False:
                    raise Exception('point was out of bounds: ', point)

        p = float(self.interp(point))
        if p>1:
            print('warning: p>1, possibly due to extrapolation')
        return min(1., max(p, 0.))

    def call_at_points(self, point_list):

        """
        Evaluates the liklihood at a set of points in parameter space
        :param point_list: a list of tuples at which to evaluate likelihood

        Returns the likelihood
        """
        f = []
        for point in point_list:
            f.append(self(point))
        return np.array(f)

class GaussianWeight(object):

    def __init__(self, mean, sigma):

        self.mean, self.sigma = mean, sigma

    def __call__(self, values):

        dx = (self.mean - values)/self.sigma
        exponent = -0.5 * dx ** 2
        w = np.exp(exponent)
        norm = np.max(w)
        return w/norm


class IndependentLikelihoods(object):

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

    def __mul__(self, other):
        density = self.densities + other.densities
        return IndependentLikelihoods(density)

    def __add__(self, other):
        assert len(other.densities) == len(self.densities)
        density_list = []
        for i in range(0, len(other.densities)):
            density_list.append(self.densities[i] + other.densities[i])
        return IndependentLikelihoods(density_list)

    def __truediv__(self, other):
        # here we want to divide this pdf
        self_pdf = self.density
        # by this one
        other_pdf = other.density
        # there we go...
        density = self_pdf / other_pdf
        # now we want to create a new DensitySamples class with this histogram/kde representation of the data
        pdf = DensitySamples.from_histogram(density,
                                            self.param_names,
                                            self.param_ranges)
        return IndependentLikelihoods([pdf])

    @property
    def density(self):

        if not hasattr(self, '_product'):
            prod = 1
            for di in self.densities:
                prod *= di.density
            prod /= np.max(prod)
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

    def __init__(self, chi_square_function, param_names, param_ranges, nbins, kwargs_weight_function={},
                 renormalize=False, N_samples_per_dim=3000000):

        """
        Used to multiply by a new prior
        :param chi_square_function: a function that computes the chi_square value from an array of samples
        :param param_names: the parameter names
        :param param_ranges: the sampling ranges for each parameter
        :param nbins: the number of histogram/kde bins
        :param kwargs_weight_function: keyword arguments for the weight function
        :param renormalize: apply a reweighting to remove possible edge effects
        """
        N = N_samples_per_dim * len(param_names)
        shape = (N, len(param_names))
        samples = np.empty(shape)
        for i, param in enumerate(param_names):
            samples[:, i] = np.random.uniform(param_ranges[i][0], param_ranges[i][1], N)
        weights = np.exp(-0.5 * chi_square_function(samples, **kwargs_weight_function))
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.density = np.histogramdd(samples, range=param_ranges, bins=nbins, weights=weights)[0].T

        self.renormalization = np.ones_like(self.density)

        if renormalize:
            boundary_correction = BoundaryCorrection(self.density)
            self.renormalization = boundary_correction.first_order_correction

        self.density *= self.renormalization ** -1

    @property
    def densities(self):
        pdf = DensitySamples.from_histogram(self.density,
                                            self.param_names,
                                            self.param_ranges)
        return [pdf]


    @property
    def averaged(self):
        return self.density

def CI(samples, level=68):
    import scipy.stats as st

    # example data set

    pk, _xk = np.histogram(samples, bins=100, range=(min(samples), max(samples)))
    xk = _xk[0:-1] + (_xk[1] - _xk[0]) / 2
    pk = pk / np.sum(pk)
    # create custom discrete random variable from data set
    rv = st.rv_discrete(values=(xk, pk))

    # scipy.stats.rv_discrete has methods for median, confidence interval, etc.
    #print("median:", rv.median())
    #print("68% CI:", rv.interval(level/100))
    return rv.median(), rv.interval(level/100)

def estimate_parameter_ranges(data, CI_level=1, scale_width=1.5):
    """

    :param data:
    :param CI_level:
    :param scale_width:
    :return:
    """
    param_ranges = []
    if CI_level == 1:
        level = 68
    elif CI_level == 2:
        level = 95
    elif CI_level == 3:
        level = 99
    else:
        raise ValueError('samples_width_scale must be 1, 2, or 3 when estimating parameter ranges; numbers '
                         'correspond to 68%, 95%, and 99% CI, respectively')
    for i in range(data.shape[1]):
        med, interval = CI(data[:, i], level)
        step_low = med - interval[0]
        step_high = interval[1] - med
        new_interval = [med - scale_width*step_low, med + scale_width*step_high]
        param_ranges.append(new_interval)
    return param_ranges

class DensitySamples(object):
    """
    This class combins several instances of SingleDensity, that are combined by averaging
    """
    def __init__(self, data, param_names, weights, param_ranges=None, bandwidth_scale=0.6,
                 nbins=12, use_kde='LINEAR', samples_width_scale=3, density=None,
                 nbins_eval=None, resampling=True, n_resample=1000000):

        """

        :param data: an array of observations, shape = (n_observations, n_dimensions)
        :param param_names: a list of parameter names
        :param weights: importance weights
        :param param_ranges: a list of parameter ranges; if not specified, the parameter ranges will be estimated from the first
        dataset in data_list
        :param bandwidth_scale: rescales the kernel bandwidth, only applicable for Gaussian KDE
        :param nbins: number of bins for histogram/KDE
        :param use_kde: bool or string; specifies the type of KDE to use, options are 'GAUSSIAN', 'LINEAR', or False (no kde)
        :param samples_width_scale: used to estimate the parameter ranges, determines parameter ranges using
        samples_width_scale standard deviations of the 1st dataset; only relevant if param_ranges is not specified
        :param density: a numpy array of shape (ndim, ndim) that specifies the relative probability across the parameter space
            if this is specified, data is not used
        :param nbins_eval: the number of bins on which to output the final pdf if using kde_type='LINEAR'
        :param resampling: when using LINEAR kde, specifies whether the final pdf is obtained from resampling the prior
        and weighting by interpolation function
        :param n_resample: the number of points to draw when resampling
        """
        if density is not None:
            self.density = density
            assert param_ranges is not None, 'When specifying the density directly, must also provide the parameter ranges ' \
                                             'along each dimension'
            self.param_ranges = param_ranges
            assert param_names is not None, 'When specifying the density directly, must also provide the parameter names'
            self.param_names = param_names
        else:
            if data.shape[1] > data.shape[0]:
                raise Exception('you have specified samples that have more dimensions that data points, '
                                'this is probably not what you want to do. data_list should be a list of datasets with'
                                'size (n_observations, n_dimensions)')

            if use_kde == 'GAUSSIAN':
                estimator = KDE(bandwidth_scale, nbins)
            elif use_kde == 'GAUSSIAN_NO_COV':
                estimator = KDE(bandwidth_scale, nbins, use_cov=False)
            elif use_kde == 'LINEAR':
                if nbins_eval is None:
                    nbins_eval = nbins
                estimator = LinearKDE(nbins,
                                      nbins_eval,
                                      resampling=resampling,
                                      n_resample=n_resample)
            elif use_kde is False or use_kde is None:
                # we still use the NDHistogram
                estimator = KDE(bandwidth_scale, nbins)
            else:
                raise ValueError('kde_type must be GAUSSIAN, GAUSSIAN_NO_COV, or LINEAR')

            if param_ranges is None:
                self.param_ranges = estimate_parameter_ranges(data, samples_width_scale)
            else:
                self.param_ranges = param_ranges
            self.param_names = param_names
            if use_kde:
                self.density = estimator(data, self.param_ranges, weights)
            else:
                self.density = estimator.NDhistogram(data, weights, self.param_ranges)
        self.density /= np.max(self.density)

    def __add__(self, other):
        for name_self, name_other in zip(self.param_names, other.param_names):
            assert name_self == name_other
        for range_self, range_other in zip(self.param_ranges, other.param_ranges):
            assert range_self[0] == range_other[0]
            assert range_self[1] == range_other[1]
        pdf_self = self.density
        pdf_other = other.density
        return DensitySamples.from_histogram(pdf_self + pdf_other, self.param_names, self.param_ranges)

    @classmethod
    def from_histogram(cls, density, param_names, param_ranges):
        """
        Create an instance of the class from a histogram
        :param density: the histogram in ndim dimensions
        :param param_names: parameter names; len(param_names) = ndim
        :param param_ranges: parameter ranges along each dimension; len(parm_ranges) = ndim
        :return:
        """
        return DensitySamples(None, param_names, None, param_ranges, density=density)

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

def likelihood_function_change(like1, like2, param_ranges, n_draw=50000, nbins=5):
    """
    This function evaluates the change or derivative between two two-dimensional likelihood functions

    :param like1: the first likelihood
    :param like2: the second likelihood
    :param param_names: parameter names for each axis
    :param param_ranges: axis ranges in each dimension
    :param n_draw: the number of samples to draw from each likelihood to compute the derivative
    :param nbins: the number of bins along each axis
    :return: the relative difference in the likelihood functions, the likelihoods themselves, and the
    total change per bin
    """
    from trikde.pdfs import InterpolatedLikelihood
    param_names = ['param1', 'param2']
    interp1 = InterpolatedLikelihood(like1, param_names, param_ranges)
    interp2 = InterpolatedLikelihood(like2, param_names, param_ranges)
    s1 = interp1.sample(n_draw)
    s2 = interp2.sample(n_draw)
    h1, _, _ = np.histogram2d(s1[:,0], s1[:,1], bins=nbins, range=(param_ranges[0], param_ranges[1]),density=True)
    h2, _, _ = np.histogram2d(s2[:,0], s2[:,1], bins=nbins, range=(param_ranges[0], param_ranges[1]),density=True)
    delta_h = h2 / h1 - 1
    total_diff = np.sum(np.absolute(delta_h)) / nbins ** 2
    return delta_h.T, h1.T, h2.T, total_diff

def figure_of_merit(interp_likelihiood, n_draw=10000):
    n = 0
    for i in range(0, n_draw):
        proposal = []
        for ran in interp_likelihiood.param_ranges:
            proposal.append(np.random.uniform(ran[0], ran[1]))
        proposal = tuple(proposal)
        like = interp_likelihiood(proposal)
        u = np.random.rand()
        if like > u:
            n += 1
    return n_draw/n

def posterior_volume_ratio(interp_likelihood_1, interp_likelihood_2,n_draw=5000):

    volume_1, volume_2 = 0, 0
    assert interp_likelihood_1.param_ranges == interp_likelihood_2.param_ranges
    for ran_1, ran_2 in zip(interp_likelihood_1.param_ranges, interp_likelihood_2.param_ranges):
        assert ran_1[0] == ran_2[0]
        assert ran_1[1] == ran_2[1]
    for n in range(0, n_draw):
        proposal = []
        for ran in interp_likelihood_1.param_ranges:
            proposal.append(np.random.uniform(ran[0], ran[1]))
        proposal = tuple(proposal)
        like_1 = interp_likelihood_1(proposal)
        like_2 = interp_likelihood_2(proposal)
        u = np.random.rand()
        if like_1 > u:
            volume_1 += 1
        if like_2 > u:
            volume_2 += 1
    posterior_volume_relative_to_prior_1 = volume_1 / n_draw
    posterior_volume_relative_to_prior_2 = volume_2 / n_draw
    return posterior_volume_relative_to_prior_1/posterior_volume_relative_to_prior_2

