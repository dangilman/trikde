from matplotlib import colors
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

class TrianglePlot(object):

    _default_contour_colors = [(colors.cnames['darkslategrey'], colors.cnames['black'], 'k'),
                               (colors.cnames['dodgerblue'], colors.cnames['blue'], 'k'),
                               (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k'),
                               (colors.cnames['lightcoral'], colors.cnames['red'], 'k'),
                               (colors.cnames['orange'], colors.cnames['gold'], 'k')]

    truth_color = 'g'
    truth_color_list = None
    _highc_contour_colors = [(colors.cnames['darkslategrey'], colors.cnames['black'], 'k'),
                               (colors.cnames['lightgreen'], colors.cnames['green'], 'k'),
                               (colors.cnames['plum'], colors.cnames['darkviolet'], 'k'),
                               (colors.cnames['navajowhite'], colors.cnames['darkorange'], 'k')]

    spacing = np.array([0.1, 0.1, 0.05, 0.05, 0.2, 0.11])
    spacing_scale = 1.
    _tick_rotation = 0
    _color_eval = 0.9
    show_intervals_68 = False

    def __init__(self, independent_likelihoods_list, param_ranges=None, cmap='gist_heat', param_name_transformation=None):

        """

        :param independent_likelihoods_list: a list of IndependentLikelihoods classes (see trikde.pdfs)
        :param cmap: name of the color map to use if not using filled contours
        :param custom_ticks:
        """

        self.param_name_transformation = param_name_transformation
        self.param_names = independent_likelihoods_list[0].param_names
        self._nchains = len(independent_likelihoods_list)
        if param_ranges is None:
            parameter_ranges = independent_likelihoods_list[0].param_ranges
        else:
            parameter_ranges = param_ranges

        if isinstance(parameter_ranges, list):
            self._prange_list = parameter_ranges
            self.parameter_ranges = {}
            for i, pname in enumerate(self.param_names):
                self.parameter_ranges.update({pname:parameter_ranges[i]})
        elif isinstance(parameter_ranges, dict):
            self.parameter_ranges = parameter_ranges
            self._prange_list = []
            for pi in self.param_names:
                self._prange_list.append(self.parameter_ranges[pi])

        self._NDdensity_list = independent_likelihoods_list
        
        self.set_cmap(cmap)

    def _load_projection_1D(self, pname, idx):

        return self._NDdensity_list[idx].projection_1D(pname)

    def _load_projection_2D(self, p1, p2, idx):

        return self._NDdensity_list[idx].projection_2D(p1, p2)

    def set_cmap(self, newcmap, color_eval=0.9, marginal_col=None):

        self.cmap = newcmap
        self.cmap_call = plt.get_cmap(newcmap)
        self._color_eval = color_eval
        self._marginal_col = marginal_col

    def contour_area(self, p1, p2, pdf_index):

        density = self._load_projection_2D(p1, p2, pdf_index)
        extent, aspect = self._extent_aspect([p1, p2])
        pmin1, pmax1 = extent[0], extent[1]
        pmin2, pmax2 = extent[2], extent[3]

        coordsx = np.linspace(extent[0], extent[1], density.shape[0])
        coordsy = np.linspace(extent[2], extent[3], density.shape[1])

        levels = [0.05, 0.32, 1.0]
        levels = np.array(levels) * np.max(grid)
        X, Y = np.meshgrid(x, y)

        ax = plt.gca()
        contours = ax.contour(X, Y, density, levels, extent=extent,
                       colors=contour_colors, linewidths=linewidths, zorder=1, linestyles=['dashed', 'solid'])

        self._contours(coordsx, coordsy, density, ax, extent=extent,
                       contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                       levels=levels)
        ax.set_xlim(pmin1, pmax1)
        ax.set_ylim(pmin2, pmax2)

    def make_joint(self, p1, p2, contour_colors=None, levels=[0.05, 0.22, 1],
                   filled_contours=True, contour_alpha=0.6,
                   fig_size=8, label_scale=1, tick_label_font=12,
                     xtick_label_rotate=0, show_contours=True,norm=None,
                   logscale=False,vmin=None,vmax=None):

        self.fig = plt.figure(1)
        self._init(fig_size)
        ax = plt.subplot(111)

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        if contour_colors == 'HighContrast':
            contour_colors = self._highc_contour_colors

        ims = []
        for i in range(self._nchains):
            axes, im = self._make_joint_i(p1, p2, ax, i, contour_colors=contour_colors, levels=levels,
                      filled_contours=filled_contours, contour_alpha=contour_alpha,
                      labsize=15*label_scale, tick_label_font=tick_label_font,
                               xtick_label_rotate=xtick_label_rotate,
                                          show_contours=show_contours,
                                          norm=norm,logscale=logscale,vmin=vmin,vmax=vmax)
            ims.append(im)
        return axes, ims

    def make_triplot(self, contour_levels=[0.05, 0.32, 1],
                     filled_contours=True, contour_alpha=0.6,
                     fig_size=8, truths=None, contour_colors=None,
                     axis_label_font=16, tick_label_font=12,
                     xtick_label_rotate=0, show_contours=True,
                     marginal_alpha=0.6, show_intervals=True,
                     display_params=None, figure=1):

        self.fig = plt.figure(figure)

        self._init(fig_size)

        axes = []
        counter = 1
        if display_params is None:
            display_params = self.param_names
        n_subplots = len(display_params)

        gs1 = gridspec.GridSpec(n_subplots, n_subplots)
        gs1.update(wspace=0.15, hspace=0.15)

        for row in range(n_subplots):
            for col in range(n_subplots):
                axes.append(plt.subplot(gs1[counter-1]))
                counter += 1

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        if contour_colors == 'HighContrast':
            contour_colors = self._highc_contour_colors
        self._auto_scale = []

        for i in range(self._nchains):

            axes.append(self._make_triplot_i(axes, i, contour_colors, contour_levels, filled_contours, contour_alpha,
                                             fig_size, truths, tick_label_font=tick_label_font,
                                             xtick_label_rotate=xtick_label_rotate,
                                             axis_label_font=axis_label_font, cmap=self.cmap_call, show_contours=show_contours,
                                             marginal_alpha=marginal_alpha, show_intervals=show_intervals,
                                             display_params=display_params))

        for key in display_params:
            max_h = []
            for scale in self._auto_scale:

                max_h.append(scale[key][1])
                plot_index = scale[key][0]
            max_h = max(max_h)

            axes[plot_index].set_ylim(0., 1.1 * max_h)

        self._auto_scale = []
        plt.subplots_adjust(left=self.spacing[0] * self.spacing_scale, bottom=self.spacing[1] * self.spacing_scale,
                            right=1 - self.spacing[2] * self.spacing_scale,
                            top=1 - self.spacing[3] * self.spacing_scale,
                            wspace=self.spacing[4] * self.spacing_scale, hspace=self.spacing[5] * self.spacing_scale)

        return axes

    def make_marginal(self, p1, contour_colors=None, levels=[0.05, 0.32, 1],
                      filled_contours=True, contour_alpha=0.6, param_names=None,
                      fig_size=8, truths=None, load_from_file=True,
                      transpose_idx=None, bandwidth_scale=0.7, label_scale=1,
                      cmap=None, xticklabel_rotate=0, bar_alpha=0.7, bar_colors=['k','b','m','r'],
                      height_scale=1.1, show_low=False, show_high=False):

        self.fig = plt.figure(1)
        self._init(fig_size)
        ax = plt.subplot(111)
        self._auto_scale = []

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        if contour_colors == 'HighContrast':
            contour_colors = self._highc_contour_colors
        self._auto_scale = []
        for i in range(self._nchains):
            out = self._make_marginal_i(p1, ax, i, contour_colors, levels, filled_contours, contour_alpha, param_names,
                                  fig_size, truths, load_from_file=load_from_file,
                                  transpose_idx=transpose_idx, bandwidth_scale=bandwidth_scale,
                                  label_scale=label_scale, cmap=cmap, xticklabel_rotate=xticklabel_rotate,
                                  bar_alpha=bar_alpha, bar_color=bar_colors[i], show_low=show_low, show_high=show_high)

        scales = []
        for c in range(0, self._nchains):
            scales.append(self._auto_scale[c][0])
        maxh = np.max(scales) * height_scale
        ax.set_ylim(0, maxh)
        pmin, pmax = self._get_param_minmax(p1)
        asp = maxh * (pmax - pmin) ** -1
        ax.set_aspect(asp ** -1)

        self._auto_scale = []

        return out

    def _make_marginal_i(self, p1, ax, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                         filled_contours=True, contour_alpha=0.6, param_names=None, fig_size=8,
                         truths=None, labsize=15, tick_label_font=14,
                         load_from_file=True, transpose_idx=None,
                         bandwidth_scale=0.7, label_scale=None, cmap=None, xticklabel_rotate=0,
                         bar_alpha=0.7, bar_color=None, show_low=False, show_high=False):
        if contour_colors[-1] == ('#FFDEAD', '#FF8C00', 'k'): # high-contrast flag
            self.truth_color = 'dodgerblue'
        autoscale = []

        density = self._load_projection_1D(p1, color_index)

        xtick_locs, xtick_labels, xlabel, rotation = self.ticks_and_labels(p1)
        pmin, pmax = self._get_param_minmax(p1)

        coords = np.linspace(pmin, pmax, len(density))

        bar_centers, bar_width, bar_heights = self._bar_plot_heights(density, coords, None)

        bar_heights *= np.sum(bar_heights) ** -1 * len(bar_centers) ** -1
        autoscale.append(np.max(bar_heights))

        max_idx = np.argmax(bar_heights)

        for i, y in enumerate(bar_heights):
            x1, x2 = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5

            ax.plot([x1, x2], [y, y], color=bar_color,
                        alpha=bar_alpha)
            ax.fill_between([x1, x2], y, color=bar_color,
                            alpha=0.6)
            ax.plot([x1, x1], [0, y], color=bar_color,
                    alpha=bar_alpha)
            ax.plot([x2, x2], [0, y], color=bar_color,
                    alpha=bar_alpha)

        ax.set_xlim(pmin, pmax)

        ax.set_yticks([])

        mean_of_distribution, [low68, high68] = self._confidence_int(pmin, pmax, bar_centers, bar_heights, 1)
        mean_of_distribution, [low95, high95] = self._confidence_int(pmin, pmax, bar_centers, bar_heights, 2)

        mean_of_distribution = 0
        for i in range(0, len(bar_heights)):
            mean_of_distribution += bar_heights[i] * bar_centers[i] / np.sum(bar_heights)

        if low95 is not None and show_low:
            ax.axvline(low95, color=bar_color,
                       alpha=0.8, linewidth=2.5, linestyle='-.')
        if high95 is not None and show_high:
            ax.axvline(high95, color=bar_color,
                       alpha=0.8, linewidth=2.5, linestyle='-.')

        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xticklabel_rotate)
        if xlabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_xlabel(xlabel, fontsize=40 * label_scale)
        else:
            ax.set_xlabel(xlabel, fontsize=labsize * label_scale)

        if truths is not None:
            if self.truth_color_list is None:
                self.truth_color_list = [self.truth_color, 'y', 'b', 'g', '0.5']
            if not isinstance(truths, list):
                truths = [truths]
            for idx_truth, truth in enumerate(truths):
                t = deepcopy(truth[p1])
                if isinstance(t, float) or isinstance(t, int):
                    pmin, pmax = self._get_param_minmax(p1)
                    if t <= pmin:
                        t = pmin * 1.075
                    ax.axvline(t, linestyle='--', color=self.truth_color_list[idx_truth], linewidth=3)
                elif isinstance(t, list):
                    ax.axvspan(t[0], t[1], alpha=0.25, color=self.truth_color_list[idx_truth])

        self._auto_scale.append(autoscale)

        return ax

    def _make_joint_i(self, p1, p2, ax, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                      filled_contours=True, contour_alpha=0.6, labsize=None, tick_label_font=None,
                               xtick_label_rotate=None, show_contours=None,
                      norm=None,logscale=False,vmin=None,vmax=None):

        density = self._load_projection_2D(p1, p2, color_index)

        extent, aspect = self._extent_aspect([p1, p2])
        pmin1, pmax1 = extent[0], extent[1]
        pmin2, pmax2 = extent[2], extent[3]

        xtick_locs, xtick_labels, xlabel, rotation = self.ticks_and_labels(p1)
        ytick_locs, ytick_labels, ylabel, _ = self.ticks_and_labels(p2)

        if filled_contours:
            if logscale:
                print('WARNING: you specified log_scale = True, but this has no effect when filled_contours=True')
            if vmin is None or vmax is None:
                print('WARNING: you specified vmin or vmax, but these arguments have no effect when filled_contours=True')
            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
            coordsy = np.linspace(extent[2], extent[3], density.shape[1])

            im = ax.imshow(density, extent=extent, aspect=aspect,
                      origin='lower', cmap=self.cmap, alpha=0, norm=norm)
            self._contours(coordsx, coordsy, density, ax, extent=extent,
                           contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                           levels=levels)
            ax.set_xlim(pmin1, pmax1)
            ax.set_ylim(pmin2, pmax2)

        else:
            if logscale:
                d = np.log10(density / np.max(density))
                if vmin is None:
                    vmin = -5.0
                if vmax is None:
                    vmax = 0.0
            else:
                d = density / np.max(density)
                if vmin is None:
                    #vmin = np.min(d)
                    vmin = 0.0
                if vmax is None:
                    vmax = 1.0
            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
            coordsy = np.linspace(extent[2], extent[3], density.shape[1])
            im = ax.imshow(d, origin='lower', cmap=self.cmap, alpha=1, vmin=vmin,
                      vmax=vmax, aspect=aspect, extent=extent, norm=norm)
            if show_contours:
                self._contours(coordsx, coordsy, density, ax, extent=extent, filled_contours=False,
                           contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                           levels=levels)
            ax.set_xlim(pmin1, pmax1)
            ax.set_ylim(pmin2, pmax2)

        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xtick_label_rotate)

        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_labels, fontsize=tick_label_font)

        if xlabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_xlabel(xlabel, fontsize=40)
        elif ylabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_ylabel(ylabel, fontsize=40)
        else:
            ax.set_xlabel(xlabel, fontsize=labsize)
            ax.set_ylabel(ylabel, fontsize=labsize)

        return ax, im

    def _make_triplot_i(self, axes, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                        filled_contours=True, contour_alpha=0.6, fig_size=8,
                        truths=None, tick_label_font=14, xtick_label_rotate=0,
                        axis_label_font=None, cmap=None,
                        show_contours=True, marginal_alpha=0.9, show_intervals=True,
                        display_params=None):

        if contour_colors[-1] == ('#FFDEAD', '#FF8C00', 'k'): # high-contrast flag
            self.truth_color = 'dodgerblue'
        size_scale = len(display_params) * 0.1 + 1
        self.fig.set_size_inches(fig_size * size_scale, fig_size * size_scale)

        marg_in_row, plot_index = 0, 0
        n_subplots = len(display_params)
        self._reference_grid = None
        autoscale = {}

        self.triplot_densities = []
        self.joint_names = []
        row = 0
        col = 0

        for _ in range(n_subplots):

            marg_done = False
            for _ in range(n_subplots):

                if self.param_names[row] not in display_params:
                    continue
                elif self.param_names[col] not in display_params:
                    continue

                if col < marg_in_row:

                    density = self._load_projection_2D(display_params[row], display_params[col], color_index)
                    self.triplot_densities.append(density)
                    self.joint_names.append(display_params[row]+'_'+display_params[col])

                    extent, aspect = self._extent_aspect([display_params[col], display_params[row]])
                    pmin1, pmax1 = extent[0], extent[1]
                    pmin2, pmax2 = extent[2], extent[3]

                    xtick_locs, xtick_labels, xlabel, rotation = self.ticks_and_labels(display_params[col])
                    ytick_locs, ytick_labels, ylabel, _ = self.ticks_and_labels(display_params[row])

                    if row == n_subplots - 1:

                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font,
                                                         rotation=xtick_label_rotate)

                        if col == 0:
                            axes[plot_index].set_yticks(ytick_locs)
                            axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                            axes[plot_index].set_ylabel(ylabel, fontsize=axis_label_font)
                        else:
                            axes[plot_index].set_yticks([])
                            axes[plot_index].set_yticklabels([])

                        axes[plot_index].set_xlabel(xlabel, fontsize=axis_label_font)


                    elif col == 0:
                        axes[plot_index].set_yticks(ytick_locs)
                        axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                        axes[plot_index].set_xticks([])
                        axes[plot_index].set_ylabel(ylabel, fontsize=axis_label_font)

                    else:
                        axes[plot_index].set_xticks([])
                        axes[plot_index].set_yticks([])
                        axes[plot_index].set_xticklabels([])
                        axes[plot_index].set_yticklabels([])

                    if filled_contours:
                        coordsx = np.linspace(extent[0], extent[1], density.shape[0])
                        coordsy = np.linspace(extent[2], extent[3], density.shape[1])
                        
                        vmax = np.max(density)
                        axes[plot_index].imshow(density.T, extent=extent, aspect=aspect,
                                                origin='lower', cmap=self.cmap, alpha=0, vmin=0, vmax=vmax)
                        self._contours(coordsx, coordsy, density.T, axes[plot_index], extent=extent,
                                       contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                                       levels=levels)
                        axes[plot_index].set_xlim(pmin1, pmax1)
                        axes[plot_index].set_ylim(pmin2, pmax2)

                    else:

                        vmax = np.max(density)
                        axes[plot_index].imshow(density.T, origin='lower', cmap=self.cmap, alpha=1, vmin=0,
                                                vmax=vmax, aspect=aspect, extent=extent)

                        if show_contours:
                            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
                            coordsy = np.linspace(extent[2], extent[3], density.shape[1])
                            self._contours(coordsx, coordsy, density.T, axes[plot_index], filled_contours=False,
                                           extent=extent,
                                           contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                                           levels=levels)
                        axes[plot_index].set_xlim(pmin1, pmax1)
                        axes[plot_index].set_ylim(pmin2, pmax2)

                    axes[plot_index].set_xlim(pmin1, pmax1)
                    axes[plot_index].set_ylim(pmin2, pmax2)

                    if truths is not None:
                        if isinstance(truths, dict):
                            t1, t2 = truths[display_params[col]], truths[display_params[row]]
                            axes[plot_index].scatter(t1, t2, color=self.truth_color, s=50)
                            axes[plot_index].axvline(t1, linestyle='--', color=self.truth_color, linewidth=3)
                            axes[plot_index].axhline(t2, linestyle='--', color=self.truth_color, linewidth=3)
                        else:
                            assert isinstance(truths, list), 'if specified, truths must be a dictionary or a ' \
                                                             'list of dictionaries'
                            if self.truth_color_list is None:
                                self.truth_color_list = [self.truth_color, 'y', '0.6', '0.3']
                            for truth_index in range(0, len(truths)):
                                t1, t2 = truths[truth_index][display_params[col]], truths[truth_index][display_params[row]]
                                axes[plot_index].scatter(t1, t2, color=self.truth_color_list[truth_index], s=50)
                                axes[plot_index].axvline(t1, linestyle='--', color=self.truth_color_list[truth_index], linewidth=3)
                                axes[plot_index].axhline(t2, linestyle='--', color=self.truth_color_list[truth_index], linewidth=3)

                elif marg_in_row == col and marg_done is False:

                    marg_done = True
                    marg_in_row += 1

                    density = self._load_projection_1D(display_params[col], color_index)

                    xtick_locs, xtick_labels, xlabel, rotation = self.ticks_and_labels(display_params[col])
                    pmin, pmax = self._get_param_minmax(display_params[col])
                    coords = np.linspace(pmin, pmax, len(density))

                    bar_centers, bar_width, bar_heights = self._bar_plot_heights(density, coords, None)

                    bar_heights *= (np.sum(bar_heights) * len(bar_centers)) ** -1
                    autoscale[display_params[col]] = [plot_index, max(bar_heights)]

                    for i, y in enumerate(bar_heights):
                        x1, x2 = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5

                        if filled_contours:
                            axes[plot_index].plot([x1, x2], [y, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                            axes[plot_index].fill_between([x1, x2], y, color=contour_colors[color_index][1],
                                                          alpha=marginal_alpha)
                            axes[plot_index].plot([x1, x1], [0, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                            axes[plot_index].plot([x2, x2], [0, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                        else:
                            if self._marginal_col is None:
                                marginal_col = cmap(self._color_eval)
                            else:
                                marginal_col = self._marginal_col
                            axes[plot_index].plot([x1, x2], [y, y], color=marginal_col,
                                                  alpha=1)
                            axes[plot_index].fill_between([x1, x2], y, color=marginal_col,
                                                          alpha=marginal_alpha)
                            axes[plot_index].plot([x1, x1], [0, y], color=marginal_col,
                                                  alpha=1)
                            axes[plot_index].plot([x2, x2], [0, y], color=marginal_col,
                                                  alpha=1)

                    axes[plot_index].set_xlim(pmin, pmax)
                    axes[plot_index].set_yticks([])

                    if show_intervals:
                        mean_of_distribution, [low68, high68] = self._confidence_int(pmin, pmax, bar_centers, bar_heights,1)
                        mean_of_distribution, [low95, high95] = self._confidence_int(pmin, pmax, bar_centers, bar_heights,2)

                    if show_intervals and low95 is not None:
                        axes[plot_index].axvline(low95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')
                    if show_intervals and high95 is not None:
                        axes[plot_index].axvline(high95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')

                    if self.show_intervals_68 and low68 is not None:
                        axes[plot_index].axvline(low68, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle=':')
                    if self.show_intervals_68 and high68 is not None:
                        axes[plot_index].axvline(high68, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle=':')


                    if col != n_subplots - 1:
                        axes[plot_index].set_xticks([])
                    else:
                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xtick_label_rotate)
                        axes[plot_index].set_xlabel(xlabel, fontsize=axis_label_font)

                    if truths is not None:
                        if self.truth_color_list is None:
                            self.truth_color_list = [self.truth_color, 'k', '0.6', '0.3']
                        if not isinstance(truths, list):
                            truths = [truths]
                        for idx_truth, truth in enumerate(truths):
                            t = deepcopy(truth[display_params[col]])
                            if isinstance(t, float) or isinstance(t, int):
                                pmin, pmax = self._get_param_minmax(display_params[col])
                                if t <= pmin:
                                    t = pmin * 1.075
                                axes[plot_index].axvline(t, linestyle='--', color=self.truth_color_list[idx_truth], linewidth=3)
                            elif isinstance(t, list):
                                axes[plot_index].axvspan(t[0], t[1], alpha=0.25, color=self.truth_color_list[idx_truth])
                    #
                    # if truths is not None:
                    #     t = deepcopy(truths[display_params[col]])
                    #     pmin, pmax = self._get_param_minmax(display_params[col])
                    #     if isinstance(t, float) or isinstance(t, int):
                    #         if t <= pmin:
                    #             t_ = pmin * 1.075
                    #         else:
                    #             t_ = t
                    #         axes[plot_index].axvline(t_, linestyle='--', color=self.truth_color, linewidth=3)
                    #
                    #     else:
                    #         t_ = 0.5*(t[0] + t[1])
                    #         axes[plot_index].axvline(t_, linestyle='--', color=self.truth_color, linewidth=3)
                    #         axes[plot_index].axvspan(t[0], t[1], color=self.truth_color, alpha=0.25)

                else:
                    axes[plot_index].axis('off')

                plot_index += 1
                col += 1

            row += 1
            col = 0

        self._auto_scale.append(autoscale)

    def _confidence_int(self, pmin, pmax, centers, heights, num_sigma, thresh=None):

        centers = np.array(centers)
        heights = np.array(heights)
        heights *= np.max(heights) ** -1

        prob_interp = interp1d(centers, heights, bounds_error=False,
                               fill_value=0)

        samples = []

        while len(samples)<10000:

            samp = np.random.uniform(pmin, pmax)
            prob = prob_interp(samp)
            u = np.random.uniform(0,1)

            if prob >= u:
                samples.append(samp)
        #print('num sigma:', num_sigma)
        mu, sigmas = compute_confidence_intervals(samples, num_sigma, thresh)

        return mu, [mu-sigmas[0], mu+sigmas[1]]

    def _extent_aspect(self, param_names):

        aspect = (self.parameter_ranges[param_names[0]][1] - self.parameter_ranges[param_names[0]][0]) * \
                 (self.parameter_ranges[param_names[1]][1] - self.parameter_ranges[param_names[1]][0]) ** -1

        extent = [self.parameter_ranges[param_names[0]][0], self.parameter_ranges[param_names[0]][1],
                  self.parameter_ranges[param_names[1]][0],
                  self.parameter_ranges[param_names[1]][1]]

        return extent, aspect

    def _init(self, fig_size):

        self._tick_lab_font = 12 * fig_size * 7 ** -1
        self._label_font = 15 * fig_size * 7 ** -1
        plt.rcParams['axes.linewidth'] = 2.5 * fig_size * 7 ** -1

        plt.rcParams['xtick.major.width'] = 2.5 * fig_size * 7 ** -1
        plt.rcParams['xtick.major.size'] = 6 * fig_size * 7 ** -1
        plt.rcParams['xtick.minor.size'] = 2 * fig_size * 7 ** -1

        plt.rcParams['ytick.major.width'] = 2.5 * fig_size * 7 ** -1
        plt.rcParams['ytick.major.size'] = 6 * fig_size * 7 ** -1
        plt.rcParams['ytick.minor.size'] = 2 * fig_size * 7 ** -1

    def _get_param_minmax(self, pname):

        ranges = self.parameter_ranges[pname]

        return ranges[0], ranges[1]

    def _get_param_inds(self, params):

        inds = []

        for pi in params:

            for i, name in enumerate(self.param_names):

                if pi == name:
                    inds.append(i)
                    break

        return np.array(inds)

    def _bar_plot_heights(self, bar_heights, coords, rebin):

        if rebin is not None:
            new = []
            if len(bar_heights) % rebin == 0:
                fac = int(len(bar_heights) / rebin)
                for i in range(0, len(bar_heights), fac):
                    new.append(np.mean(bar_heights[i:(i + fac)]))

                bar_heights = np.array(new)
            else:
                raise ValueError('must be divisible by rebin.')

        bar_width = np.absolute(coords[-1] - coords[0]) * len(bar_heights) ** -1
        bar_centers = []
        for i in range(0, len(bar_heights)):
            bar_centers.append(coords[0] + bar_width * (0.5 + i))

        integral = np.sum(bar_heights) * bar_width * len(bar_centers) ** -1

        bar_heights = bar_heights * integral ** -1

        return bar_centers, bar_width, bar_heights

    def _contours(self, x, y, grid, ax, linewidths=4, filled_contours=True, contour_colors='',
                  contour_alpha=1., extent=None, levels=[0.05, 0.32, 1]):

        levels = np.array(levels) * np.max(grid)
        X, Y = np.meshgrid(x, y)

        if filled_contours:

            ax.contour(X, Y, grid, levels, extent=extent,
                       colors=contour_colors, linewidths=linewidths, zorder=1, linestyles=['dashed', 'solid'])

            ax.contourf(X, Y, grid, [levels[0], levels[1]], colors=[contour_colors[0], contour_colors[1]],
                        alpha=contour_alpha * 0.5, zorder=1,
                        extent=extent)

            ax.contourf(X, Y, grid, [levels[1], levels[2]], colors=[contour_colors[1], contour_colors[2]],
                        alpha=contour_alpha, zorder=1,
                        extent=extent)

        else:
            ax.contour(X, Y, grid, extent=extent, colors=contour_colors, zorder=1,
                       levels=levels,
                       linewidths=linewidths)

    def ticks_and_labels(self, pname):

        rotation = self._tick_rotation

        decimals, nticks = auto_decimal_places(self.parameter_ranges[pname][0], self.parameter_ranges[pname][1])

        tick_locs = np.round(np.linspace(self.parameter_ranges[pname][0], self.parameter_ranges[pname][1], nticks), decimals)
        tick_labels = tick_locs

        if self.param_name_transformation is None:
            param_name = pname
        else:
            param_name = self.param_name_transformation(pname)

        return tick_locs, tick_labels, param_name, rotation

    def get_parameter_confidence_interval(self, parameter, clevel, chain_num=None,
                                          show_percentage=False, return_intervals=False,
                                          print_intervals=True, thresh=None):

        if print_intervals:
            print('parameter name: ', parameter)
            if thresh is None:
                if show_percentage:
                    print('68% confidence intervals: \nformat: median (lower, upper) (-%, +%)\n')
                else:
                    print('68% confidence intervals: \nformat: median (lower, upper) (param_min, param_max)\n')
            else:
                if show_percentage:
                    print(str(100 * thresh) + '% confidence intervals: \nformat: median (lower, upper) (-%, +%)\n')
                else:
                    print(str(100 * thresh) + '% confidence intervals: \nformat: median (lower, upper)\n')

        medians, uppers, lowers = [], [], []

        for idx in range(0, self._nchains):

            if chain_num is not None:
                if idx != chain_num:
                    continue

            samples = self._load_projection_1D(parameter, idx)
            pmin, pmax = self._get_param_minmax(parameter)

            coords = np.linspace(pmin, pmax, len(samples))
            bar_centers, bar_widths, bar_heights = self._bar_plot_heights(samples, coords, None)

            median, [lower, upper] = self._confidence_int(pmin, pmax, bar_centers, bar_heights, clevel, thresh)

            #chain.append({''})

            if print_intervals:
                print('SAMPLES ' + str(idx + 1) + ':')
                if show_percentage:
                    print(str(median) + ' (' + str(lower) + ', ' + str(upper) + ')')
                else:
                    print(str(median) + ' ('+str(lower)+', '+str(upper)+')')
                print('width: ', upper - lower)
            medians.append(median)
            uppers.append(upper)
            lowers.append(lower)

        if return_intervals:
            return (medians, uppers, lowers)
        else:
            return None

def auto_decimal_places(param_min, param_max):

    nticks = 5

    if param_min == 0:
        OM_low = -1
    else:
        OM_low = int(np.log10(abs(param_min)))

    if param_max == 0:
        OM_high = -1
    else:
        OM_high = int(np.log10(abs(param_max)))

    OM_min = min(OM_low, OM_high)

    if OM_min > 0:
        decimals = 0
    else:
        decimals = abs(OM_min) + 2

    dynamic_range = abs(OM_high - OM_low)
    if dynamic_range > 0:
        decimals += 0
    else:
        decimals += 1

    if decimals > 2:
        nticks -= 1
    if decimals > 3:
        nticks -= 1
    if decimals > 4:
        nticks -= 1

    return decimals, nticks

def compute_confidence_intervals_histogram(sample, num_sigma):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restricted to three. %s not valid" % num_sigma)
    num = len(sample)
    median = np.median(sample)
    sorted_sample = np.sort(sample)

    num_threshold1 = int(round((num-1)*0.841345))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    if num_sigma == 1:
        upper_sigma1 = sorted_sample[num_threshold1 - 1]
        lower_sigma1 = sorted_sample[num - num_threshold1 - 1]
        return median, [median-lower_sigma1, upper_sigma1-median]
    if num_sigma == 2:
        upper_sigma2 = sorted_sample[num_threshold2 - 1]
        lower_sigma2 = sorted_sample[num - num_threshold2 - 1]
        return median, [median-lower_sigma2, upper_sigma2-median]

def compute_confidence_intervals(sample, num_sigma, thresh=None):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if thresh is not None and num_sigma > 3:
        raise ValueError("Number of sigma-constraints restricted to three. %s not valid" % num_sigma)
    num = len(sample)
    median = np.median(sample)
    sorted_sample = np.sort(sample)

    if thresh is None:
        num_threshold1 = int(round((num-1)*0.841345))
        num_threshold2 = int(round((num-1)*0.977249868))
        num_threshold3 = int(round((num-1)*0.998650102))

        if num_sigma == 1:
            upper_sigma1 = sorted_sample[num_threshold1 - 1]
            lower_sigma1 = sorted_sample[num - num_threshold1 - 1]
            return median, [median-lower_sigma1, upper_sigma1-median]
        if num_sigma == 2:
            upper_sigma2 = sorted_sample[num_threshold2 - 1]
            lower_sigma2 = sorted_sample[num - num_threshold2 - 1]
            return median, [median-lower_sigma2, upper_sigma2-median]
    else:

        assert thresh <= 1
        thresh = (1 + thresh)/2
        num_threshold = int(round((num-1) * thresh))
        upper = sorted_sample[num_threshold - 1]
        lower = sorted_sample[num - num_threshold - 1]
        return median, [median - lower, upper - median]
