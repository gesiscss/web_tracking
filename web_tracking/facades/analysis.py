from math import ceil, floor

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from IPython.display import display
import statsmodels.api as sm

from matplotlib import cm
from matplotlib.colors import rgb2hex

from ..measures.entropy import Entropy
from ..measures.entropy_rate import EntropyRate
from ..measures.predictability import Predictability
from ..utils.utils import Utils

class Analysis():

    ENTROPY_OPTIONS = ('shannon', 'random')
    ENTROPY_RATE_OPTIONS = ('lz')
    PREDICTABILITY_OPTIONS = ('pi_unc', 'pi_rand', 'pi_max')
    STATS_COLS = ['domain_count', 'domain_count_coverage',
                  'domain_nunique', 'domain_unique_coverage',
                  'active_seconds_sum', 'active_seconds_coverage', 'active_seconds_mean', 'active_seconds_median',
                  'gap_seconds_mean']
    STATS_FORMATTERS = {'domain_count': '{:,}', 'domain_count_coverage': '{:.2%}',
                        'domain_nunique': '{:,}', 'domain_unique_coverage': '{:.2%}',
                        'active_seconds_sum': '{:,.0f}', 'active_seconds_coverage': '{:.2%}', 'active_seconds_mean': '{:,.0f}',
                        'gap_seconds_mean': '{:,.0f}'}


    def __init__(self):
        super().__init__()
        self.Entropy = Entropy()
        self.EntropyRate = EntropyRate()
        self.Predictability = Predictability()
        # styles
        ## line width frame
        mpl.rcParams['figure.figsize']=(3.3, 2)
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['patch.linewidth'] = 2
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['grid.linestyle'] = ':'
        mpl.rcParams['grid.color'] = 'black'
        # mpl.rcParams['text.fontsize'] = 10
        mpl.rcParams['legend.fontsize'] = 8
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        # xticks
        mpl.rcParams['xtick.major.size'] = 4
        mpl.rcParams['xtick.minor.size'] = 2
        mpl.rcParams['ytick.major.size'] = 4
        mpl.rcParams['ytick.minor.size'] = 2
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.labelpad'] = 4
        mpl.rcParams['backend'] = 'ps'
        mpl.rcParams['xtick.major.pad'] = 4
        mpl.rcParams['ytick.major.pad'] = 4
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
        params = {'mathtext.default': 'sf'}
        plt.rc('figure', facecolor='w')
        plt.rcParams.update(params)


    def compute_entropy(self, df, features, workers=1, options=[*ENTROPY_OPTIONS, *ENTROPY_RATE_OPTIONS]):
        self.df_entropy = pd.DataFrame()

        if any([opt in Analysis.ENTROPY_OPTIONS for opt in options]):
            opts = [opt for opt in options if opt in Analysis.ENTROPY_OPTIONS]
            self.Entropy.calculate(df, features=features, options=opts, workers=workers)
            self.df_entropy = self.Entropy.df.copy()

        if any([opt in Analysis.ENTROPY_RATE_OPTIONS for opt in options]):
            self.EntropyRate.calculate(df, features=features, workers=workers)
            self.df_entropy = self.df_entropy.merge(self.EntropyRate.df, on='user')

        return self.df_entropy


    def compute_predictability(self, df_entropy, features, options=PREDICTABILITY_OPTIONS):
        self.df_predictability = pd.DataFrame()
        self.Predictability.calculate(df_entropy, features=features)
        self.df_predictability = self.Predictability.df
        return self.df_predictability


    def compute_regression_model(self, df, output, variables):
        TEMP = df[[output, *variables]].dropna()
        X = TEMP[variables]
        X = sm.add_constant(X)
        y = TEMP[output]
        MODEL = sm.OLS(y, X).fit()
        COEFS = MODEL \
            .conf_int().rename(columns={0: 'coef_min', 1: 'coef_max'}).drop(index=['const']) \
            .merge(MODEL.params.to_frame().rename(columns={0: 'coef'}), left_index=True, right_index=True) \
            .merge(MODEL.pvalues.to_frame().rename(columns={0: 'p_value'}), left_index=True, right_index=True) \
            .reset_index().rename(columns={'index': 'attribute'})
        return MODEL.rsquared_adj, COEFS


    def plot_coefficient_intervals(self, coefs, threshold, ylabel='Attributes', **kwargs):
        TEMP = coefs.assign(n=lambda df: range(1, len(df)+1))
        COLORS = ['tomato' if p < threshold else 'steelblue' for p in TEMP.p_value]
        ax = TEMP.plot.scatter(x='coef', y='n', color=COLORS, **kwargs)
        ax.axvline(0, color='grey', linestyle='--', alpha=.5)
        ax.set_yticks(TEMP.n)
        ax.set_yticklabels(TEMP.attribute.tolist())
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Coefficient')
        ax.grid(color='lightgrey', linestyle='--', linewidth=.75, alpha=.3)

        for n, c in zip(TEMP.n.tolist(), COLORS):
            ax.fill_betweenx(n,  TEMP.query(f"n == {n}").coef_min, TEMP.query(
                f"n == {n}").coef_max, color=c, alpha=.75)

        return ax


    def display_descriptive_stats(self, df, columns):
        attributes_categorical = []
        attributes_numeric = []
        attributes_other = []

        for col in columns:
            if pd.api.types.is_object_dtype(df[col]):
                attributes_categorical.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                attributes_numeric.append(col)
            else:
                attributes_other.append(col)

        desc_categorical, dist_categorical = self.__get_categorical_descriptive_stats__(df, attributes_categorical)
        desc_numeric = self.__get_numeric_descriptive_stats__(df, attributes_numeric)

        if len(attributes_categorical) > 0:
            display(desc_categorical.style \
                    .background_gradient('Blues', subset=['count']) \
                    .background_gradient('Oranges', subset=['missing']) \
                    .background_gradient('Purples', subset=['unique']))

        for attr in dist_categorical:
            display(
                dist_categorical[attr].nlargest(25, 'Count').style \
                    .background_gradient('Blues', subset=['Count']) \
                    .bar(subset=['Share'], color='dodgerblue') \
                    .format({'Share': '{:.4f}%'}) \
                    .apply(lambda row: ['color: firebrick; font-weight:bold;'] * len(row) if pd.isnull(row[attr.title()]) else [''] * len(row) ,axis=1)
            )

        if len(attributes_numeric) > 0:
            display(desc_numeric.style \
                    .background_gradient('Blues', subset=['count']) \
                    .background_gradient('Oranges', subset=['missing']) \
                    .background_gradient('Purples', subset=['unique']) \
                    .background_gradient('GnBu', subset=['mean', 'std']) \
                    .background_gradient('PuBu', subset=['min', '25%', '50%', '75%', 'max']))

        return desc_categorical, dist_categorical, desc_numeric


    def display_descriptive_stats_per_attribute(self, df, df_meta, merge_on, attribute):
        total_domain_count = df.domain.count()
        num_unq_domains = df.domain.nunique()
        total_active_seconds = df.active_seconds.sum()

        stats = df.merge(df_meta, on=merge_on) \
                    .groupby(attribute) \
                    .agg({
                        'domain': ['count', 'nunique'],
                        'active_seconds': ['sum', 'mean', 'median'],
                        'gap_seconds': ['mean']
                    })

        stats.columns = [ '_'.join(c) for c in stats.columns ]

        stats = stats.assign(
                    domain_count_coverage   = lambda df: df['domain_count'] / total_domain_count,
                    domain_unique_coverage  = lambda df: df['domain_nunique'] / num_unq_domains,
                    active_seconds_coverage = lambda df: df['active_seconds_sum'] / total_active_seconds
                )


        display(
            stats[Analysis.STATS_COLS].style \
                .format(Analysis.STATS_FORMATTERS) \
                .background_gradient('Greens', subset=['domain_count_coverage']) \
                .background_gradient('GnBu', subset=['domain_unique_coverage']) \
                .background_gradient('Oranges', subset=['active_seconds_coverage']) \
                .background_gradient('Purples', subset=['active_seconds_mean']) \
                .background_gradient('BuPu', subset=['active_seconds_median']) \
                .background_gradient('PuRd', subset=['gap_seconds_mean'])
        )

        return stats


    def display_distribution_plot(self, df, df_meta, merge_on, measure, subject, attribute=None, **kwargs):
        # measure: entropy, predictability
        # subject: domain, category
        X_LABEL = 'Predictability' if measure.lower() == 'pi' else measure.title()
        Y_LABEL = 'Density'
        ATTR    = 'Individual' if attribute is None else attribute.title()
        measure = 'pi' if measure.lower() == 'predictability' else measure
        columns = sorted(self.__get_columns__(df.columns, measure, subject))
        # limits  = { 'x': (floor(df[columns].min().min()), ceil(df[columns].max().max())) }
        legends = [self.__rename_legend__(c) for c in columns]
        # colors  = []
        # font_title = {'weight': 'bold', 'size': 18}
        # font_label = {'weight': 'normal', 'size': 16}
        # plot_args = dict(grid=True, xlim=limits['x'], figsize=(12, 8), fontsize=12, linewidth=2)
        plot_args = dict(grid=True)

        if attribute is None:
            colors = self.__get_colors__(1)
            ax = df[columns].plot.kde(**plot_args, color=colors[0], linestyle=self.__get_linestyle__(0))
            ax.legend(legends)

        else:
            attrs = df_meta[attribute].dropna().unique()
            legends_w_attr = []
            colors = self.__get_colors__(len(attrs))

            for ind, attr in enumerate(attrs):
                legends_w_attr += [ f"{l}: {attr}" for l in legends]
                if ind == 0:
                    ax = df.merge(df_meta, on=merge_on).query(f"{attribute} == '{attr}'")[columns].plot.kde(**plot_args, color=colors[ind], linestyle=self.__get_linestyle__(ind))
                else:
                    df.merge(df_meta, on=merge_on).query(f"{attribute} == '{attr}'")[columns].plot.kde(**plot_args, color=colors[ind], linestyle=self.__get_linestyle__(ind), ax=ax)

            ax.legend(legends_w_attr)

        self.__fill_under_line__(ax, Utils.flatten(colors))

        # ax.set_title(f"{X_LABEL} of {subject.title()} by {ATTR}", fontdict=font_title)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        plt.show(ax)


    def __get_categorical_descriptive_stats__(self, df, attributes):
        descriptive = pd.DataFrame()
        columns = ['data_type', 'count', 'missing', 'unique']
        distribution = {}

        for attr in attributes:
            descriptive = descriptive.append(
                pd.DataFrame([('categorical',
                                df[attr].count(),
                                len(df[attr]) - df[attr].count(),
                                df[attr].nunique())],
                            index=[attr],
                            columns=columns)
            )
            distribution[attr] = df[attr].value_counts(dropna=False).to_frame() \
                                        .assign(Share=lambda df: 100 * df[attr] / df[attr].sum()) \
                                        .reset_index().sort_values('index') \
                                        .rename(columns={attr: 'Count', 'index': attr.title()})

        return descriptive, distribution


    def __get_numeric_descriptive_stats__(self, df, attributes):
        descriptive = pd.DataFrame()

        for attr in attributes:
            temp = df[attr].describe().to_frame().T \
                            .assign(
                                data_type = 'numeric',
                                missing   = len(df[attr])-df[attr].count(),
                                unique    = df[attr].nunique()
                            )[['data_type', 'count', 'missing', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            descriptive = descriptive.append(temp)

        return descriptive


    def __get_columns__(self, columns, measure, subject):
        return list(
            filter(lambda c: measure.lower() in c and subject.lower() in c, columns)
        )


    def __rename_legend__(self, text):
        if 'entropy' in text:
            if 'random' in text:
                return '$S^{rand}$'
            elif 'shannon' in text:
                return '$S^{unc}$'
            elif 'lz' in text or 'lempel_ziv' in text:
                return '$S$'
            else:
                return None
        elif 'pi' in text:
            if 'rand' in text:
                return '$\Pi_{rand}$'
            elif 'unc' in text:
                return '$\Pi_{unc}$'
            elif 'max' in text:
                return '$\Pi_{max}$'
            else:
                return None
        else:
            raise ValueError('Undefined legend labels.')


    def __get_colors__(self, n):
        oranges = cm.get_cmap('Oranges')
        blues = cm.get_cmap('Blues')
        greens = cm.get_cmap('Greens')
        return list(map(list, zip(
            [ rgb2hex(oranges(i/n)) for i in range(1, n+1) ],
            [ rgb2hex(blues(i/n)) for i in range(1, n+1) ],
            [ rgb2hex(greens(i/n)) for i in range(1, n+1) ],
        )))


    def __fill_under_line__(self, ax, colors):
        for i in range(0, len(colors)):
            l = ax.lines[i]
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x, y, color=colors[i], alpha=0.1)


    def __get_linestyle__(self, index):
        linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
        return linestyles[index % len(linestyles)]