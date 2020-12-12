"""Model black box functions to be optimised."""

import numpy as np
import pandas as pd
import altair as alt
import gpflow

DTYPE = float
DOMAIN_MIN = -1
DOMAIN_MAX = 1
DOMAIN_WIDTH = DOMAIN_MAX - DOMAIN_MIN
DOMAIN = DOMAIN_MIN, DOMAIN_MAX
COLOURSCHEME = 'redyellowgreen'

rng = np.random.default_rng()


def prior_draw(kernelf, xx):
    return np.random.multivariate_normal(np.zeros_like(xx), kernelf(xx.reshape((-1, 1))))


def xx(npoints):
    return np.linspace(DOMAIN_MIN, DOMAIN_MAX, npoints)


def random_submerged_skyline(npoints):
    """Return values of the skyline of a submerged house with a chimney at points."""
    # Choose random house and chimney locations
    chimney_width = DOMAIN_WIDTH / 20
    chimney_depth = rng.uniform(0, .5)
    chimney_height = .07
    house_height = .4
    margin = DOMAIN_WIDTH / 20
    min_house_width = DOMAIN_WIDTH / 4
    max_house_width = 3 / 4 * DOMAIN_WIDTH
    house_width = rng.uniform(min_house_width, max_house_width)
    house_left = rng.uniform(DOMAIN_MIN + margin, DOMAIN_MAX - margin - house_width)
    chimney_left = rng.uniform(house_left + margin, house_left + house_width - margin - chimney_width)
    # Draw random skyline, adjusting for house and chimney height
    x = xx(npoints)
    kernelf = gpflow.kernels.Matern52(1, lengthscales=5)
    skyline = prior_draw(kernelf, x)
    house = (x >= house_left) & (x <= house_left + house_width)
    chimney = (x >= chimney_left) & (x <= chimney_left + chimney_width)
    skyline[house] += house_height
    skyline[chimney] += chimney_height
    skyline -= skyline.max() + chimney_depth
    # Check the chimney is the highest point
    if skyline[chimney].max() <= skyline[~ chimney].max():
        # It isn't, resample
        return random_submerged_skyline(npoints)
    else:
        return x, skyline


class GPBlackBox:
    """Black box function to be optimised drawn from a Gaussian process."""

    def __init__(self, ndim=1):
        self.kernel = gpflow.kernels.Matern32(lengthscales=.4) + gpflow.kernels.Linear(variance=.4**2)
        self.noise_variance = .3**2
        # Give one data point at origin with value 0
        self.x = np.zeros((1, ndim), dtype=DTYPE)
        self.y = np.zeros((1, 1), dtype=DTYPE)
        self._update_model()

    def _update_model(self):
        self.model = gpflow.models.GPR(self.xy, kernel=self.kernel, noise_variance=self.noise_variance)

    def xgrid(self, num):
        if 1 == self.ndim:
            return np.linspace(DOMAIN_MIN, DOMAIN_MAX, num).reshape(num, 1)
        if 2 == self.ndim:
            xx = np.linspace(DOMAIN_MIN, DOMAIN_MAX, num).reshape(num, 1)
            xx0, xx1 = np.meshgrid(xx, xx)
            return np.asarray([np.ravel(xx0), np.ravel(xx1)]).T
        raise ValueError(f'Cannot create x-grid when x has dimensions {self.ndim} > 2')

    @property
    def ndim(self):
        return self.x.shape[-1]

    @property
    def xy(self):
        return (self.x, self.y)

    def data(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        return (pd.DataFrame(x)
                .rename(columns=dict((i, f'x{i}') for i in range(self.ndim)))
                .assign(y=y))

    def __call__(self, x):
        x = np.asarray(x).astype(DTYPE)
        if x.ndim < 2:
            x = x.reshape((-1, self.ndim))
        assert x.shape[-1] == self.ndim
        assert DOMAIN_MIN <= x.min()
        assert x.max() <= DOMAIN_MAX
        mean, var = self.model.predict_y(x)
        y = np.random.normal(loc=mean, scale=np.sqrt(var))
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self._update_model()
        return y

    def sample_f(self, num):
        xx = self.xgrid(num)
        f = self.model.predict_f_samples(xx).numpy()
        return self.data(xx, f).rename(columns={'y': 'f'})

    def plot_xy(self):
        if 1 == self.ndim:
            return self._plot_xy_1()
        if 2 == self.ndim:
            return self._plot_xy_2()
        raise ValueError(f'Cannot plot x-y when x has dimensions {self.ndim} > 2')

    def _plot_xy_1(self):
        return (
            alt.Chart(self.data())
            .mark_circle(size=60)
            .encode(x=alt.X('x0:Q', scale=alt.Scale(domain=DOMAIN)), y='y'))

    def _plot_xy_2(self):
        return (
            alt.Chart(self.data())
            .mark_circle(size=60, stroke='black', strokeWidth=1)
            .encode(x=alt.X('x0:Q', scale=alt.Scale(domain=DOMAIN)),
                    y=alt.X('x1:Q', scale=alt.Scale(domain=DOMAIN)),
                    color=alt.Color('y:Q', scale=alt.Scale(scheme=COLOURSCHEME, domainMid=0))))
