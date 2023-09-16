import numpy as np
import matplotlib.pyplot as plt


def image2vec(image):
    return image.flatten()


def vec2image(vec):
    return vec.reshape(28, 28)


def read_mnist(normalize=True):
    labels = np.load('dataset/train-labels.npy').astype(int)
    images = np.load('dataset/train-images.npy').astype(float)
    if normalize:
        images = images / 255  # rescale to be between 0 and 1
    return images, labels


class MnistPlotter:
    """ Helper class for visualizing digits easily """

    # plot params to make figures nice
    size = 28
    cmap = 'Greys'
    dpi = 96
    lw = lw = dpi / (1024*32)

    def draw_image(self, image, label=None):
        # params
        figsize = (6, 6)

        # plot
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        ax1 = fig.add_subplot(111)
        self._draw_single_image(ax1, image, label)

        return fig

    def draw_two_images(self, im1, im2):
        # params
        figsize = (12, 6)

        # plot
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax, image in zip([ax1, ax2], [im1, im2]):
            self._draw_single_image(ax, image)

        return fig

    def draw_three_images(self, im1, im2, im3, **kwargs):

        # params
        figsize = (18, 6)

        # plot
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        for ax, image in zip([ax1, ax2, ax3], [im1, im2, im3]):
            self._draw_single_image(ax, image, **kwargs)

        return fig

    def _draw_single_image(self, ax, image, label=None, fs=50, **kwargs):

        # default kwargs
        default_kwargs = dict(vmin=0, vmax=1, edgecolor='k', lw=self.lw)
        default_kwargs.update(kwargs)

        # params
        lw_border = 6
        lim = [0, self.size]

        # plot image
        cmap = plt.get_cmap(self.cmap)
        ax.pcolormesh(image, cmap=cmap, **default_kwargs)

        # draw boundaries
        for v in [0, self.size]:
            ax.plot([v, v], lim, lw=lw_border, color='k')
            ax.plot(lim, [v, v], lw=lw_border, color='k')

        # handle axis
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_axis_off()

        # draw label
        if label is not None:
            ax.text(0.03, 0.84, str(label), fontsize=fs, color='k', transform=ax.transAxes)
