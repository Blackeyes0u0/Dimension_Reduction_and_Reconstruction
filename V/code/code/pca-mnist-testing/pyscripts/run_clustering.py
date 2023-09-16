import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tools import read_mnist, image2vec, vec2image


# parameters
n_components = 75
digits = [0, 1, 2, 3]

# read data
images, labels = read_mnist()
mask = np.array([l in digits for l in labels])
images = images[mask]
labels = labels[mask]

# run PCA
X = np.array([image2vec(im) for im in images])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X_rec = pca.inverse_transform(X_pca)
images_rec = [vec2image(x) for x in X_rec]


# colors
color_dict = {0: 'tab:blue',
              1: 'tab:orange',
              2: 'tab:red',
              3: 'tab:green',
              4: 'tab:purple',
              5: 'k',
              6: 'tab:olive',
              7: 'tab:brown',
              8: 'tab:cyan',
              9: 'tab:pink'}
colors = [color_dict[l] for l in labels]

# plot params
ms1 = 4
ms2 = 15
fs = 18

# setup
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# plot clustering
for ax, i in zip([ax1, ax2, ax3, ax4], [1, 2, 3, 4]):
    x = X_pca[:, 0]
    y = X_pca[:, i]
    ax.scatter(x, y, c=colors, s=ms1)
    ax.set_xlabel('Principal component 1', fontsize=fs)
    ax.set_ylabel('Principal component {}'.format(i), fontsize=fs)
    ax.tick_params(labelsize=fs)

# legend
for d in digits:
    ax1.plot(np.nan, np.nan, 'o', ms=ms2, label=str(d), color=color_dict[d])
ax1.legend(loc=1, fontsize=fs)


fig.tight_layout()
plt.show()
