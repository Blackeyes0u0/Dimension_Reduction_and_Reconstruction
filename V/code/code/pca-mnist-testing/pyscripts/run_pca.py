import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tools import MnistPlotter, read_mnist, image2vec, vec2image

# parameters
n_components = 75
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# read data
images, labels = read_mnist()
mask = np.array([l in digits for l in labels])
images = images[mask]
labels = labels[mask]


# setup
plotter = MnistPlotter()


# run PCA
X = np.array([image2vec(im) for im in images])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X_rec = pca.inverse_transform(X_pca)
images_rec = [vec2image(x) for x in X_rec]

# print result
explained_variance = pca.explained_variance_ratio_.sum()
print('Using {} components, explained variance is {}'.format(n_components, explained_variance))


# showcase reconstruction
i = 20
plotter.draw_two_images(images[i], images_rec[i])

# plot principal images
n = 12
fig = plt.figure(figsize=(18, 12), dpi=96)
for i in range(n):
    ax = fig.add_subplot(3, 4, i+1)
    image = 26 * vec2image(pca.components_[i])
    plotter._draw_single_image(ax, image, label=None)

plt.show()
