import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature, io, filters

# Generate noisy image of a square
im = io.imread("tif with ice/rgb_converted_int16.png", as_gray=True)

# im = ndi.gaussian_filter(im, 4)
# im += 0.2 * np.random.random(im.shape)
# im = filters.gaussian(im, sigma=5)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=0.9)
edges2 = feature.canny(im, sigma=1)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()
