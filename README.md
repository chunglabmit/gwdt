# GWDT: Grey-weighted distance transform

This package implements the grey-weighted
distance transform in N dimensions. The idea and algorithm
are taken from
*Xiao H, Peng H. "APP2: automatic tracing of 3D neuron
 morphology based on hierarchical pruning of a
  gray-weighted image distance-tree".
Bioinformatics. 2013;29(11):1448–1454.
 doi:10.1093/bioinformatics/btt170*
 
 Usage:
 ```python

from gwdt import gwdt
import numpy as np
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_otsu
import tifffile

img = tifffile.imread("myimg.tiff").astype(np.float32)
#
# Normalize the image intensities to be nice
#
norm_img = (img - img.min()) / (img.max() - img.min())
#
# the background is defined as voxels less than zero,
# so calculate a threshold and subtract to get an
# image that has the background intensity set
# appropriately
#
thresh = threshold_otsu(norm_img)
fgnd_img = norm_img - thresh
fgnd_img[fgnd_img < 0] = 0
structure = generate_binary_structure(img.ndim, 1)
distance_transform = gwdt(fgnd_img, structure)
#
# You'll get a nice output that will be great for 3D
# skeletonization.
#

```