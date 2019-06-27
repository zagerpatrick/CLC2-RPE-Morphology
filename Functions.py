# Import packages and modules
import random
import re
import glob
import collections
import math
import numpy as np
import pandas as pd
import scipy as sp
import skimage
import mahotas as mh
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma

from tqdm import tqdm
from numpy import (mean, float_, dot, interp, uint8, uint16,
                   uint64, log10, any as np_any, all as np_all)
from skimage import io
from skimage import filters
from skimage.filters import gaussian
from skimage import img_as_ubyte
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from matplotlib import colors as c
import matplotlib.gridspec as gridspec

def imread_rgb(f):
    '''
    Function used to read in rgb images properly through
    skimage ImageCollection.
    '''
    return skimage.io.imread(f, as_gray=True)

def factor_maker(factortype, filenamelist, steps):
    '''
    Extract relevant factors from filenames and convert them into
    Pandas Series'.

    Parameters
    factortype : list
        List of the factor of interest.
    filenamelist : list
        List of ERG filenames
    steps: int
        Number of stimulus steps in the ERG experiment.
    '''

    output = []

    for f in range(len(filenamelist)):
        for s in range(len(factortype)):
            if factortype[s] in filenamelist[f]:
                output.extend([factortype[s] for i in range(steps)])
            else:
                pass

    return pd.Series(output)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def hessian(image, return_hessian=True, return_eigenvalues=True, return_eigenvectors=True):
    '''Calculate hessian, its eigenvalues and eigenvectors
    image - n x m image. Smooth the image with a Gaussian to get derivatives
            at different scales.
    return_hessian - true to return an n x m x 2 x 2 matrix of the hessian
                     at each pixel
    return_eigenvalues - true to return an n x m x 2 matrix of the eigenvalues
                         of the hessian at each pixel
    return_eigenvectors - true to return an n x m x 2 x 2 matrix of the
                          eigenvectors of the hessian at each pixel
    The values of the border pixels for the image are not calculated and
    are zero
    '''
    #The Hessian, d(f(x0, x1))/dxi/dxj for i,j = [0,1] is approximated by the
    #following kernels:

    #d00: [[1], [-2], [1]]
    #d11: [[1, -2, 1]]
    #d01 and d10: [[   1, 0,-1],
                  #[   0, 0, 0],
                  #[  -1, 0, 1]] / 2


    #The eigenvalues of the hessian:
    #[[d00, d01]
     #[d01, d11]]
     #L1 = (d00 + d11) / 2 + ((d00 + d11)**2 / 4 - (d00 * d11 - d01**2)) ** .5
     #L2 = (d00 + d11) / 2 - ((d00 + d11)**2 / 4 - (d00 * d11 - d01**2)) ** .5

     #The eigenvectors of the hessian:
     #if d01 != 0:
       #[(L1 - d11, d01), (L2 - d11, d01)]
    #else:
       #[ (1, 0), (0, 1) ]


    #Ideas and code borrowed from:
    #http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    #http://www.longair.net/edinburgh/imagej/tubeness/


    hessian = np.zeros((image.shape[0], image.shape[1], 2, 2))
    hessian[1:-1, :, 0, 0] = image[:-2, :] - (2 * image[1:-1, :]) + image[2:, :]
    hessian[1:-1, 1:-1, 0, 1] = hessian[1:-1, 1:-1, 0, 1] = (
        image[2:, 2:] + image[:-2, :-2] -
        image[2:, :-2] - image[:-2, 2:]) / 4
    hessian[:, 1:-1, 1, 1] = image[:, :-2] - (2 * image[:, 1:-1]) + image[:, 2:]
    #
    # Solve the eigenvalue equation:
    # H x = L x
    #
    # Much of this from Eigensystem2x2Float.java from tubeness
    #
    A = hessian[:, :, 0, 0]
    B = hessian[:, :, 0, 1]
    C = hessian[:, :, 1, 1]

    b = -(A + C)
    c = A * C - B * B
    discriminant = b * b - 4 * c

    # pn is something that broadcasts over all points and either adds or
    # subtracts the +/- part of the eigenvalues

    pn = np.array([1, -1])[np.newaxis, np.newaxis, :]
    L = (- b[:, :, np.newaxis] +
         (np.sqrt(discriminant)[:, :, np.newaxis] * pn)) / 2
    #
    # Report eigenvalue # 0 as the one with the highest absolute magnitude
    #
    L[np.abs(L[:, :, 1]) > np.abs(L[:, :, 0]), :] =\
      L[np.abs(L[:, :, 1]) > np.abs(L[:, :, 0]), ::-1]


    if return_eigenvectors:
        #
        # Calculate for d01 != 0
        #
        v = np.ones((image.shape[0], image.shape[1], 2, 2)) * np.nan
        v[:, :, :, 0] =  L - hessian[:, :, 1, 1, np.newaxis]
        v[:, :, :, 1] = hessian[:, :, 0, 1, np.newaxis]
        #
        # Calculate for d01 = 0
        default = np.array([[1, 0], [0, 1]])[np.newaxis, :, :]
        v[hessian[:, :, 0, 1] == 0] = default
        #
        # Normalize the vectors
        #
        d = np.sqrt(np.sum(v * v, 3))
        v /= d[:, :, :, np.newaxis]

    result = []
    if return_hessian:
        result.append(hessian)
    if return_eigenvalues:
        result.append(L)
    if return_eigenvectors:
        result.append(v)
    if len(result) == 0:
        return
    elif len(result) == 1:
        return result[0]
    return tuple(result)

def enhance_neurites(image, sigma):

    smoothed = skimage.filters.gaussian(image, sigma)

    hess = hessian(smoothed, return_hessian=False, return_eigenvectors=False)

    # The positive values are darker pixels with lighter
    # neighbors. The original ImageJ code scales the result
    # by sigma squared - I have a feeling this might be
    # a first-order correction for e**(-2*sigma), possibly
    # because the hessian is taken from one pixel away
    # and the gradient is less as sigma gets larger.
    result = -hess[:, :, 0] * (hess[:, :, 0] < 0) * (sigma ** 2)

    return result

# Setup colormap for segmented images

colors = list(map(plt.cm.nipy_spectral, range(0, 256, 1)))
random.shuffle(colors)
colors[0] = (0.,0.,0.,1.)
rmap = c.ListedColormap(colors)
