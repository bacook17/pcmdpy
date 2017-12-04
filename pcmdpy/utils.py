# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import warnings

# A module to create various utility functions

def make_pcmd(data):
    n_filters = data.shape[0]
    if (n_filters > 2):
        warnings.warn('make_pcmd not implemented for n_filters > 2. Discarding filters beyond first two.', RuntimeWarning)
    elif (n_filters < 2):
        raise IndexError("Must be at least 2 images to create a PCMD")
    colors = (data[0] - data[1]).flatten()
    mags = data[1].flatten()
    
    return np.array([colors, mags])

def make_hess(pcmd, bins, charlie_err=False, err_min=2.):
    n_dim = pcmd.shape[0]
    n = pcmd.shape[1] #total number of pixels
    if (n_dim != bins.shape[0]):
        raise IndexError("The first dimensions of pcmd and bins must match")
    counts = np.histogramdd(pcmd.T, bins=bins)[0].astype(float)
    #add "everything else" bin
    counts = counts.flatten()
    counts = np.append(counts, n - np.sum(counts))
    err = np.sqrt(counts)
    
    if charlie_err:
        #this is Charlie's method for inflating errors
        err[counts < 1.] = 0.1
        err[counts < 2.] *= 10.
    else:
        #inflate small errors, with inflation decreasing exponentially at higher counts
        err += err_min * np.exp(-err)

    #normalize by number of pixels
    hess = counts / n
    err /= n
    
    return counts, hess, err

def wrap_image(image, w_border):
    assert(image.ndim == 2)
    Nx, Ny = image.shape
    if (w_border >= Nx) or (w_border >= Ny):
        message = "wrap_image is not implemented for cases where border is wider than existing image"
        print(w_border)
        raise NotImplementedError(message)
    w_roll = w_border / 2
    im_temp = np.tile(image, [2,2])
    im_temp = np.roll(np.roll(im_temp, w_roll, axis=0), w_roll, axis=1)

    return im_temp[:Nx+w_border, :Ny+w_border]

def subdivide_image(image, d_sub, w_border=0):
    assert(image.ndim == 2)
    Nx, Ny = image.shape
    if (Nx != Ny):
        message = "image must be square"
        raise NotImplementedError(message)
    if (Nx % d_sub != 0):
        message = "subdivide_image is only implemented if image can be cleanly subdivided"
        raise NotImplementedError(message)
    Nx_sub, Ny_sub = Nx / d_sub , Ny / d_sub

    if w_border > 0:
        image = wrap_image(image, w_border)

    sub_im_matrix = np.zeros((d_sub, d_sub, Nx_sub + w_border, Ny_sub + w_border))
    for i in range(d_sub):
        for j in range(d_sub):
            x_slice = slice(Nx_sub*i, Nx_sub*(i+1) + w_border)
            y_slice = slice(Ny_sub*j, Ny_sub*(j+1) + w_border)
            sub_im_matrix[i,j] = image[x_slice, y_slice]
    return sub_im_matrix

