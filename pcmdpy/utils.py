# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import warnings
import sys


# A module to create various utility functions
def my_assert(bool_statement, fail_message=None):
    if bool_statement:
        return
    else:
        if fail_message is not None:
            print(fail_message)
        else:
            print("custom assertion failed")
        sys.exit(1)


#def make_pcmd(data):
#    n_filters = data.shape[0]
#    if (n_filters < 2):
#        raise IndexError("Must be at least 2 images to create a PCMD")
#    else:
#        mags = data[0].flatten()
#        results = [mags]
#        for i in range(1, n_filters):
#            colors = (data[i] - data[0]).flatten()
#            results.append(colors)
#        return np.array(results)

    
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
    my_assert(image.ndim == 2,
              "images must be 2-dimensional")
    Nx, Ny = image.shape
    if (w_border >= Nx) or (w_border >= Ny):
        message = "wrap_image is not implemented for cases where border is wider than existing image"
        print(w_border)
        raise NotImplementedError(message)
    w_roll = w_border // 2
    im_temp = np.tile(image, [2,2])
    im_temp = np.roll(np.roll(im_temp, w_roll, axis=0), w_roll, axis=1)

    return im_temp[:Nx+w_border, :Ny+w_border]

def subdivide_image(image, d_sub, w_border=0):
    my_assert(image.ndim == 2,
              "images must be 2-dimensional")
    Nx, Ny = image.shape
    if (Nx != Ny):
        message = "image must be square"
        raise NotImplementedError(message)
    if (Nx % d_sub != 0):
        message = "subdivide_image is only implemented if image can be cleanly subdivided"
        raise NotImplementedError(message)
    Nx_sub, Ny_sub = Nx // d_sub , Ny // d_sub

    if w_border > 0:
        image = wrap_image(image, w_border)

    sub_im_matrix = np.zeros((d_sub, d_sub, Nx_sub + w_border, Ny_sub + w_border))
    for i in range(d_sub):
        for j in range(d_sub):
            x_slice = slice(Nx_sub*i, Nx_sub*(i+1) + w_border)
            y_slice = slice(Ny_sub*j, Ny_sub*(j+1) + w_border)
            sub_im_matrix[i,j] = image[x_slice, y_slice]
    return sub_im_matrix

def subpixel_shift(image, dx, dy):
    assert(np.abs(dx)<= 1.)
    assert(np.abs(dy)<= 1.)
    #roll the image by -1, 0, +1 in x and y
    rolls = np.zeros((3, 3, image.shape[0], image.shape[1]))
    for i, x in enumerate([-1, 0, 1]):
        for j, y in enumerate([-1, 0, 1]):
            rolls[j,i] = np.roll(np.roll(image, x, axis=1), y, axis=0)
    #make the coefficients for each corresponding rolled image
    coeffs = np.ones((3, 3))
    if np.isclose(dx, 0.):
        coeffs[:, 0] = coeffs[:, 2] = 0.
    elif dx < 0.:
        coeffs[:, 2] = 0.
        coeffs[:, 0] *= -dx
        coeffs[:, 1] *= 1 + dx
    else:
        coeffs[:, 0] = 0.
        coeffs[:, 2] = dx
        coeffs[:, 1] *= 1 - dx
        
    if np.isclose(dy, 0.):
        coeffs[0, :] = coeffs[2, :] = 0.
    elif dy < 0.:
        coeffs[2, :] = 0.
        coeffs[0, :] *= -dy
        coeffs[1, :] *= 1 + dy
    else:
        coeffs[0, :] = 0.
        coeffs[2, :] *= dy
        coeffs[1, :] *= 1 - dy
    assert(np.isclose(np.sum(coeffs), 1.))
    result = np.zeros((image.shape[0], image.shape[0]))
    for i in range(3):
        for j in range(3):
            result += coeffs[i, j] * rolls[i, j]
    return result


def generate_image_dithers(image, shifts=[0., 0.25, 0.5, 0.75], norm=False):
    n = len(shifts)
    X = image.shape[0]
    Y = image.shape[1]
    tiles = np.zeros((n, n, X, Y))
    for i, dx in enumerate(shifts):
        for j, dy in enumerate(shifts):
            tiles[j, i] = subpixel_shift(image, dx, dy)
            if norm:
                tiles[j, i] /= np.sum(tiles[j, i])
    return tiles
