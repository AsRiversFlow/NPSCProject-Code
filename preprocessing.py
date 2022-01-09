from input_image import euc_distance
from output_image import make_square

import numpy as np

def pre_process(neighbourhoods, centres, centre_rad_ratio):
    """preprocessing - reshape data and
        return as neighbourhood array and centres array (X, Y)"""
    X = remove_image_filler(neighbourhoods, 1, centre_rad_ratio)
    Y = remove_image_filler(centres, 0, centre_rad_ratio)

    return X, Y


def remove_image_filler(img_arr, centre_flag, centre_rad_ratio):
    """ remove the zeros from either a square or flattened image
        returns flattened """
    img_arr = make_square(img_arr) #if not already square, makes square

    # single image
    if(len(img_arr.shape) == 2):
        new_img_arr = np.empty(img_arr.shape[0] * img_arr.shape[1], dtype=np.uint8)
        new_img_arr = remove_image_filler_single(img_arr, centre_flag, centre_rad_ratio)
    else:
        # array of images
        new_img_arr = np.empty((img_arr.shape[0], img_arr.shape[1] * img_arr.shape[2]), dtype=np.uint8)
        for ii in range(img_arr.shape[0]):
            new_img_arr[ii] = remove_image_filler_single(img_arr[ii], centre_flag, centre_rad_ratio)

    return new_img_arr


def remove_image_filler_single(img_arr, centre_flag, centre_rad_ratio):
    """takes a single image, removes outside and (if necessary) centre 0s
        returns flattened image"""
    shape = img_arr.shape
    radius = shape[0] // 2
    centre = (radius, radius)
    centre_rad = radius // centre_rad_ratio
    new_arr = np.empty(img_arr.shape[0] * img_arr.shape[1], dtype=np.uint8)
    count = 0
    # iterate through each element
    for ii in range(img_arr.shape[0]):
        for jj in range(img_arr.shape[1]):
            # if inside big circle
            if(euc_distance(centre, (ii, jj)) <= radius):
                # if outside small circle, as necessary
                if((centre_flag == 1 and
                    euc_distance(centre, (ii, jj)) >= centre_rad) or
                    (centre_flag != 1)):
                    # add to new array
                    new_arr[count] = img_arr[ii][jj]
                    count += 1

    return new_arr


def restore_image_filler(rad, img_arr, centre_flag, centre_rad_ratio):
    """ restores images to have filler 0s so they can be output as squares """
    shape = img_arr.shape
    if(len(shape) == 1):
        new_img_arr = restore_image_filler_single(rad, img_arr, centre_flag, centre_rad_ratio)
    else:
        new_img_arr = np.empty((shape[0], rad * 2, rad * 2), dtype=np.uint8)
        for ii in range(shape[0]):
            new_img_arr[ii] = restore_image_filler_single(rad, img_arr[ii], centre_flag, centre_rad_ratio)

    return new_img_arr


def restore_image_filler_single(radius, img_arr, centre_flag, centre_rad_ratio):
    """ creates a new (square) array from the flattened and reduced image """
    new_arr = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    count = 0
    centre = (radius, radius)
    centre_rad = radius // centre_rad_ratio
    for ii in range(radius * 2):
        for jj in range(radius * 2):
            # if inside big circle
            if(euc_distance(centre, (ii, jj)) <= radius):
                # if outside small circle, as necessary
                if((centre_flag == 1 and
                    euc_distance(centre, (ii, jj)) >= centre_rad) or
                    (centre_flag != 1)):
                    # add to new array
                    new_arr[ii][jj] = img_arr[count]
                    count += 1

    return new_arr


def standard_flatten(img_list):
    """Not in use"""
    # n_samples -> no of tuples, i.e. number of neigh/centre pairs
    n_samples = len(img_list)

    X_size = len(img_list[0][0])
    X_size = X_size * X_size
    Y_size = len(img_list[0][1])
    Y_size = Y_size * Y_size

    X = np.empty((n_samples, X_size), dtype=np.uint8)
    Y = np.empty((n_samples, Y_size), dtype=np.uint8)
    # reshape each element individually, and add reshaped tuples to new list
    for ii in range(n_samples):
        X[ii] = np.reshape(img_list[ii][0], -1)
        Y[ii] = np.reshape(img_list[ii][1], -1)

    return X, Y

