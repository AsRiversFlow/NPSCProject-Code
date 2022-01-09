import numpy as np

def import_image(image_name):
    """read in the image"""
    # importing PIL
    from PIL import Image
    import numpy as np

    # Read image
    img = Image.open(image_name)

    # convert to array
    img_arr = np.asarray(img, dtype=np.uint8)

    return img_arr


def get_samples(img_arr, num, centre_rad_ratio, neighbourhood_di):
    """ take a specified number of samples """
    neighbourhoods = np.empty((num, neighbourhood_di, neighbourhood_di), dtype=np.uint8)
    centres = np.empty((num, neighbourhood_di // centre_rad_ratio, neighbourhood_di // centre_rad_ratio), dtype=np.uint8)
    for ii in range(num):
        sample = (get_sample(img_arr, neighbourhood_di, centre_rad_ratio))
        neighbourhoods[ii] = (sample[0])
        centres[ii] = (sample[1])
    return (neighbourhoods, centres)


def get_sample(img_arr, size, centre_rad_ratio):
    """ take a random circle from the image """
    import random
    # generate dimensions
    start = (random.randint(0, img_arr.shape[0] - size),
             random.randint(0, img_arr.shape[1] - size))
    end = (start[0] + size, start[1] + size)

    # crop image outside
    sample = crop_circle(img_arr[start[0]:end[0], start[1]:end[1]].copy())
    # remove centre
    neighbourhood, centre = take_centre_circle(sample, centre_rad_ratio)

    return neighbourhood, centre


def crop_circle(img_arr):
    """crop to circle with diameter equal to square width"""
    radius = img_arr.shape[0] // 2
    centre = (radius, radius)
    for ii in range(img_arr.shape[0]):
        for jj in range(img_arr.shape[1]):
            if euc_distance(centre, (ii, jj)) > radius:
                img_arr[ii][jj] = 0
    return img_arr


def take_centre_circle(neighbourhood_circle_arr, centre_rad_ratio):
    """black out centre circle from neighbourhood, crop edge of centre circle"""
    # find the radius of the centre circle
    centre_rad = neighbourhood_circle_arr.shape[0] // centre_rad_ratio // 2
    # centre of the image as a tuple
    centre = (neighbourhood_circle_arr.shape[0] // 2, neighbourhood_circle_arr.shape[1] // 2)

    # duplicate the centre and crop to circle
    centre_circle_arr = neighbourhood_circle_arr[
        centre[0] - centre_rad: centre[0] + centre_rad,
        centre[1] - centre_rad: centre[1] + centre_rad
        ].copy()
    centre_circle_arr = crop_circle(centre_circle_arr)

    # iterate through each pixel in neighbourhood
    for ii in range(centre[0] - centre_rad, centre[0] + centre_rad):
        for jj in range(centre[1] - centre_rad, centre[1] + centre_rad):
            if euc_distance(centre, (ii, jj)) < centre_rad:
                neighbourhood_circle_arr[ii][jj] = 0

    # return both
    return neighbourhood_circle_arr, centre_circle_arr


def euc_distance(pointA, pointB):
    """Euclidean distance - must be tuples in 2D"""
    from math import sqrt
    return sqrt((pointA[1] - pointB[1])**2 + (pointA[0] - pointB[0])**2)



