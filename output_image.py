import numpy as np

PREDICTION_DIFFERENCE = 15
COLOUR_BLUE = 3 # this is wrong
COLOUR_RED = 4 # this is also wrong


def output_array_as_image(img_array, image_names = ["Sample"]):
    """converts numpy array to images and spits out the images
        Can handle array of multiple images"""

    shape = img_array.shape
    if(len(shape) == 1): # not square, one image
        output_image(make_square(img_array), image_names[0])
    elif(len(shape) == 2):
        if(shape[0] == shape[1]): # already square, one image
            output_image(img_array, image_names[0])
        else: # not square, >=1 image
            squares = make_square(img_array)
            for ii in range(squares.shape[0]):
                output_image(squares[ii], image_names[ii])
    elif(len(shape) == 3): #already square, more than one image
        for ii in range(shape[0]):
            output_image(img_array[ii], image_names[ii])
    else:
        print("Can't output images. Shape of img_array: ", img_array.shape)


def output_image(img_array, image_name = "Sample"):
    """ just spits the image out  - uses matplotlib plot """
    #from PIL import Image
    #img = Image.fromarray(img_array, mode="L")
    #img.show()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.imshow(img_array, cmap='gray', origin='lower', vmin=0.0, vmax=255.0)
    plt.title(image_name)
    plt.colorbar()
    plt.show()


def compare(orig_arrs, predict_arrs):
    """ take an array of predicted centres and compare with original centres """
    if(len(orig_arrs[0][1]) == 1):
        make_square(orig_arrs)
    if(len(predict_arrs[0][1]) == 1):
        make_square(predict_arrs)

    compare_arr = nd.empty((len(orig_arrs), len(orig_arrs[0]), len(orig_arrs[0][0])))
    for ii in range(len(orig_arrs)):
        compare_result(compare_arrs[ii], orig_arrs[ii], predict_arrs[ii])


def output_results(arr, name, start, end):
    """ creates array of strings names and calls
        output_array_as_image on the given splice"""
    # create string array of plot names
    names = np.empty(len(arr), dtype=object)
    for ii in range(len(arr)):
        names[ii] = name + " no. " + str(ii + 1)

    # output
    output_array_as_image(arr[start:end], names[start:end])


def compare_result(arr, orig_arr, predict_arr):
    """ map difference between original and prediction """
    for ii in range(orig_arr.shape[0]):
        for jj in range(len(orig_arr.shape[1])):
            if(predict_arr[ii][jj] - orig_arr[ii][jj] >= PREDICTION_DIFFERENCE):
                arr[ii][jj] = COLOUR_RED
            elif(predict_arr[ii][jj] - orig_arr[ii][jj] <= -PREDICTION_DIFFERENCE):
                arr[ii][jj] = COLOUR_BLUE
            else:
                arr[ii][jj] = predict_arr[ii][jj]

    return arr


def make_square(arr):
    """ convert a 2D array of flat images to a 3D array of square images """
    from math import sqrt
    shape = arr.shape
    if not (len(shape) >= 2 and shape[-1] == shape[-2]):
        if(len(shape) == 1):
            length = int(sqrt(shape[0]))
            result = np.reshape(arr, (length, length))
        elif(len(shape) == 2):
            length = int(sqrt(shape[1]))
            result = np.reshape(arr, (shape[0], length, length))
    else:
        result = arr

    return result
