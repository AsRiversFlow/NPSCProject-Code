from input_image import import_image, get_samples
from output_image import output_array_as_image, output_image
from imageProcessing import train_model
from preprocessing import pre_process, remove_image_filler, restore_image_filler

import numpy as np

# must be true that neighbourhood_DI // centre_rad % 4
NUM_SAMPLES_MIN = 100
NUM_SAMPLES_MAX = 1100
NUM_SAMPLES_STEP = 200

NEIGHBOURHOOD_DI_MIN = 120
NEIGHBOURHOOD_DI_MAX = 240
NEIGHBOURHOOD_DI_STEP = 40

CENTRE_RAD_RATIO = 5


def main():
    # import image as numpy array
    img_arr = import_image("BSE_No_boundary_PS_1.tif")
    print("Original image height:", img_arr.shape[0],
        "and length:", img_arr.shape[1])
    count = 0
    for ii in range(NUM_SAMPLES_MIN, NUM_SAMPLES_MAX + 1, NUM_SAMPLES_STEP):
        for jj in range(NEIGHBOURHOOD_DI_MIN, NEIGHBOURHOOD_DI_MAX + 1, NEIGHBOURHOOD_DI_STEP):
            count += 1
            print("\nTest", count)
            try: #catch exceptions - the tests will likely eventually call a filesize too big
                run_test_set(img_arr, ii, jj, CENTRE_RAD_RATIO)
            except Exception as e:
                print("Failed on Num samples =", ii, "; Diameter =", jj)
                print("Error:", e)

    print("Successfully ran ", count, "tests.")


def run_test_set(img_arr, num_samples, neighbourhood_di, centre_rad_ratio):
    # select a single square of random size as an array
    (neighbourhoods, centres) = get_samples(img_arr, num_samples, centre_rad_ratio, neighbourhood_di)

    rmse_mean = 0.0
    for ii in range(3):
        rmse, name = run_test(neighbourhoods, centres, centre_rad_ratio)
        rmse_mean += rmse
    rmse_mean = rmse_mean / 3.0

    # MACHINE LEARNING
    print("Samples =", num_samples, "; Ratio =", centre_rad_ratio, "; Diameter =", neighbourhood_di,
        "; Algorithm =", name, "; Mean RMSE of 3 tests =", rmse_mean, "\n")


def run_test(neighbourhoods, centres, centre_rad_ratio):
    Y_rad = centres.shape[1] // 2
    X, Y = pre_process(neighbourhoods, centres, centre_rad_ratio)
        # X = list of neighbourhoods, Y = list of centres

    return train_model(X, Y, Y_rad, centre_rad_ratio)


def test_filler(X, Y):
    """ Flatten and remove filler on X and Y, convert back and
        check if the same as original"""
    output_array_as_image(Y, ["Y Original"])
    output_array_as_image(X, ["X Original"])
    X_rad = int(sqrt(X.shape[1])) // 2
    Y_rad = int(sqrt(Y.shape[1])) // 2

    X_new = remove_image_filler(X, 1, CENTRE_RAD_RATIO)
    Y_new = remove_image_filler(Y, 0, CENTRE_RAD_RATIO)
    X_new = restore_image_filler(X_rad, X_new, 1, CENTRE_RAD_RATIO)
    Y_new = restore_image_filler(Y_rad, Y_new, 0, CENTRE_RAD_RATIO)

    output_array_as_image(Y, ["Y Transformed"])
    output_array_as_image(X, ["X Transformed"])


main()
