from output_image import output_results
from preprocessing import restore_image_filler

import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import matplotlib.pyplot as plt
#import skimage.restoration I think this was inpainting


# MACHINE LEARNING
def train_model(X, Y, Y_rad, centre_rad_ratio):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)
    # rmse, name = LR(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad)
    rmse, name = KNN(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad)
    # rmse, name = SVR(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad)
    # rmse, name = RF(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad)

    return rmse, name


def LR(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad):
    """ Train and test linear regression algorithm on data """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_validation)

    error = sqrt(mean_squared_error(Y_validation, Y_pred)) #calculate rmse

    # output results to SciView plots
    #Y_validation = restore_image_filler(Y_rad, Y_validation, 0, centre_rad_ratio)
    #Y_pred = restore_image_filler(Y_rad, Y_pred, 0, centre_rad_ratio)
    #output_results(Y_validation, "LR Validation", -2, -1)
    #output_results(Y_pred, "LR Prediction", -2, -1)

    return error, "Linear Regression"


def KNN(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad):
    """ Train and test KNN regression on data """

    # preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_validation = pd.DataFrame(scaler.fit_transform(X_validation))
    #Y_train = pd.DataFrame(scaler.fit_transform(Y_train))
    #Y_validation = pd.DataFrame(scaler.fit_transform(Y_validation))

    K, K_error, Y_pred = find_K(X_train, Y_train, X_validation, Y_validation)

    # output results to SciView plots
    # Y_pred = np.array(Y_pred, dtype=np.uint8)
    #Y_validation = restore_image_filler(Y_rad, Y_validation, 0, centre_rad_ratio)
    #Y_pred = restore_image_filler(Y_rad, Y_pred, 0, centre_rad_ratio)
    #output_results(Y_validation, "KNN Validation", -4, -1)
    #output_results(Y_pred, "KNN Prediction", -4, -1)

    msg = "KNN"
    print("K =", K)

    return K_error, msg


def find_K(X_train, Y_train, X_validation, Y_validation):
    errors = np.empty(len(X_train))
    for ii in range(len(X_train)):
        # run predictions
        model = neighbors.KNeighborsRegressor(n_neighbors = (ii + 1))
        model.fit(X_train, Y_train)  #fit the model
        Y_pred = model.predict(X_validation) #make prediction on test set
        errors[ii] = sqrt(mean_squared_error(Y_validation, Y_pred)) #calculate rmse

    temp = errors[0]
    K = 1
    for ii in range(len(errors)):
        if(errors[ii] < temp):
            temp = errors[ii]
            K = ii + 1

    # run on optimised K value
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train, Y_train)  #fit the model
    Y_pred = model.predict(X_validation) #make prediction on test set

    return (K, temp, Y_pred)


def SVR(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad):
    """ Train and test support vector regression algorithm on data """

    #preprocessing
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_validation = pd.DataFrame(scaler.fit_transform(X_validation))
    Y_train = pd.DataFrame(scaler.fit_transform(Y_train))
    Y_validation = pd.DataFrame(scaler.fit_transform(Y_validation))

    model = svm.SVR(gamma='scale')
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # calculate and output error
    error = sqrt(mean_squared_error(Y_validation, Y_pred)) #calculate rmse

    # output results to SciView plots
    Y_validation = restore_image_filler(Y_rad, Y_validation, 0, centre_rad_ratio)
    Y_pred = restore_image_filler(Y_rad, Y_pred, 0, centre_rad_ratio)
    output_results(Y_validation, "SVR Validation", -3, -1)
    output_results(Y_pred, "SVR Prediction", -3, -1)

    return error, "Support Vector Regression"


def RF(centre_rad_ratio, X_train, X_validation, Y_train, Y_validation, Y_rad):
    """ Train and test random forests regression on data """
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_validation)
    error = sqrt(mean_squared_error(Y_validation, Y_pred)) #calculate rmse

    # output results to SciView plots
    #Y_validation = restore_image_filler(Y_rad, Y_validation, 0, centre_rad_ratio)
    #Y_pred = restore_image_filler(Y_rad, Y_pred, 0, centre_rad_ratio)
    #output_results(Y_validation, "RF Validation", -3, -1)
    #output_results(Y_pred, "RF Prediction", -3, -1)

    return error, "Random Forests"
