#########################################################################################################
#  Description: Functions for running regression models
#
#########################################################################################################

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestRegressor as RandomForestReg
# from sklearn.linear_model import LinearRegression as LinearReg

# sklearn Toolkit
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import logging

from lib.visualization import modeling_tools
from lib.utils import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

# TODO: Figure out a centralized way to install/handle logging. Does it really need to be instantiated per file?
log = logging.getLogger(__name__)

#########################################################################################################
# Wrapper function to take input features and output from files with specific problems to solve and
# call run_models for CV and / or test


def run_models_wrapper(x, y, run_cv_flag=False, num_model_iterations=1, plot_learning_curve=False, test_size=0.2,
                       clf_class=RandomForestReg, **kwargs):

    # Create train / test split only for test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if run_cv_flag:
        # Run model on cross-validation dataproc and predict test dataproc on trained model
        log.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning KFold Cross-Validation / Test" + sf.Color.END)
        y_pred_kfold = run_model_regression(run_test_only=0, x_train=x_train, y_train=y_train, x_test=x_test,
                                            y_test=y_test, num_model_iterations=num_model_iterations,
                                            plot_learning_curve=plot_learning_curve, clf_class=clf_class, **kwargs)
        log.debug('KFold Output Dimensions: %s', y_pred_kfold.shape)

    # Run test - Train model and predict on test dataproc
    log.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning Only Test" + sf.Color.END)
    y_pred_test = run_model_regression(run_test_only=1, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                       num_model_iterations=num_model_iterations, clf_class=clf_class, **kwargs)
    log.debug('Test Output Dimensions: %s', y_pred_test.shape)

    if run_cv_flag:
        y_pred_df = pd.DataFrame(y_pred_kfold, columns=['Total points predicted'])
    else:
        y_pred_df = pd.DataFrame(y_pred_test, columns=['Total points predicted'])

    return y_pred_df


def run_model_regression(run_test_only, x_train, y_train, x_test, y_test, num_model_iterations=1,
                         plot_learning_curve=False, clf_class=RandomForestReg, **kwargs):
    # # @brief: For cross-validation, Runs the model and gives rmse / mse. Also, will run the trained model
    # #         on test dataproc if run_test_only is set
    # #         For test, trains the model on train dataproc and predicts rmse / mse for test dataproc
    # # @param: x_train, x_test - Input features (numpy array)
    # #         y_train, y_test - expected output (numpy array)
    # #         plot_learning_curve (only for cv) - bool
    # #         num_model_iterations - Times to run the model (to average the results)
    # #         clf_class - Model to run (if specified model doesn't run,
    # #                     then it'll have to be imported from sklearn)
    # #         **kwargs  - Model inputs, refer sklearn documentation for your model to see available parameters
    # # @return: None

    # Plot learning curve only for cv
    if not run_test_only and plot_learning_curve:
        title = "Learning Curves for regression"
        # Train dataproc further split into train and CV
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% dataproc randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(x_train.shape[0], n_iter=100, test_size=0.2, random_state=0)

        modeling_tools.plot_learning_curve(clf_class(**kwargs), title, x_train, y_train, cv=cv, n_jobs=-1)

        if not os.path.isdir("temp_pyplot_regr_dont_commit"):
            # Create dir if it doesn't exist. Do not commit this directory or contents.
            # It's a temp store for pyplot images
            os.mkdir("temp_pyplot_regr_dont_commit")

        # plt.show()
        plt.savefig("temp_pyplot_regr_dont_commit/learning_curve.png")

    # Error metrics - mean-squared error and root mse
    rmse_cv = rmse_test = 0.0
    mse_cv = mse_test = 0.0

    for _ in range(num_model_iterations):
        if run_test_only:  # test
            y_pred_test = run_test(x_train, y_train, x_test, clf_class, **kwargs)
            # calculate root mean squared error
            # Pep8 warning not valid
            rmse_test += ((np.mean((y_pred_test - y_test) ** 2)) ** 0.5)
            mse_test += np.mean((y_pred_test - y_test) ** 2)

            # Print first 10 actual and predicted values for test
            log.debug(y_test[0:10])
            log.debug(sf.format_float_0_2f(y_pred_test[0:10]))

            log.debug(np.mean(y_test))
            log.debug(np.mean(y_pred_test))

        else:  # cv
            y_pred_cv, y_pred_test = run_kfold_cv(x_train, y_train, x_test, clf_class, **kwargs)
            # Pep8 warning not valid
            rmse_cv += ((np.mean((y_pred_cv - y_train) ** 2)) ** 0.5)
            mse_cv += np.mean((y_pred_cv - y_train) ** 2)

            # Pep8 warning not valid
            rmse_test += ((np.mean((y_pred_test - y_test) ** 2)) ** 0.5)
            mse_test += np.mean((y_pred_test - y_test) ** 2)

            # Print first 10 actual and predicted values for cv
            log.debug(y_train[0:10])
            log.debug(sf.format_float_0_2f(y_pred_cv[0:10]))

            log.debug(np.mean(y_train))
            log.debug(np.mean(y_pred_cv))

            # Print first 10 actual and predicted values for test
            log.debug(y_test[0:10])
            log.debug(sf.format_float_0_2f(y_pred_test[0:10]))

            log.debug(np.mean(y_test))
            log.debug(np.mean(y_pred_test))

    if not run_test_only:
        rmse_cv /= num_model_iterations
        mse_cv /= num_model_iterations

        log.info(sf.Color.BOLD + sf.Color.DARKCYAN +
                 "\nCV Root Mean Squared Error {:.2f} Mean Squared Error {:.2f}".format(rmse_cv,
                                                                                        mse_cv) + sf.Color.END)

    rmse_test /= num_model_iterations
    mse_test /= num_model_iterations

    log.info(sf.Color.BOLD + sf.Color.DARKCYAN +
             "\nTest Root Mean Squared Error {:.2f} Mean Squared Error {:.2f}".format(rmse_test,
                                                                                      mse_test) + sf.Color.END)
    if run_test_only:
        return y_pred_test
    else:
        return y_pred_cv

# Run k-fold cross-validation and predict on test dataproc from last trained model


def run_kfold_cv(x_train, y_train, x_test, clf_class, **kwargs):

    # Construct a kfolds object from train dataproc
    kf = KFold(len(y_train), n_folds=5, shuffle=True)
    y_pred_cv = y_train.copy()

    # log.debug(kf)
    # Initialize to avoid pep8 warning, thought clf will always be initialized below
    clf = 0

    # Iterate through folds
    for train_index, cv_index in kf:
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        x_cv_train, x_cv_test = x_train[train_index], x_train[cv_index]
        y_cv_train = y_train[train_index]

        if train_index[0]:
            log.debug(clf)

        clf.fit(x_cv_train, y_cv_train)

        # Predict on cv dataproc
        y_pred_cv[cv_index] = clf.predict(x_cv_test)

    # Now predict test dataproc on trained model
    y_pred_test = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.debug(clf.feature_importances_)

    # log.info(clf.estimators_)

    return y_pred_cv, y_pred_test


# Train model and test with CV dataset (over num_train_iterations). Use trained model to predict on test dataproc
def run_cv_test(x_train, y_train, x_test, num_train_iterations=1, test_size=0.2, clf_class=RandomForestReg, **kwargs):
    # Further split train dataproc into train / cv split
    x_cv_train, x_cv_test, y_cv_train, y_cv_test = train_test_split(x_train, y_train,
                                                                    test_size=test_size, random_state=42)

    # Initialize a classifier with key word arguments. warm_start always need to be set to true as we want model
    # to use solution of previous call to fit
    clf = clf_class(**kwargs)

    log.debug(clf)

    time.sleep(5)  # sleep time in seconds

    for i in range(num_train_iterations):
        print "Iteration ", i
        clf.set_params(n_estimators=10000*(i+1))
        clf.fit(x_cv_train, y_cv_train)

        clf.predict(x_cv_test)

    y_pred_test = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.debug(clf.feature_importances_)

    # log.info(clf.estimators_)

    return y_pred_test


# Train model and predict on test dataproc
def run_test(x_train, y_train, x_test, clf_class, **kwargs):
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    log.debug(clf)

    time.sleep(5)  # sleep time in seconds

    clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature Importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.debug(clf.feature_importances_)

    # log.info(clf.estimators_)

    return y_pred_test


##################################################################################################################
