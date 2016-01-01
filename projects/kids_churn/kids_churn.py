#########################################################################################################
#  Description: Main file for kids_churn dataset. Key function is to transform dataset into needed
#  input_features and output. Currently only synthetic dataproc option available
#########################################################################################################
from __future__ import division

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestClassifier as RandomForest
# from sklearn.ensemble import BaggingClassifier as Bagging
# from sklearn.svm import SVC as SVC  # Support vector machines
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.linear_model import LogisticRegression as LogReg
# from sklearn.linear_model import RidgeClassifier as Ridge
# from sknn.mlp import Classifier as NeuralNetClassifier, Layer as NeuralNetLayer
# from sklearn.ensemble import GradientBoostingClassifier as GradBoost
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import logging.config

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

__dir__ = os.path.dirname(__file__)
__data__ = os.path.join(__dir__, 'data')
__root__ = os.path.join(__dir__, '../..')
__utils__ = os.path.join(__root__, 'lib/utils')
__plots__ = os.path.join(__root__, 'lib/visualization')

#########################################################################################################

sys.path.append(__root__)
# Ignore PEP8 Warning. Need path before import to enable execution from project folder
from lib.utils import support_functions as sf
from lib.visualization import visualization
from models import ensemble_models


# Setup logging

log = logging.getLogger("info")

#########################################################################################################


def kids_churn(use_synthetic_data=False, feature_scaling=True):

    log.info("Importing data")
    if use_synthetic_data:
        in_data_file = os.path.join(__data__, 'synthetic_kids_ver1.csv')
        if os.path.isfile(in_data_file):
            churn_df = pd.read_csv(in_data_file, sep=',')
        else:
            raise ValueError("Synthetic data not available")
    else:
        raise ValueError("Actual data not available")

    col_names = churn_df.columns.tolist()

    log.info(sf.Color.BOLD + sf.Color.GREEN + "Column names:" + sf.Color.END)
    log.info(col_names)

    to_show = col_names[:]

    log.info(sf.Color.BOLD + sf.Color.GREEN + "\nSample dataproc:" + sf.Color.END)
    log.info(churn_df[to_show].head(6))

    # Isolate target data
    y = np.array(churn_df['Churn'])

    log.debug(y)

    to_drop = ['Churn']

    churn_feat_space = churn_df.drop(to_drop, axis=1)

    feature_names = churn_feat_space.columns.tolist()

    log.debug(feature_names)

    x = churn_feat_space.as_matrix().astype(np.float)

    if feature_scaling:
        # Feature scaling and normalization
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    log.debug(x)

    y = np.array(y)

    log.info("Feature space holds %d observations and %d features" % x.shape)
    log.info("Unique target labels:")
    log.info(np.unique(y))

    return [x, y]

####################################################################################################


def main():

    start_time = time.time()

    prec_recall_plot = True

    # Choose models for the ensemble. Uncomment to choose model needed
    estimator_model0 = RandomForest
    estimator_keywords_model0 = dict(n_estimators=100, verbose=0, criterion='gini', n_jobs=-1,
                                     max_features=5, warm_start='True', oob_score='True', bootstrap='True')

    # estimator_model1 = GradBoost
    # estimator_keywords_model1 = dict(n_estimators=100, loss='deviance', learning_rate=0.01, verbose=0, max_depth=5,
    #                                  subsample=1.0)

    # estimator = SVC
    # estimator_keywords = dict(C=1, kernel='rbf', class_weight='auto')
    # estimator_model2 = LogReg
    # estimator_keywords_model2 = dict(solver='liblinear')

    # dict model names and parameters always need to have keys model0, model1, model2...
    model_names_list = dict(model0=estimator_model0)
    model_parameters_list = dict(model0=estimator_keywords_model0)

    [input_features, output] = kids_churn(use_synthetic_data=True, feature_scaling=True)

    # ensemble_models.majority_voting(input_features, output, model_names_list, model_parameters_list,
    #                                 run_cv_flag=False, num_model_iterations=1, plot_learning_curve=False,
    #                                 run_prob_predictions=True, classification_threshold=0.5)

    if prec_recall_plot:
        # Divide 0 and 0.9 by 21 equally distributed values (including both).
        # Ignoring 1.0 as it has Fbeta_score of 0
        num_of_thresholds = np.linspace(0, 0.9, 21)

        threshold = np.zeros((len(num_of_thresholds), 1), dtype=float)

        precision = np.zeros((len(num_of_thresholds), 1), dtype=float)

        recall = np.zeros((len(num_of_thresholds), 1), dtype=float)

        fbeta_score = np.zeros((len(num_of_thresholds), 1), dtype=float)

        idx = 0

        for classification_threshold in num_of_thresholds:
            prec_recall = ensemble_models.average_prob(input_features, output, model_names_list, model_parameters_list,
                                                       run_cv_flag=False, num_model_iterations=1,
                                                       plot_learning_curve=False, run_prob_predictions=True,
                                                       classification_threshold=classification_threshold)

            threshold[idx] = classification_threshold
            precision[idx] = round(prec_recall[0] * 100)  # Convert to %
            recall[idx] = round(prec_recall[1] * 100)
            fbeta_score[idx] = round(prec_recall[2] * 100)

            idx += 1

        # Call function for plotting
        vis = visualization.Plots()
        vis.basic_2d_plot(x=threshold, y=(precision, recall, fbeta_score),
                          legends=("Precision", "Recall", "Fbeta_score (beta=2)"),
                          title="Precision Recall Curve", xaxis_label="Classification Threshold",
                          yaxis_label="Score %")

    ##################################

    print("Total time: %0.3f" % float(time.time() - start_time))

####################################################################################################

if __name__ == "__main__":

    print('Current Working Directory: %s' % os.getcwd())

    # Setup Logging
    logging_config_file = os.path.join(__utils__, 'logging.conf')
    logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
    log = logging.getLogger('debug')

    # Call main()
    main()
