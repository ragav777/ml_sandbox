#########################################################################################################
#  Description: Main file for telecom_churn dataset. Key function is to transform dataset into needed
#  input_features and output
#########################################################################################################
from __future__ import division

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestClassifier as RandomForest
# from sklearn.ensemble import BaggingClassifier as Bagging
# from sklearn.svm import SVC as SVC  # Support vector machines
# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LogReg
# from sklearn.linear_model import RidgeClassifier as Ridge
# from sknn.mlp import Classifier as NeuralNetClassifier, Layer as NeuralNetLayer
from sklearn.ensemble import GradientBoostingClassifier as GradBoost
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
from models import ensemble_models as ml_ens

#########################################################################################################

# Setup logging

log = logging.getLogger("info")

#########################################################################################################


def telecom_churn(use_synthetic_data=False, feature_scaling=True):

    print('Current Working Directory: %s' % os.getcwd())

    log.debug("Importing data")
    if use_synthetic_data:
        in_data_file = os.path.join(__data__, 'data_synthetic.csv')
        if os.path.isfile(in_data_file):
            churn_df = pd.read_csv(in_data_file, sep=',')
        else:
            raise ValueError("Synthetic data not available")
        # split rows for working on partial dataproc
        start_row = 5000
        end_row = 9999
    else:
        in_data_file = os.path.join(__data__, 'train_data.csv')
        churn_df = pd.read_csv(in_data_file, sep=', ')
        # split rows for working on partial dataproc
        start_row = 0
        end_row = 4999

    churn_df = churn_df.iloc[start_row:end_row].copy()

    churn_df = churn_df.reset_index()

    col_names = churn_df.columns.tolist()

    log.info(sf.Color.BOLD + sf.Color.GREEN + "Column names:" + sf.Color.END)
    log.info(col_names)

    to_show = col_names[:6] + col_names[-6:]

    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample dataproc:" + sf.Color.END)
    log.info(churn_df[to_show].head(6))

    # Isolate target data
    churn_result = churn_df['Churn?']
    y = np.where(churn_result == 'True.', 1, 0)

    log.debug(y)

    # We don't need these columns. Index is created only when do a partial split
    if 'index' in col_names:
        to_drop = ['index', 'Area Code', 'Phone', 'Churn?']
    else:
        to_drop = ['Area Code', 'Phone', 'Churn?']

    churn_feat_space = churn_df.drop(to_drop, axis=1)

    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan", "VMail Plan"]

    churn_feat_space[yes_no_cols] = (churn_feat_space[yes_no_cols] == 'yes')

    # Below segment replaces column 'State' with a number for each state (alphabetically sorted)
    # separate state into it's own df as it's easier to operate on later
    churn_feat_state = churn_feat_space[['State']]

    state = np.unique(churn_feat_state)

    for index, row in churn_feat_state.iterrows():
        churn_feat_state.iat[index, 0] = int(np.where(state == row['State'])[0])

    churn_feat_space['State'] = churn_feat_state

    # log.debug(churn_feat_space['State'])

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
    log.info("Unique target labels: ")
    log.info(np.unique(y))

    return [x, y]

####################################################################################################


def main():

    start_time = time.time()

    # Choose models for the ensemble. Uncomment to choose model needed
    estimator_model0 = RandomForest
    estimator_keywords_model0 = dict(n_estimators=1000, verbose=0, criterion='entropy', n_jobs=-1,
                                     max_features=5, class_weight='auto')

    estimator_model1 = GradBoost
    estimator_keywords_model1 = dict(n_estimators=1000, loss='deviance', learning_rate=0.01, verbose=0, max_depth=5,
                                     subsample=1.0)

    # estimator = SVC
    # estimator_keywords = dict(C=1, kernel='rbf', class_weight='auto')
    estimator_model2 = LogReg
    estimator_keywords_model2 = dict(solver='liblinear')

    # dict model names and parameters always need to have keys model0, model1, model2...
    model_names_list = dict(model0=estimator_model0, model1=estimator_model1, model2=estimator_model2)
    model_parameters_list = dict(model0=estimator_keywords_model0, model1=estimator_keywords_model1,
                                 model2=estimator_keywords_model2)

    [input_features, output] = telecom_churn(use_synthetic_data=False, feature_scaling=True)

    ml_ens.majority_voting(input_features, output, model_names_list, model_parameters_list,
                           run_cv_flag=False, num_model_iterations=1, plot_learning_curve=False)

    ##################################

    # Neural network
    # estimator = NeuralNetClassifier
    # estimator_keywords = dict(layers=[NeuralNetLayer("Rectifier", units=64), NeuralNetLayer("Rectifier", units=32),
    #                                   NeuralNetLayer("Softmax")],
    #                           learning_rate=0.001, n_iter=50)

    # Pep8 shows a warning for all other estimators other than RF (probably because RF is the default class in
    # telecom / kids churn. This is not a valid warning and has been validated

    # Choose problem to solve

    # telecom_churn(use_synthetic_data=False, num_model_iterations=1, plot_learning_curve=False, feature_scaling=True,
    #               clf_class=estimator, **estimator_keywords)

    # telecom_churn(use_synthetic_data=True, num_model_iterations=1, plot_learning_curve=True, feature_scaling=True,
    #               clf_class=estimator, **estimator_keywords)

    # kids_churn(use_synthetic_data=True, num_model_iterations=1, plot_learning_curve=False, feature_scaling=True,
    #            clf_class=estimator, **estimator_keywords)

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
