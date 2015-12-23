#########################################################################################################
# Description: Function trying to predict fantasy points, yards and TDs for various players based on
# their performance over past many years
#
#########################################################################################################

# Imports for various models (Turn on as needed)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor as RandomForestReg
from sklearn.preprocessing import LabelEncoder
# sklearn Toolkit
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import os
import sys
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
from models import baseline_models_regression as ml_reg
from models import baseline_models_unsupervised as ml_us

#########################################################################################################


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        # Array of column names to encode
        self.columns = columns

    def fit(self, x, y=None):
        return self  # Not Relevant Here

    def transform(self, x):

        """
        def transform(self,x):
        Transforms columns of x specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in x.
        """

        out = x.copy()
        if self.columns is not None:
            for col in self.columns:
                out[col] = LabelEncoder().fit_transform(out[col])
        else:
            for colname, col in out.iteritems():
                out[colname] = LabelEncoder().fit_transform(col)
        return out

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


#########################################################################################################


def prep_reg_nfl_pred(feature_scaling=False, data_csv='dataproc/nfl_pred_data.csv'):

    log = logging.getLogger('debug')

    nfl_df = sf.load_model_data(data_csv)

    # Previewing Column Names
    col_names = nfl_df.columns.tolist()
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Column Names:" + sf.Color.END)
    log.info(col_names)

    # Previewing Source Data
    to_show = col_names[:]
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Loaded Data:" + sf.Color.END)
    log.info(nfl_df[to_show].head(3))

    # Isolate Output Data
    y = np.array(nfl_df['points'])

    # Columns to drop (From Features Data Frame)
    to_drop = ['points', 'rush_att', 'rush_td', 'rush_yd', 'rec_td', 'rec_yd', 'fumble', 'pass_yd',
               'pass_td', 'pass_int']
    base_feat_space = nfl_df.drop(to_drop, axis=1)

    # Previewing Feature Names
    feature_names = base_feat_space.columns.tolist()
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Feature Names:" + sf.Color.END)
    log.debug(feature_names)

    # Using Label Encoding to Rebase the Values in these Columns
    base_feat_space = MultiColumnLabelEncoder(columns=['name', 'pos', 'year', 'against', 'team']).fit_transform(
        base_feat_space)
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Post Label Encoding:" + sf.Color.END)
    log.info(base_feat_space.head(3))

    # Make NumPy Array For Base Features
    # base_x = base_feat_space.as_matrix().astype(np.float)

    # Handling Polynomials based on Base Features
    features_with_polynomials = ['rush_att_av', 'rush_td_av', 'rush_yd_av', 'rec_td_av', 'rec_yd_av', 'fumble_av',
                                 'pass_yd_av', 'pass_td_av', 'pass_int_av']
    polynomial_feat_space = base_feat_space[features_with_polynomials].copy(deep=True)
    for feature in features_with_polynomials:
        new_feature = feature + '^2'
        log.debug('New Feature %s Added Based on %s', new_feature, feature)
        polynomial_feat_space[new_feature] = polynomial_feat_space[feature] ** 2

    # Removing original features from polynomial feature space
    polynomial_feat_space = polynomial_feat_space.drop(features_with_polynomials, axis=1)
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Polynomial Features:" + sf.Color.END)
    log.info(polynomial_feat_space.head(3))

    # Make NumPy Array for Polynomial Features
    # polynomial_x = polynomial_feat_space.as_matrix().astype(np.float)
    # Scaling & Normalization

    if feature_scaling:

        # Base Features - Scaling & Normalization
        base_feat_space = pd.DataFrame(preprocessing.StandardScaler().fit_transform(base_feat_space),
                                       columns=base_feat_space.columns)
        log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Post Base Features Scaling:" + sf.Color.END)
        log.info(base_feat_space.head(3))

        # Polynomial Features - Scaling & Normalization
        polynomial_feat_space = pd.DataFrame(preprocessing.StandardScaler().fit_transform(polynomial_feat_space),
                                             columns=polynomial_feat_space.columns)
        log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Post Polynomial Features Scaling:" + sf.Color.END)
        log.info(polynomial_feat_space.head(3))

    # Merge Base & Polynomial Features; Convert to NumPy Array
    x_df = pd.concat([base_feat_space, polynomial_feat_space], axis=1)
    x = x_df.as_matrix().astype(np.float)

    # Handle Feature Scaling and Normalization
    # if feature_scaling:
    #    scaler = StandardScaler()
    #    x = scaler.fit_transform(x)

    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Transformed Data:" + sf.Color.END)
    log.info(x[0:3])

    log.info("Feature Space holds %d Observations and %d Features" % x.shape)

    return [x, y, nfl_df]


##################################################################################################################


def prep_us_nfl_pred(input_df, use_csv=False, data_csv='data/output_dont_commit/reg_output.csv'):

    if use_csv:
        input_df = sf.load_model_data(data_csv)

    # Model Run - K-Means Clustering - Data Preparation
    # Adding Mean Squared Error Column
    input_df['Squared Error'] = input_df.apply(lambda row: ((row['Total points'] -
                                                             row['Total points predicted']) ** 2), axis=1)

    # Dropping Unwanted Columns
    us_columns_to_drop = ['name', 'pos', 'year', 'home', 'against', 'week', 'score', 'opp_score',
                          'month', 'team', 'rush_att', 'rush_td', 'rush_yd', 'rec_td', 'rec_yd',
                          'fumble', 'pass_yd', 'pass_td', 'pass_int', 'rush_att_av',
                          'rush_td_av', 'rush_yd_av', 'rec_td_av', 'rec_yd_av', 'fumble_av',
                          'pass_yd_av', 'pass_td_av', 'pass_int_av', 'rush_att_av^2',
                          'rush_td_av^2', 'rush_yd_av^2', 'rec_td_av^2', 'rec_yd_av^2', 'fumble_av^2',
                          'pass_yd_av^2', 'pass_td_av^2', 'pass_int_av^2', 'points', 'Total points predicted']
    us_input_df = input_df.drop(us_columns_to_drop, axis=1)

    log.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Clustering Input Data:" + sf.Color.END)
    log.info(us_input_df.head(3))

    # Converting to NumPy Array
    input_npa = us_input_df.as_matrix().astype(np.float)

    return input_df, input_npa


def main():

    log = logging.getLogger(__name__)

    # Machine Learning Chosen Models
    reg_estimator = RandomForestReg
    us_estimator = KMeans

    # To Run Stage or Not To?
    run_reg = True
    run_us = False

    # Run Data Prep Only
    # To debug dataproc processing related issues
    run_only_dataprep = False

    start_time = time.time()

    if run_reg:

        # Model Run - Regression - Random Forest - Estimator Keywords = dict()
        reg_estimator_keywords = dict(n_estimators=10000, verbose=0, warm_start='True', n_jobs=-1,
                                      max_features=5)

        # Model Run - Regression - Data Preparation
        in_data_file = os.path.join(__data__, 'nfl_pred_data.csv')
        [reg_input_npa, reg_output_npa, reg_df] = prep_reg_nfl_pred(feature_scaling=False,
                                                                    data_csv=in_data_file)

        log.debug('Input Dimensions: %s', reg_df.shape)

        if not run_only_dataprep:
            # Model Run - Regression - Random Forest
            log.info('Running Regression.....')
            reg_pred_df = ml_reg.run_models_wrapper(reg_input_npa, reg_output_npa, run_cv_flag=True,
                                                    num_model_iterations=1,
                                                    plot_learning_curve=False, test_size=0.2, clf_class=reg_estimator,
                                                    **reg_estimator_keywords)

            log.debug('Output Dimensions: %s', reg_pred_df.shape)

            # Model Run - Regression - Output Processing
            # Combine Input & Output Data Frames
            reg_result_df = pd.concat([reg_df, reg_pred_df], axis=1)

            # Model Run - Regression - Data Recording
            # Write Regression Results to CSV
            reg_out_file = os.path.join(__data__, 'output_dont_commit/reg_output.csv')
            reg_result_df.to_csv(reg_out_file)

            # Model Run - Regression - Data Recording
            # Write Transformed Inputs to CSV (To verify dataproc coherency and ordering)
            # Don't need this after verification done
            reg_in_file = os.path.join(__data__, 'output_dont_commit/reg_transformed_input.csv')
            np.savetxt(reg_in_file, reg_input_npa, delimiter=",")

    if run_us:

        # Model Run - Clustering - K-Means - Estimator Keywords = dict()
        us_estimator_keywords = dict(init='k-means++', n_init=10, verbose=0)
        if run_reg:
            [reg_result_df, us_input_npa] = prep_us_nfl_pred(reg_result_df, use_csv=False, data_csv='dummy.csv')
        else:
            reg_out_file = os.path.join(__data__, 'output_dont_commit/reg_output.csv')
            dummy_df = pd.DataFrame(np.nan, index=[0], columns=['A'])
            [reg_result_df, us_input_npa] = prep_us_nfl_pred(dummy_df, use_csv=True,
                                                             data_csv=reg_out_file)

        if not run_only_dataprep:
            # Model Run - K-Means Clustering
            us_kcluster_df = ml_us.run_clustering(us_input_npa, make_plots=False, clf_class=us_estimator, min_cluster=3,
                                                  max_cluster=5, **us_estimator_keywords)

            # Model Run - K-Means Clustering - Output Processing
            # Combine Input & Output Data Frames
            us_result_df = pd.concat([reg_result_df, us_kcluster_df], axis=1)

            # Model Run - K-Means Clustering - Data Recording
            # Write Regression Results to CSV
            us_output_file = os.path.join(__data__, 'output_dont_commit/us_output.csv')
            us_result_df.to_csv(us_output_file)

    print("Total time: %0.3f" % float(time.time() - start_time))

##################################################################################################################

if __name__ == "__main__":

    print('Current Working Directory: %s' % os.getcwd())

    # Setup Logging
    logging_config_file = os.path.join(__utils__, 'logging.conf')
    logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # Call main()
    main()

##################################################################################################################
