#########################################################################################################
#  Description: Create synthetic dataproc to add to original dataset
#  to obtain dataproc that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################
import logging
import logging.config
import random

import pandas as pd

from lib import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging
logging.config.fileConfig('logging.conf')

logger = logging.getLogger("debug")


# Adding synthetic dataproc to increase dataproc size - Create temporary df and then randomly modify values
# (between 80% to 120% of original value) in there and concat to churn_df.
# Need to be run only once to save time, then new df will be saved to file dataproc/data_synthetic.csv
def create_synthetic_data_offsets_from_original():
    churn_df = pd.read_csv('dataproc/train_data.csv', sep=', ')

    temp_df = churn_df.copy()

    for index, row in temp_df.iterrows():
        # Return a random number between 0.8 and 1.2 for account length and # of vmail messages. For churned customers,
        # reduce random rate range to 0.95 to 1.05 (or you are introducing too much noise and
        # algo doesn't perform well)
        if temp_df.iat[index, 20] == "True.":
            random_multiplier = sf.random_float(0.95, 1.05)
        else:
            random_multiplier = sf.random_float(0.8, 1.2)

        temp_value = temp_df.iat[index, 1]
        temp_df.iloc[index:index + 1, 1:2] = int(temp_value * random_multiplier)

        if temp_df.iat[index, 20] == "True.":
            random_multiplier = sf.random_float(0.95, 1.05)
        else:
            random_multiplier = sf.random_float(0.8, 1.2)

        temp_value = temp_df.iat[index, 6]
        temp_df.iloc[index:index + 1, 6:7] = int(temp_value * random_multiplier)

        # Modify columns 7-19 as well, but use same random number for mins, calls and charges,
        # as they have to all be proportional. Each have a different format hence being calculated individually
        for col in range(7, 19, 3):
            if temp_df.iat[index, 20] == "True.":
                random_multiplier = sf.random_float(0.95, 1.05)
            else:
                random_multiplier = sf.random_float(0.8, 1.2)

            temp_value = temp_df.iat[index, col]
            temp_df.iloc[index:index + 1, col:col + 1] = float("{0:.1f}".format(temp_value * random_multiplier))

            temp_value = temp_df.iat[index, col + 1]
            temp_df.iloc[index:index + 1, col + 1:col + 2] = int(temp_value * random_multiplier)

            temp_value = temp_df.iat[index, col + 2]
            temp_df.iloc[index:index + 1, col + 2:col + 3] = float("{0:.2f}".format(temp_value * random_multiplier))

    concat_df = [churn_df, temp_df]

    churn_df = pd.concat(concat_df, ignore_index=True)

    churn_df.to_csv('dataproc/data_synthetic.csv', sep=',')

    logger.debug(churn_df)


# Adding synthetic dataproc to increase dataproc size - use mean and standard deviation from each feature's original dataproc
# to create more artificial dataproc. Need to be run only once to save time, then new df will be saved to
# file dataproc/data_synthetic.csv
def create_synthetic_data_gaussian():
    churn_df = pd.read_csv('dataproc/train_data.csv', sep=', ')

    # Create a copy to modify
    temp_df = churn_df.copy()

    # churn_df_stats = pd.DataFrame([churn_df.mean(), churn_df.std()], index=['mean', 'std'])

    # Find mean / standard deviation for customers who churned / stayed separately
    groupby_churn_mean = churn_df.groupby('Churn?').mean()
    groupby_churn_std = churn_df.groupby('Churn?').std()

    # Find mean for charge per min, to be applied to new random mins calculated later
    charge_per_min_mean = [(churn_df['Day Charge'] / churn_df['Day Mins']).mean(),
                           (churn_df['Eve Charge'] / churn_df['Eve Mins']).mean(),
                           (churn_df['Night charge'] / churn_df['Night Mins']).mean(),
                           (churn_df['Intl Charge'] / churn_df['IntlMins']).mean()]

    min_list = [u'Day Mins', u'Eve Mins', u'Night Mins', u'IntlMins']
    for column_name in temp_df.columns:
        # Apply mean and sigma obtained for customers who churned or not seperately using values calculated from
        # original dataproc
        if column_name in [u'Account Length', u'VMail Message', u'Day Calls', u'Eve Calls', u'Night Calls',
                           u'Intl Calls', u'CustServ Calls']:
            temp_groupby_churn_false = temp_df.groupby('Churn?').get_group('False.')[column_name].apply(
                lambda x: abs(int(random.gauss(groupby_churn_mean[column_name][0], groupby_churn_std[column_name][0]))))
            temp_groupby_churn_true = temp_df.groupby('Churn?').get_group('True.')[column_name].apply(
                lambda x: abs(int(random.gauss(groupby_churn_mean[column_name][1], groupby_churn_std[column_name][1]))))
            temp_df[column_name] = (pd.concat([temp_groupby_churn_true, temp_groupby_churn_false])).sort_index()
        elif column_name in min_list:
            temp_groupby_churn_false = temp_df.groupby('Churn?').get_group('False.')[column_name].apply(lambda x: abs(
                float("{0:.1f}".format(
                    random.gauss(groupby_churn_mean[column_name][0], groupby_churn_std[column_name][0])))))
            temp_groupby_churn_true = temp_df.groupby('Churn?').get_group('True.')[column_name].apply(lambda x: abs(
                float("{0:.1f}".format(
                    random.gauss(groupby_churn_mean[column_name][1], groupby_churn_std[column_name][1])))))
            temp_df[column_name] = (pd.concat([temp_groupby_churn_true, temp_groupby_churn_false])).sort_index()

    charge_list = [u'Day Charge', u'Eve Charge', u'Night charge', u'Intl Charge']
    for column_name in charge_list:
        # Since charges are directly derived from mins, we shouldn't randomize charge
        # Multiply # of mins with charge_per_min_mean from original dataproc. Need to make sure this dataproc has a really
        # low sigma or below approach might be invalid
        temp_df[column_name] = temp_df[min_list[charge_list.index(column_name)]] * charge_per_min_mean[
            charge_list.index(column_name)]
        temp_df[column_name] = temp_df[column_name].apply(lambda x: float("{0:.2f}".format(x)))

    groupby_temp_mean = temp_df.groupby('Churn?').mean()
    groupby_temp_std = temp_df.groupby('Churn?').std()

    # Compare original and synthetic dataproc stats
    logger.debug("\noriginal dataproc mean\n")
    logger.debug(groupby_churn_mean)

    logger.debug("\nSynthetic dataproc mean\n")
    logger.debug(groupby_temp_mean)

    logger.debug("\noriginal dataproc sigma\n")
    logger.debug(groupby_churn_std)

    logger.debug("\nSynthetic dataproc sigma\n")
    logger.debug(groupby_temp_std)

    concat_df = [churn_df, temp_df]

    churn_df = pd.concat(concat_df, ignore_index=True)

    churn_df.to_csv('dataproc/data_synthetic.csv', sep=',')

    logger.debug(churn_df)


if __name__ == "__main__":
    # create_synthetic_data_offsets_from_original()
    create_synthetic_data_gaussian()
