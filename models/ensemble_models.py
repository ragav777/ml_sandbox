#########################################################################################################
#  Description: Functions to create ensembles.
#  Description: Functions to create ensembles.
#
#########################################################################################################
# sklearn Toolkit
import logging
import logging.config

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from models import baseline_models
from lib.utils import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

log = logging.getLogger("info")

#########################################################################################################

# Use majority voting to predict classes of new ensemble. For even number of models, split = majority!
def majority_voting(input_features, output, model_names, model_parameters, run_cv_flag=False, num_model_iterations=1,
                    plot_learning_curve=False, run_prob_predictions=True):
    # Check if a minimum of 3 models are there
    if len(model_names) < 2:
        raise ValueError("Need a minimum of 2 models to do an ensemble")

    actual_output_values = dict()
    predicted_output_values = dict()

    num_of_models = len(model_names)

    # Get actual and predicted values for each model
    for idx in range(num_of_models):
        model_key = "model{:d}".format(idx)
        log.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning Model {:s}".format(model_names[model_key]) +
                    sf.Color.END)
        # Append to dictionary with dynamically created key names above
        [actual_output_values[model_key], predicted_output_values[model_key]] = \
            baseline_models.run_models_wrapper(x=input_features, y=output, run_cv_flag=run_cv_flag,
                                               num_model_iterations=num_model_iterations,
                                               plot_learning_curve=plot_learning_curve,
                                               run_prob_predictions=run_prob_predictions,
                                               clf_class=model_names[model_key], **model_parameters[model_key])

        # accuracy(actual_output_values[actual_output_name], predicted_output_values[predicted_output_name])

    y_predicted_ensemble = predicted_output_values['model0'].copy()

    # # Create ensemble prediction using majority voting scheme
    for sample in np.ndindex(predicted_output_values['model0'].shape):
        y_predicted_sum = 0  # Reset for every sample
        for actual_key_name in actual_output_values.iterkeys():
            if predicted_output_values[actual_key_name][sample]:
                y_predicted_sum += 1

        # Need to have either numerator or denominator in round() as float to roundup
        if y_predicted_sum >= round(num_of_models / 2.0):
            y_predicted_ensemble[sample] = 1
        else:
            y_predicted_ensemble[sample] = 0

    accuracy_value = baseline_models.accuracy(actual_output_values['model0'], y_predicted_ensemble)

    beta = 2.0

    prec_recall = precision_recall_fscore_support(y_true=actual_output_values['model0'], y_pred=y_predicted_ensemble,
                                                  beta=beta, average='binary')

    # Log Accuracy and precision / recall values for the ensemble
    log.info(sf.Color.BOLD + sf.Color.DARKCYAN + "\nEnsemble output for test dataproc" + sf.Color.END)

    log.info(
        sf.Color.BOLD + sf.Color.DARKCYAN + "\nAccuracy {:.2f}".format(accuracy_value * 100) +
        sf.Color.END)

    log.info(
        sf.Color.BOLD + sf.Color.DARKCYAN + "\nPrecision {:.2f} Recall {:.2f} Fbeta-score {:.2f}".format(
            prec_recall[0] * 100, prec_recall[1] * 100, prec_recall[2] * 100) + sf.Color.END)

##################################################################################################################
