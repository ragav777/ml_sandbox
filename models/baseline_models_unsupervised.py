#########################################################################################################
#  Contains a collection of unsupervised learning algorithms
#
#########################################################################################################

# sklearn Toolkit
import logging

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from lib.utils import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "agopalak"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

log = logging.getLogger(__name__)

#########################################################################################################


def run_clustering(x, make_plots=False, clf_class=KMeans, min_cluster=3, max_cluster=3, **kwargs):
    """
    Runs any supported clustering algorithms (Support List: K-Means)

    :param x:
    :param make_plots:
    :param clf_class:
    :param min_cluster:
    :param max_cluster:
    :param kwargs:
    :return:
    """

    log.info("Feature Space Holds %d Observations and %d Features" % x.shape)

    # Run model on test dataproc
    log.info(sf.Color.BOLD + sf.Color.BLUE + "Runnning K-Means" + sf.Color.END)

    # Creating empty output cluster array
    x_clusters_cumm = np.empty([x.shape[0], 0])
    x_clusters_cols = []
    log.debug('Dimensions of Output Cluster Array @ Start: %s', x_clusters_cumm.shape)

    for i in range(min_cluster, max_cluster+1):

        log.info(sf.Color.BOLD + sf.Color.YELLOW + "Number of Clusters: %d" % i + sf.Color.END)
        x_clusters_cols.append(str(i) + ' Cluster Run')

        # Run K-Means Clustering
        clf = clf_class(n_clusters=i, **kwargs)
        clf.fit(x)
        x_clusters = clf.predict(x)

        # Re-shape Clustering Output
        x_clusters = x_clusters.reshape(-1, 1)
        log.debug('Dimensions of K-Means Run Array: %s', x_clusters.shape)

        # Merge individual cluster runs
        x_clusters_cumm = np.hstack((x_clusters_cumm, x_clusters))
        log.debug('Dimensions of Output Cluster Array: %s', x_clusters_cumm.shape)

    # TODO: Need to write clustering plot routines
    if make_plots:
        pass

    # Create a Pandas Data Frame
    x_clusters_cumm_df = pd.DataFrame(x_clusters_cumm, columns=x_clusters_cols)

    return x_clusters_cumm_df

##################################################################################################################
