"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from ikats.core.library.spark import SSessionManager, SparkUtils
from ikats.core.resource.api import IkatsApi

import time
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

from ikats.core.library.exception import IkatsException, IkatsInputTypeError
from pyspark.ml.clustering import KMeans as KMeansSpark
from pyspark.ml.linalg import Vectors

LOGGER = logging.getLogger(__name__)

# def crash_test(nb_ts, length_ts, nb_clu):
#     a = np.random.rand(nb_ts, length_ts)
#     model = KMeans(nb_clu)
#     model.fit(a)
#     return model.labels_

# TODO: 1 - AJUSTER le modèle - Partie sklearn
def fit_kmeans_sklearn_internal(ts_list, n_cluster, random_state=None):
    """
    The internal wrapper to fit K-means on time series with scikit-learn.

    :param data: the dataset
    :type data : a list of dicts

    :param n_cluster: theumber of clusters to form
    :type n_cluster: int

    :param random_state: the seed used by the random number generator (if int)
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state: int or NoneType
    .. note:: specify `random_state` to make the results reproducible

    :return model: The KMeans model fitted on the input data-set
    :rtype model: sklearn.cluster.k_means_.KMeans

    :raises IkatsException: error occurred.
    """
    LOGGER.info(" --- Starting K-Means fit with scikit-learn --- ")
    try:
        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        data = []
        for i in ts_list:
            startLoadingTime = time.time()
            # Read TS from it's TSUID; shape = (2, nrow)
            ts_data = IkatsApi.ts.read([i['tsuid']])[0]
            LOGGER.debug("TSUID: %s, Gathering time: %.3f seconds", i, time.time() - startLoadingTime)
            lisTS = [i[1] for i in ts_data]
            data.append(lisTS)
        datArray = np.array(data)
        # TODO : Ce bout de code est réutilisé pour être le point d'entrée de MDS

        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        # TODO: Attention : de base, c'est la distance euclidienne qui est utilisée dans sklearn. Incorporer DTW ?
        model = KMeans(n_clusters=n_cluster, random_state=random_state)
        model.fit(datArray)
        LOGGER.info(" --- Finished fitting K-Means to data --- ")
        # --------------------------------
        # 3 - Display the labels clustered
        # --------------------------------
        LOGGER.info(" --- Exporting results to sklearn.cluster.KMeans format --- ")
        return model
    except Exception:
        msg = "Unexpected error: fit_kmeans_sklearn_internal(..., {}, {}, {})"
        raise IkatsException(msg.format(ts_list, n_cluster, random_state))

# TODO: 1bis - AJUSTER le modèle - Partie Spark
def fit_kmeans_spark_internal(ts_list, n_cluster, random_state=None):
    """
    The internal wrapper to fit K-means on time series with Spark.

    :param data: The data-set : key (TS id), values (list of floats)
    :type data : list of dicts

    :param n_cluster: Number of clusters to form as well as the number of centroids to generate
    :type n_cluster: int

    :param random_state: The seed used by the random number generator (if int).
    If None, the random number generator is the RandomState instance used by np.random.
    :type random_state: int or NoneType

    .. note:: specify `random_state` to make the results reproducible.

    :return model: The KMeans model fitted on the input data-set.
    :rtype model:

    :raises IkatsException: error occurred.
    """
    SSessionManager.get()
    spark = SSessionManager.spark_session
    spark.sparkContext.setLogLevel('WARN')
    LOGGER.info(" --- Starting K-Means fit with Spark --- ")
    try:
        # -----------------------------------------------------------
        # 1 - Process the data in the shape needed by Spark's K-Means
        # -----------------------------------------------------------
        # Perform operation iteratively on each TS
        data = []
        for i in ts_list:
            startLoadingTime = time.time()
            # Read TS from it's TSUID; shape = (2, nrow)
            ts_data = IkatsApi.ts.read([i['tsuid']])[0]
            LOGGER.debug("TSUID: %s, Gathering time: %.3f seconds", i, time.time() - startLoadingTime)
            lisTS = [x[1] for x in ts_data]
            vecTS = Vectors.dense(lisTS)
            data.append((vecTS,))
        dataDf = spark.createDataFrame(data, ['time_series'])
        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        # TODO: Attention : de base, c'est la distance euclidienne qui est utilisée dans pySpark. Incorporer DTW ?
        kmeanSpark = KMeansSpark(featuresCol='time_series', k=2, seed=random_state)
        modelSpark = kmeanSpark.fit(dataDf)
        LOGGER.info(" --- Finished fitting K-Means to data --- ")
        # --------------------------------
        # 3 - Display the labels clustered
        # --------------------------------
        LOGGER.info(" --- Exporting results to suitable format ---")
        transformed = modelSpark.transform(dataDf).select('time_series', 'prediction')
        result = transformed.collect()
        return modelSpark, result
    except Exception:
        msg = "UNEXPECTED ERROR in fit_kmeans_spark_internal(..., {}, {}, {})"
        raise IkatsException(msg.format(ts_list, n_cluster, random_state))
    finally:
        SSessionManager.stop()


# TODO : 2 - TRANSFORMER les résultats.
# TODO : Améliorer cette étape avec t-SNE (voir marque-page). Sparkisation ?
# TODO :  Incorporation au code de kmeans pour ne pas répéter certaines parties
def mds_representation_kmeans(fitted_model, ts_list, random_state_mds=None):
    """
    Compute the MultiDimensional Scaling (MDS) transformation to the K-Means results.
    Purpose: a two dimensional representation of the clustering.

    :param fit_model: The K-Means fitted model.
    :type fit_model: sklearn.cluster.k_means_.KMeans

    :param data: The initial data-set (ex: the paa obtained after *back_transform_sax*
    :type data: table

    :param random_state_mds: The seed used by the random number generator (if int).
    If None, the random number generator is the RandomState instance used by np.random.
    :type random_state_mds: int or NoneType

    .. note:: specify `random_state_mds` to make the results reproducible.

    :return tuple of results :
        mds: multidimensional scaling algorithm result (2 dimensional representation for the visualisation)
        pos: the position (x,y) of the initial data-set after an mds transformation
        centers_pos: the position (x,y) of the centroids after an mds transformation
    :rtype : sklearn.manifold.mds.MDS, numpy array, numpy array
    """
    LOGGER.info("Starting MultiDimensional Scaling (MDS) transformation ...")
    try:
        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        data = []
        for i in ts_list:
            startLoadingTime = time.time()
            # Read TS from it's TSUID; shape = (2, nrow)
            ts_data = IkatsApi.ts.read([i['tsuid']])[0]
            LOGGER.debug("TSUID: %s, Gathering time: %.3f seconds", i, time.time() - startLoadingTime)
            lisTS = [i[1] for i in ts_data]
            data.append(lisTS)
        datArray = np.array(data)
        # Addition of the centroids
        datArrayCentroids = np.concatenate((datArray, fitted_model.cluster_centers_))
        # ----------------------------------------------------------------
        # 2 - Compute the matrice with Euclidian distances between each TS
        # ----------------------------------------------------------------
        matDist = euclidean_distances(datArrayCentroids)
        # ----------------------------------------------------------------
        # 3 - Compute the Multidimensional scaling (MDS)
        # ----------------------------------------------------------------
        # TODO : Voir la pertinence de ces paramètres (et des autres), surtout dissimilarity
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state_mds)
        position = mds.fit_transform(matDist)
        a = len(position)
        # -----------------------------------------------------------
        # 4 - Separate the result for the TS-centroids and for the TS
        # -----------------------------------------------------------
        # Positions of the centroids are in the *n_clusters* last lines of the table
        centroidsPosition = position[range(a - fitted_model.n_clusters, a)]
        # Points position of each point of the initial data-set (without the position of the centroids)
        pointsPosition = position[range(0, a - fitted_model.n_clusters)]
        LOGGER.info("   ... finished MDS transformation")
        return mds, pointsPosition, centroidsPosition
    except Exception:
        msg = "Unexpected error: mds_representation_kmeans(..., {}, {}, {})"
        raise IkatsException(msg.format(fitted_model, ts_list, random_state_mds))


# TODO : 3 - FORMATER les résultats
def format_kmeans(centers_pos, pos, tsuid_list, model):
    """
    Build the output of the algorithm (dict) according to the catalog.

    :param centers_pos: The position (x,y) of all the centroids after an mds transformation (list of pairs).
    :type centers_pos: list

    :param pos: The position of all the points (x, y) of the data-set after an mds transformation (n * 2 matrix).
    :type pos: list

    :param model: The K-Means model used.
    :type model: sklearn.cluster.k_means_.KMeans

    :param tsuid_list: List of all the TSUID
    :type tsuid_list: list

    :return: dict formatted as awaited (tsuid, mds new coords of each point, centroid coords)
    :rtype: dict
    """
    LOGGER.info("Exporting results to the K-Means format ...")

    # Example of the dict *result*:
    # {
    #   "C1": {
    #    "centroid": [x,y],
    #    "*tsuid1*": [x,y],
    #    "*tsuid2*": [x,y]
    #   },
    #   "C2" : ...
    # }

    # Initializing result structure
    result = dict()

    # For each cluster
    for center in range(1, model.n_clusters + 1):

        center_label = "C" + str(center)  # "C1", "C2",...

        result[center_label] = {}  # result["C1"] = ...

        # position of the centroid : {"C1": {"centroid": [x,y]}}
        #
        result[center_label]["centroid"] = list(centers_pos[center - 1])

        # position of each point of the data-set : {"C1": {"*tsuid1*": [x,y], ... }}
        #

        # For each points of the data-set:
        for index in range(0, len(tsuid_list)):
            # if the data is in the current cluster
            if model.labels_[index] == center - 1:
                # position of the data : "*tsuid1*": [x,y]
                result[center_label][tsuid_list[index]] = list(pos[index])
    return result

# TODO : Main
def fit_kmeans_on_ts(data, nb_clusters, random_state=None, nb_points_by_chunk=50000, spark=None):
    """
    Performs K-means algorithm on time series either with Spark either with scikit-learn.

    :param data: List of TS to cluster
    :type data: List of dict

    :param nb_clusters: Number of clusters of the Kmeans
    :type nb_clusters: int

    :param random_state: Used for results reproducibility
    :type random_state: int

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :param spark: Flag indicating if Spark usage is:
        * forced (case True),
        * forced to be not used (case False)
        * case None: Spark usage is checked (function of amount of data)
    :type spark: bool or NoneType

    :return tuple of results : model, km
        model: The K-Means model used.
        km: results summarized into a dict (see format_kmeans() ).
    """
    # -----------------
    # 1 - Fit the model
    # -----------------
    # Arg `spark=True`: spark usage forced
    # Arg `spark=None`: Check using criteria (nb_points and number of ts)
    if spark is True or (spark is None and SparkUtils.check_spark_usage(tsuid_list=tsuid_list,
                                                                        nb_ts_criteria=100,
                                                                        nb_points_by_chunk=nb_points_by_chunk)):
        # kmeans_spark_internal(ts_list=tsuid_list, nb_clusters=nb_clusters, nb_points_by_chunk=nb_points_by_chunk)
        model_spark = fit_kmeans_spark(data, n_cluster, random_state=None)
        return model_spark
    else:
        model_sklearn = fit_kmeans_sklearn_internal(data, nb_clusters, random_state=None)
        return model_sklearn

    # ------------------------------------
    # 2 - Compute the MDS (Multidimensional scaling) (purpose : 2 dimensional visualisation)
    # Note that the seed (random_state_mds) is the same
    _, pos, centers_pos = mds_representation_kmeans(fit_model=model, data=paa,
                                                    random_state_mds=random_state)
    # -----------------------
    # 3 - Prepare the outputs
    # -----------------------
    k_means = format_kmeans(centers_pos=centers_pos, pos=pos, tsuid_list=list(paa.keys()), model=model)
    return model, k_means
