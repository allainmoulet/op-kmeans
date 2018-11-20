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
import time
import numpy as np
import logging

from ikats.core.library.spark import SSessionManager, SparkUtils
from ikats.core.resource.api import IkatsApi
from ikats.core.library.exception import IkatsException

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

from pyspark.ml.clustering import KMeans as KMeansSpark
from pyspark.ml.linalg import Vectors

LOGGER = logging.getLogger(__name__)


def fit_kmeans_sklearn_internal(ts_list, n_cluster, random_state_kmeans=None):
    """
    The internal wrapper to fit K-means on time series with scikit-learn.

    :param ts_list: the time series to cluster
    :type ts_list: list of dicts

    :param n_cluster: the number of clusters to form
    :type n_cluster: int

    :param random_state_kmeans: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_kmeans: int or NoneType

    :return a tuple of those 2 elements:
        model: The KMeans model fitted on the input 'ts_list'
        data: Only the values of the time series extracted from ts_list
    :rtype
        model: sklearn.cluster.k_means_.KMeans
        data: numpy.ndarray
    """
    LOGGER.info(" --- Starting K-Means fit with scikit-learn --- ")
    try:
        start_loading_time = time.time()
        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        data = []
        for i in ts_list:
            # Read TS from it's TSUID; shape = (2, nrow)
            ts_data = IkatsApi.ts.read([i['tsuid']])[0]
            list_ts = [i[1] for i in ts_data]
            data.append(list_ts)
        data = np.array(data)
        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        model = KMeans(n_clusters=n_cluster, random_state=random_state_kmeans)
        model.fit(data)
        LOGGER.debug(" --- Finished fitting K-Means to data in: %.3f seconds --- ", time.time() - start_loading_time)
        return model, data
    except Exception:
        msg = "Unexpected error: fit_kmeans_sklearn_internal(..., {}, {}, {})"
        raise IkatsException(msg.format(ts_list, n_cluster, random_state_kmeans))


# TODO: 1bis - AJUSTER le modèle - Partie Spark
def fit_kmeans_spark_internal(ts_list, n_cluster, random_state_kmeans=None):
    """
    The internal wrapper to fit K-means on time series with Spark

    :param ts_list: the time series to cluster
    :type ts_list: list of dicts

    :param n_cluster: the number of clusters to form
    :type n_cluster: int

    :param random_state_kmeans: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_kmeans: int or NoneType

    :return a tuple of those 3 elements:
        model: The KMeans model fitted on the input 'ts_list'
        data: Only the values of the time series extracted from ts_list
        result: list of Row Spark objects with predictions

    :rtype
        model: pyspark.ml.clustering.KMeansModel
        data: numpy.ndarray
        result: list of pyspark.sql.types.Row
    """
    SSessionManager.get()
    spark = SSessionManager.spark_session
    spark.sparkContext.setLogLevel('WARN')
    LOGGER.info("--- Starting K-Means fit with Spark ---")
    try:
        start_loading_time = time.time()
        # -----------------------------------------------------------
        # 1 - Process the data in the shape needed by Spark's K-Means
        # -----------------------------------------------------------
        # Perform operation iteratively on each TS
        data_spark = []
        # Build of data for
        data = []
        for i in ts_list:
            # Read TS from it's TSUID; shape = (2, nrow)
            ts_data = IkatsApi.ts.read([i['tsuid']])[0]
            list_ts = [x[1] for x in ts_data]
            vec_ts = Vectors.dense(list_ts)
            data_spark.append((vec_ts,))
            data.append(list_ts)
        data_df = spark.createDataFrame(data_spark, ['time_series'])
        data = np.array(data)
        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        kmeans_spark = KMeansSpark(featuresCol='time_series', k=2, seed=random_state_kmeans)
        model_spark = kmeans_spark.fit(data_df)
        LOGGER.info("--- Finished fitting K-Means to data ---")
        # --------------------------------
        # 3 - Display the labels clustered
        # --------------------------------
        LOGGER.info("--- Exporting results to suitable format ---")
        transformed = model_spark.transform(data_df).select('time_series', 'prediction')
        result = transformed.collect()
        LOGGER.info("--- Finished to export the results ---")
        LOGGER.debug(" --- Finished fitting K-Means to data in: %.3f seconds --- ", time.time() - start_loading_time)
        return model_spark, data, result
    except Exception:
        msg = "UNEXPECTED ERROR in fit_kmeans_spark_internal(..., {}, {}, {})"
        raise IkatsException(msg.format(ts_list, n_cluster, random_state_kmeans))
    finally:
        SSessionManager.stop()


# TODO : Améliorer cette étape avec t-SNE (voir marque-page)
# TODO: Sparkisation ?
def mds_representation_kmeans(fitted_model, data, random_state_mds=None):
    """
    Compute the MultiDimensional Scaling (MDS) transformation to the K-Means results.
    Purpose: a two dimensional representation of the clustering.

    :param fitted_model: The K-Means fitted model with scikit-learn or Spark
    :type fitted_model: sklearn.cluster.k_means_.KMeans OR pyspark.ml.clustering.KMeansModel

    :param data: the time series returned by the K-Means step
    :type data: numpy.ndarray

    :param random_state_mds: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_mds: int or NoneType

    :return
        mds: The result of the MultiDimensional Scaling algorithm in 2 dimensions for the visualisation
        pointsPosition: The position (x, y) of the initial dataset after the MDS transformation
        centroidsPosition: The position (x, y) of the centroids after the MDS transformation
    :rtype tuple with
        mds: sklearn.manifold.mds.MDS,
        pointsPosition: numpy.ndarray
        centroidsPosition: numpy.ndarray
    """
    LOGGER.info("--- Starting MultiDimensional Scaling (MDS) transformation ---")
    try:
        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        if fitted_model.__module__ == 'pyspark.ml.clustering':
            centroids = np.array(fitted_model.clusterCenters())
        elif fitted_model.__module__ == 'sklearn.cluster.k_means_':
            centroids = fitted_model.cluster_centers_
        nb_clusters = len(centroids)

        # Addition of the centroids
        data_array_centroids = np.concatenate((data, centroids))
        # ----------------------------------------------------------------
        # 2 - Compute the matrice with Euclidian distances between each TS
        # ----------------------------------------------------------------
        mat_dist = euclidean_distances(data_array_centroids)
        # ----------------------------------------------------------------
        # 3 - Compute the Multidimensional scaling (MDS)
        # ----------------------------------------------------------------
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state_mds)
        position = mds.fit_transform(mat_dist)
        a = len(position)
        # -----------------------------------------------------------
        # 4 - Separate the result for the TS-centroids and for the TS
        # -----------------------------------------------------------
        # Positions of the centroids are in the *n_clusters* last lines of the table
        centroids_position = position[range(a - nb_clusters, a)]
        # Points position of each point of the initial data-set (without the position of the centroids)
        points_position = position[range(a - nb_clusters)]
        LOGGER.info("--- Finished MDS transformation ---")
        return mds, points_position, centroids_position
    except Exception:
        msg = "Unexpected error: mds_representation_kmeans(..., {}, {}, {})"
        raise IkatsException(msg.format(fitted_model, data, random_state_mds))


def format_kmeans(result_mds, ts_list, result_kmeans):
    """
    Build the output of the algorithm (dict) according to the catalog.

    :param result_mds: The result obtained after the MDS transformation with:
        the MDS model used
        the position (x, y) of all the centroids and after the MDS transformation
        the position (x, y) of all the points after the MDS transformation
    :type result_mds: tuple

    :param ts_list: List of all the TSUID
    :type ts_list: list of dicts

    :param result_kmeans: The result obtained at the K-Means step
    :type result_kmeans: a tuple with those 3 elements:
        model: sklearn.cluster.k_means_.KMeans OR pyspark.ml.clustering.KMeansModel
        data: numpy.ndarray
        result: list of pyspark.sql.types.Row

    :return: dict formatted as shown below, with: tsuid, mds new coordinates of each point, mds centroid coordinates
    :rtype: dict of dicts
    """
    # Example of obtained result:
    # {
    #   'C1': {
    #    'centroid': [x, y],
    #    '*tsuid1*': [x, y],
    #    '*tsuid2*': [x, y]
    #   },
    #   'C2': {
    #   ...
    #   }
    # }
    LOGGER.info("--- Exporting results to the wanted format ---")
    try:
        if result_kmeans[0].__module__ == 'pyspark.ml.clustering':
            # Get the predictions in a list
            pred_list = []
            for i in result_kmeans[2]:
                pred_list.append(i['prediction'])
                predictions = np.array(pred_list, dtype=np.int32)
        elif result_kmeans[0].__module__ == 'sklearn.cluster.k_means_':
            predictions = result_kmeans[0].labels_

        nb_clusters = result_kmeans[1].shape[1]
        pos = result_mds[1]
        centroids_pos = result_mds[2]
        result = {}
        # For each cluster
        for i in range(nb_clusters):
            center_label = 'C' + str(i + 1)
            result[center_label] = {}
            # Position of the centroid: 'centroid': [x, y]
            result[center_label]['centroid'] = list(centroids_pos[i])
            # For each time serie of the input data
            for j in range(len(ts_list)):
                # If the point is in the current cluster
                if predictions[j] == i:
                    # Position of this point: 'tsuid': [x, y]
                    result[center_label][ts_list[j]['tsuid']] = list(pos[j])
        LOGGER.info("--- Finished to import the results ---")
        return result
    except Exception:
        msg = "Unexpected error: format_kmeans(result_mds, ts_list, result_kmeans)(..., {}, {}, {})"
        raise IkatsException(msg.format(result_mds, ts_list, result_kmeans))


def fit_kmeans_on_ts(ts_list, nb_clusters, random_state=None, nb_points_by_chunk=50000, spark=None):
    """
    Performs K-means algorithm on time series either with Spark either with scikit-learn.

    :param ts_list: The time series to be clustered
    :type ts_list: list of dicts

    :param nb_clusters: The number of clusters of the K-means model
    :type nb_clusters: int

    :param random_state: Used for results reproducibility
    :type random_state: int

    :param nb_points_by_chunk: The size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :param spark: Flag indicating if Spark usage is:
        * forced (case True),
        * forced to be not used (case False)
        * case None: Spark usage is checked (function of amount of data)
    :type spark: bool or NoneType

    :return
        kmeans: The K-Means model used
        result: The obtained clusters with centroids and time series
    :rtype
        kmeans: Depends if Spark or scikit-learn has been used : sklearn.cluster.k_means_.KMeans
            or pyspark.ml.clustering.KMeansModel
        result: dict of dicts
    """
    # --------------------
    # 0 - Check the inputs
    # --------------------
    # Argument `ts_list`
    if type(ts_list) is not list:
        raise TypeError("TYPE ERROR: type of argument `ts_list` is {}, expected 'list'".format(type(ts_list)))
    elif not ts_list:
        raise ValueError("VALUE ERROR: argument `ts_list` is empty")
    # Argument `nb_clusters`
    if type(nb_clusters) is not int:
        raise TypeError("TYPE ERROR: type of argument `nb_clusters` is {}, expected 'int'".format(type(nb_clusters)))
    elif not nb_clusters or nb_clusters < 1:
        raise ValueError("VALUE ERROR: argument `nb_clusters` must be an integer greater than 1")
    # Argument `random_state`
    if type(random_state) is not int and random_state is not None:
        raise TypeError("TYPE ERROR: type of argument `random_state` is {}, expected 'int' or 'NoneType'".format(type(random_state)))
    elif type(random_state) is int and random_state < 0:
        raise ValueError("VALUE ERROR: argument `random_state` must be a positive integer")
    # Argument `nb points by chunk`
    if type(nb_points_by_chunk) is not int:
        raise TypeError("TYPE ERROR: type of argument `nb_points_by_chunk` is {}, expected 'int'".format(nb_points_by_chunk))
    elif not nb_points_by_chunk or nb_points_by_chunk < 0:
        raise ValueError("VALUE ERROR: argument `nb_points_by_chunk` must be an integer greater than 0")
    # Argument `spark`
    if type(spark) is not bool and spark is not None:
        raise TypeError("TYPE ERROR: type of argument `spark` is {}, expected `bool` or `NoneType`".format(type(spark)))
    # -----------------
    # 1 - Fit the model
    # -----------------
    # Arg `spark=True`: spark usage forced
    # Arg `spark=None`: Check using criteria (nb_points and number of ts)
    if spark or (spark is None and SparkUtils.check_spark_usage(tsuid_list=ts_list, nb_ts_criteria=100,
                                                                nb_points_by_chunk=nb_points_by_chunk)):
        res_kmeans = fit_kmeans_spark_internal(ts_list=ts_list, n_cluster=nb_clusters, random_state_kmeans=random_state)
    else:
        res_kmeans = fit_kmeans_sklearn_internal(ts_list=ts_list,
                                                 n_cluster=nb_clusters,
                                                 random_state_kmeans=random_state)
    # ------------------------------------
    # 2 - Compute the MDS (Multidimensional scaling) (purpose : 2 dimensional visualisation)
    # Note that the seed (random_state_mds) is the same
    res_mds = mds_representation_kmeans(fitted_model=res_kmeans[0], data=res_kmeans[1], random_state_mds=random_state)
    # -----------------------
    # 3 - Prepare the outputs
    # -----------------------
    result = format_kmeans(result_mds=res_mds, ts_list=ts_list, result_kmeans=res_kmeans)
    return res_kmeans[0], result
