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

from sklearn.cluster import KMeans
# Spark utils
from ikats.core.library.spark import SSessionManager, SparkUtils


def kmeans_on_ts(ts_list, nb_clusters, nb_points_by_chunk=50000, spark=None):
    """
    Performs K-means algorithm on time series either with Spark either with scikit-learn.

    :param ts_list: List of TS to cluster
    :type ts_list: List of dict

    :param nb_clusters: Number of clusters of the Kmeans
    :type nb_clusters: int

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
    :rtype : sklearn.cluster.k_means_.KMeans, dict
    """
    # ts_list (non empty list)
    if type(ts_list) is not list:
        raise TypeError("Arg. type `ts_list` is {}, expected `list`".format(type(ts_list)))
    elif not ts_list:
        raise ValueError("`ts_list` provided is empty !")

    try:
        # Extract TSUID from ts_list
        tsuid_list = [x['tsuid'] for x in ts_list]
    except Exception:
        raise ValueError("Impossible to get tsuid list...")

    # Number of clusters (int > 0)
    if type(nb_clusters) is not int or nb_clusters < 0:
        raise TypeError("Arg. type `nb_clusters` is {}, expected positive int".format(type(nb_clusters)))
    elif not nb_clusters:
        raise ValueError("`nb_clusters` provided is empty !")

    # Nb points by chunk (int > 0)
    if type(nb_points_by_chunk) is not int or nb_points_by_chunk < 0:
        raise TypeError("Arg. `nb_points_by_chunk` must be an integer > 0, get {}".format(nb_points_by_chunk))

    # spark (bool or None)
    if type(spark) is not bool and spark is not None:
        raise TypeError("Arg. type `spark` is {}, expected `bool` or `NoneType`".format(type(spark)))

    # 1/ Check for spark usage and run
    # ----------------------------------------------------------
    if spark is True or (spark is None and SparkUtils.check_spark_usage(tsuid_list=tsuid_list,
                                                                        nb_ts_criteria=100,
                                                                        nb_points_by_chunk=nb_points_by_chunk)):
    # Arg `spark=True`: spark usage forced
    # Arg `spark=None`: Check using criteria (nb_points and number of ts)
        return kmeans_spark(ts_list=tsuid_list, nb_clusters=nb_clusters, nb_points_by_chunk=nb_points_by_chunk)
    else:
        return kmeans_sklearn(ts_list=tsuid_list, nb_clusters=nb_clusters)


def kmeans_spark(ts_list, nb_clusters, nb_points_by_chunk):
    """
    Performs K-means algorithm on time series with Spark.

    :param ts_list: List of TS to cluster
    :type ts_list: List of dict

    :param nb_clusters: Number of clusters of the Kmeans
    :type nb_clusters: int

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :return tuple of results : model, km
        * model: The K-Means model used.
        * km: results summarized into a dict (see format_kmeans()).
    # TODO: trouver le type de sortie
    :rtype : *********, dict
    """
    pass

def kmeans_sklearn(ts_list, nb_clusters):
    """
    Performs K-means algorithm on time series with sklearn.

    :param ts_list: List of TS to cluster
    :type ts_list: List of dict

    :param nb_clusters: Number of clusters of the Kmeans
    :type nb_clusters: int

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :return tuple of results : model, km
        * model: The K-Means model used.
        * km: results summarized into a dict (see format_kmeans()).
    # TODO: trouver le type de sortie
    :rtype : *********, dict
    """
    pass
