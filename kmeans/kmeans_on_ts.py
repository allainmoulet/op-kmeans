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
import pandas as pd
import logging

from ikats.core.library.spark import SSessionManager, SparkUtils
from ikats.core.resource.api import IkatsApi

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.clustering import KMeans as KMeansSpark
from pyspark.ml.linalg import Vectors

LOGGER = logging.getLogger(__name__)

# TODO: TS doivent être alignées
# TODO: `period` doit être dans les md de chaque TS


def _check_alignement(tsuid_list):
    """
    Check the alignment of the provided list of TS (`tsuid_list`): same `start_date`, `end_date`, and `period`.
    Operator `quality_stat` shall be launch on ts_list before !

    :param tsuid_list: List of tsuid of TS to check
    :type tsuid_list: List of str
    ..Example: ['tsuid1', 'tsuid2', ...]

    return: Tuple composed by:
        * start date (int)
        * end date (int)

    :raises:
        * ValueError: TS are not aligned
        * ValueError: Some metadata are missing (start date, end date, nb points)

    ..Note: First TS is the reference. Indeed, ALL TS must be aligned (so aligned to the first TS)
    """
    # Read metadata
    meta_list = IkatsApi.md.read(tsuid_list)

    # Perform operation iteratively on each TS
    for tsuid in tsuid_list:

        # 1/ Retrieve meta data and check available meta-data
        # --------------------------------------------------------------------------
        md = meta_list[tsuid]

        # CASE 1: no md (sd, ed, nb_point) -> raise ValueError
        if 'ikats_start_date' not in md.keys() and 'ikats_end_date' not in md.keys():
            raise ValueError("No MetaData (start / end date) associated with tsuid {}... Is it an existing TS ?".format(tsuid))
        # CASE 2: metadata `period` not available -> raise ValueError
        elif 'qual_ref_period' not in md.keys():
            raise ValueError("No MetaData `qual_ref_period` with tsuid {}... Please launch `quality indicator`".format(tsuid))
        # CASE 3: OK (metadata `period` available...) -> continue
        else:
            period = int(float(md['qual_ref_period']))
            sd = int(md['ikats_start_date'])
            ed = int(md['ikats_end_date'])

        # 2/ Check if data are aligned (same sd, ed, period)
        # --------------------------------------------------------------------------
        # CASE 1: First TS -> get as reference
        if tsuid == tsuid_list[0]:
            ref_sd = sd
            ref_ed = ed
            ref_period = period

        # CASE 2: Other TS -> compared to the reference
        else:
            # Compare `sd`
            if sd != ref_sd:
                raise ValueError("TS {}, metadata `start_date` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, sd, ref_sd))
            # Compare `ed`
            elif ed != ref_ed:
                raise ValueError("TS {}, metadata `end_date` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, ed, ref_ed))
            # Compare `period`
            elif period != ref_period:
                raise ValueError("TS {}, metadata `ref_period` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, ed, ref_period))

    return ref_sd, ref_ed


def fit_kmeans_sklearn_internal(tsuid_list, n_cluster, random_state_kmeans=None):
    """
    The internal wrapper to fit K-means on time series with scikit-learn.

    :param tsuid_list: List of tsuid to use
    :type tsuid_list: list of str

    :param n_cluster: the number of clusters to form
    :type n_cluster: int

    :param random_state_kmeans: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_kmeans: int or NoneType

    :return a tuple of those 2 elements:
        * model_sklearn: The KMeans model fitted on the input 'ts_list'
        * result_sklearn: Data frame with indexes, and columns CLUSTER_ID, TSUID, t_0, ..., t_n for the values of time
        series
    :rtype
        * model_sklearn: sklearn.cluster.k_means_.KMeans
        * result_sklearn: pandas.core.frame.DataFrame

    """
    LOGGER.info(" --- Starting K-Means fit with scikit-learn --- ")
    try:
        start_loading_time = time.time()
        # -------------------------------------------------------------
        # 0 - Check TS alignment (ValueError instead)
        # -------------------------------------------------------------
        _check_alignement(tsuid_list)

        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        # Extract data (just data values, not timestamps)
        data_sklearn = np.array(IkatsApi.ts.read(tsuid_list))[:, :, 1]
        # Shape = (n_ts, n_times)

        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        model_sklearn = KMeans(n_clusters=n_cluster, random_state=random_state_kmeans)
        model_sklearn.fit(data_sklearn)

        # Retrieve centroids
        centroids_sklearn = model_sklearn.cluster_centers_
        # shape = (n_cluster, n_times)

        # Retrieve cluster_id for each TS
        cluster_id = model_sklearn.labels_
        # shape = (n_ts,)

        # ---------------------
        # 3 - Reformat data into Pandas Dataframe
        # ---------------------
        # VALUES
        data_df = pd.DataFrame(data_sklearn)

        # Rename columns (1 col = 1 times)
        n_times = data_sklearn.shape[1]
        data_df.columns = ["t_" + str(i) for i in range(n_times)]
        # Example: data-df =
        #    t_0  t_1  ...
        # 0   7   3    ...
        # ...

        # CLUSTER ID
        cluster_df = pd.DataFrame({'TSUID': tsuid_list, 'CLUSTER_ID': cluster_id})
        # Example:
        #    CLUSTER_ID                                      TSUID
        # 0        0  2630EF00000100076C0000020006E0000003000771
        # ...

        # CENTROIDS
        centroids_df = pd.DataFrame(centroids_sklearn, columns=["t_" + str(i) for i in range(n_times)])
        # Add the columns TSUID and CLUSTER_IDbefore concatenation with centroids_df
        temp = pd.DataFrame({'TSUID': ['C' + str(i) for i in range(1, n_cluster + 1)], 'CLUSTER_ID': range(n_cluster)})
        # Column "TSUID" contains "C1", ..., "C{n_cluster}"

        centroids_df = pd.concat([temp, centroids_df], axis=1)
        # Example:
        #    CLUSTER_IDTSUID   t_0  ... t_n
        # 0        0   C1    8.0  ... 3.5
        # 1        1   C2   13.0  ... 16.0
        # ...

        # ---------------------
        # 4 - Concatenate all results into single DF
        # ---------------------
        # Concatenate these DF by columns
        result_sklearn = pd.concat([cluster_df, data_df], axis=1)
        # Example: result_sklearn =
        #    CLUSTER_ID                                      TSUID t_0 t_1
        # 0        0  2630EF00000100076C0000020006E0000003000771   7   3
        # ...

        # Add centroids at the end of the dataframe and reset of indexes
        result_sklearn = pd.concat([result_sklearn, centroids_df], ignore_index=True)
        # Example:
        #    CLUSTER_ID                                      TSUID t_0  t_1 ...
        # 0        0  2630EF00000100076C0000020006E0000003000771   7    3
        # 1        0  14ED6B00000100076C0000020006E0000003000772   9    4
        # 2        1  B9C00C00000100076C0000020006E0000003000773  14   15
        # 3        1  5F2C9B00000100076C0000020006E0000003000774  12   17
        # 4        0                                          C1   8  3.5
        # 5        1                                          C2  13   16 ...
        # Two last lines: the 2 centroids
        LOGGER.debug(" --- Finished fitting K-Means to data in: %.3f seconds --- ", time.time() - start_loading_time)
        # For now, do not return model
        return result_sklearn  # model_sklearn
    finally:
        LOGGER.info("--- Finished to run fit_kmeans_spark_internal() function ---")


def fit_kmeans_spark_internal(tsuid_list, n_cluster, nb_ts_by_chunks, random_state_kmeans=None):
    """
    The internal wrapper to fit K-means on time series with Spark

    :param tsuid_list: List of tsuid to use
    :type tsuid_list: list of str

    :param n_cluster: the number of clusters to form
    :type n_cluster: int

    :param nb_ts_by_chunks: size of chunks in number of ts (assuming time series is periodic and without holes)
    :type nb_ts_chunks: int

    :param random_state_kmeans: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_kmeans: int or NoneType

    :return a tuple of those 3 elements:
        model: The KMeans model fitted on the input 'ts_list'
        data_df_spark: Data frame with indexes, and columns TSUID, VALUES and CLUSTER
        centroids_spark: The centroids of each class

    :rtype data_df_spark: pandas.core.frame.DataFrame

    """
    LOGGER.info("--- Starting K-Means fit with Spark ---")
    # -------------------------------------------------------------
    # 0 - Check TS alignment (ValueError instead)
    # -------------------------------------------------------------
    # All TS are aligned, get the start / end date of ref
    sd, ed = _check_alignement(tsuid_list)

    SSessionManager.get()

    try:
        start_loading_time = time.time()
        # retrieve spark context
        sc = SSessionManager.get_context()

        # Get the number of partitions = n_ts / nb_ts_by_chunks
        nb_chunks = max(1, int(len(tsuid_list) / nb_ts_by_chunks))

        # Distribute the TS names (tsuid)
        rdd_ts = sc.parallelize(tsuid_list, nb_chunks)

        # 1/ Read entire TS, and put each TS into one DenseVector (1 TS per Vector)
        def __map_extract(chunked_tsuid_list):
            """
            Extract data corresponding to `tsuid`, and return formatted result.
            :param chunked_tsuid_list: The current tsuid_list of the ts to extract (str) in the current chunk
            :return: tuple composed by:
                * input `tsuid`
                * DenseVector containing all data of `tsuid` (not timestamps)
            """
            # Extract values (not timestamps), and transform into list (avoid error with spark vs numpy)
            # Extract data
            ts_values = IkatsApi.ts.read(chunked_tsuid_list, sd=sd, ed=ed)
            # Shape = (n_ts, n_times, 2) = (~ n_ts_by_chunk, n_times, 2)

            # Get the DATA only
            ts_values = np.array(ts_values)[:, :, 1]
            # Shape = (n_ts, n_time)

            # Transform into (Spark do not accept np.arrays !)
            ts_values = ts_values.tolist()
            # len = n_ts

            # Build result: put into tuple each ts result (tsuid, values)
            result = [(chunked_tsuid_list[i], Vectors.dense(ts_values[i])) for i in range(len(chunked_tsuid_list))]
            # Result = [(tsuid1, Vectors.dense(ts_value1) ), ...]
            return result

        # DESCRIPTION: Extract TS
        # INPUT  : tsuid list into an rdd
        # OUTPUT : rdd containing [('tsuid', DenseVector([TS_VALUES]) ), ...]
        rdd_ts_values = rdd_ts.mapPartitions(lambda x: __map_extract(list(x)))
        # To limit the call `IkatsApi.read`, we apply the function on a list of TSUID
        # To access to a list of TSUID, we use the mapPartitions (1 partition = list of `nb_ts_by_chunks` tsuid)
        # In mapPartitions, `x` is an `iterator` (-> list(x) is the true list of TSUID)

        # DESCRIPTION: Transform into DataFrame
        # INPUT  : rdd containing [('tsuid', DenseVector([TS_VALUES]) ), ...]
        # OUTPUT : Dataframe containing [TSUID: string, VALUES: vector]
        df_spark = rdd_ts_values.toDF(['TSUID', 'VALUES'])

        # ---------------------
        # 2 - Fit the algorithm
        # ---------------------
        # Init model for performing Kmeans
        # input column: 'VALUES'; output column = "CLUSTER_ID"
        kmeans_spark = KMeansSpark(featuresCol='VALUES',predictionCol="CLUSTER_ID",
                                   k=n_cluster, seed=random_state_kmeans)
        # Fit model with data transformed
        model_spark = kmeans_spark.fit(df_spark)
        LOGGER.info("--- Finished fitting K-Means to data ---")
        # --------------------------------
        # 3 - Cluster DATA
        # --------------------------------
        LOGGER.info("--- Exporting results to suitable format ---")
        # DESCRIPTION: Cluster initial DataFrame
        # INPUT  : Dataframe containing [TSUID: string, VALUES: vector] (col VALUES contains ts data per row)
        # OUTPUT : Dataframe containing [TSUID: string, VALUES: vector, predic]
        transformed = model_spark.transform(df_spark)

        # Example:
        # +--------------------+--------------------+----------+
        # |               TSUID|              VALUES|CLUSTER_ID|
        # +--------------------+--------------------+----------+
        # |D73FA500000100000...|[0.05188564211130...|         0|
        # ...

        # --------------------------
        # 4 - Collect the clustering and format data into pandas DatafFrame
        # --------------------------
        # Retrieve cluster centers
        centroids_spark = pd.DataFrame(model_spark.clusterCenters())
        # Example:
        #          0         1         2         3         4         5         6   ...
        # 0  0.074564 -0.056785 -0.012004 -0.048975 -0.019338 -0.024185  0.027072  ...
        # ...

        # Retrieve the nb of times (= number of col)
        n_times = len(centroids_spark.columns.values)

        # Rename columns into 't_{time}'
        centroids_spark.columns = ["t_" + str(i) for i in range(n_times)]

        # Add the columns TSUID and CLUSTER_ID before concatenation with centroids_df
        temp = pd.DataFrame({'TSUID': ['C' + str(i) for i in range(1, n_cluster + 1)], 'CLUSTER_ID': range(n_cluster)})
        # Column "TSUID" contains "C1", ..., "C{n_cluster}"
        # Example:
        #    CLUSTER_ID TSUID
        # 0           0    C1
        # 1           1    C2

        # Concatenate at left the created columns 'TSUID' and 'CLUSTER_ID'
        centroids_df = pd.concat([temp, centroids_spark], axis=1)
        # Example:
        #    CLUSTER_ID TSUID   t_0  ... t_n
        # 0        0    C1      8.0  ... 3.5
        # 1        1    C2     13.0  ... 16.0
        # ...

        # --------------------------
        # 5 - Collect clustered data and format data into pandas DataFrame
        # --------------------------
        # A function that transform `vector` into `List` of double
        vector_to_list = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

        # DESCRIPTION : Transorm column containing result (type Vector) into multiple columns
        # INPUT  : A DataFrame with result columns ["Timestamp", _INPUT_COL, _OUTPUT_COL]: int, Vector, Vector
        # OUTPUT : Same DF with muliple columns (one per PC):  ["Timestamp", _INPUT_COL, _OUTPUT_COL, PC1, ..., PC{k}]
        result = transformed.withColumn("t_", vector_to_list(col("VALUES")))\
            .select(["TSUID", "CLUSTER_ID"] + [col("t_")[i] for i in range(n_times)])
        # Example:
        # +--------------------+----------+--------------------+--------------------+--------------------+
        # |               TSUID|CLUSTER_ID|               t_[0]|               t_[1]|               t_[2]| ...
        # +--------------------+----------+--------------------+--------------------+--------------------+
        # |D73FA500000100000...|         0| 0.05188564211130142|0.006946070585399866|-0.05263324826955795| ...
        # ...

        # Collect result and store into pandas DF
        result = result.toPandas()
        # Example:
        #                                          TSUID  CLUSTER_ID     t_[0]  ...
        # 0   D73FA5000001000001000002000002000003000009           0  0.051886  ...
        # ...

        # Rename columns
        result.columns = ['TSUID', 'CLUSTER_ID'] + ["t_" + str(i) for i in range(n_times)]
        # Example:
        #                                          TSUID  CLUSTER_ID     t_0  ...
        # 0   D73FA5000001000001000002000002000003000009           0  0.051886  ...
        # ...

        # ---------------------
        # 6 - Concatenate all results into single DF
        # ---------------------
        # Add centroids at the end of the dataframe and reset of indexes
        result_spark = pd.concat([result, centroids_df], ignore_index=True)
        # Example:
        #    CLUSTER_ID                                      TSUID t_0  t_1 ...
        # 0        0  2630EF00000100076C0000020006E0000003000771   7    3
        # 1        0  14ED6B00000100076C0000020006E0000003000772   9    4
        # 2        1  B9C00C00000100076C0000020006E0000003000773  14   15
        # 3        1  5F2C9B00000100076C0000020006E0000003000774  12   17
        # 4        0                                          C1   8  3.5
        # 5        1                                          C2  13   16 ...
        # Two last lines: the 2 centroids

        return result_spark
    finally:
        LOGGER.info("--- Finished to run fit_kmeans_spark_internal() function ---")
        SSessionManager.stop()


def mds_representation_kmeans(data, random_state_mds=None):
    """
    Compute the MultiDimensional Scaling (MDS) transformation to the K-Means results.
    Purpose: a two dimensional representation of the clustering.

    :param data:  Result of Kmeans fitting. Data frame with indexes, and columns CLUSTER_ID, TSUID, t_0, ..., t_n
    for the values of time
    :type data: pandas.core.frame.DataFrame

    :param random_state_mds: the seed used by the random number generator (if int) to make the results reproducible
    If None, the random number generator is the RandomState instance used by np.random
    :type random_state_mds: int or NoneType

    :return all_position: The position (x, y) of the initial dataset and of the centroids after the MDS transformation
    :rtype all_position:: pandas.core.frame.DataFrame
    """
    LOGGER.info("--- Starting MultiDimensional Scaling (MDS) transformation ---")

    # Get the values only from dataframe `data` -> np.array
    data_array = data.drop(['CLUSTER_ID', 'TSUID'], axis=1).values

    # ----------------------------------------------------------------
    # 1 - Compute the matrice with Euclidian distances between each TS
    # ----------------------------------------------------------------
    # Compute dissimilarity matrix
    mat_dist = euclidean_distances(data_array)
    # ----------------------------------------------------------------
    # 2 - Compute the Multidimensional scaling (MDS)
    # ----------------------------------------------------------------
    # Compute model
    mds = MDS(n_components=2, random_state=random_state_mds, n_jobs=-1, dissimilarity='precomputed')

    # Compute MDS
    position = mds.fit_transform(mat_dist)
    # Shape = (n_ts + n_cluster, 2) -> 2 = MDS.n_components

    # Retrieve the total number of rows in `data`
    n_row = position.shape[0]  # n_ts + n_cluster

    LOGGER.info("--- Finished MDS transformation ---")

    # -----------------------------------------------------------
    # 3 - Return results as DataFrame
    # -----------------------------------------------------------
    # Change `position` into DF
    all_position = pd.DataFrame(position, columns=['COMP_1', 'COMP_2'])
    # Example:
    #      COMP_1    COMP_2
    # 0  5.278414 -5.459057
    # ...

    # Add columns `CLUSTER_ID` and `TSUID` to sort results
    all_position = pd.concat([data[['CLUSTER_ID', 'TSUID']], all_position], axis=1)
    # Example:
    #    CLUSTER_ID                                       TSUID    COMP_1    COMP_2
    # 0           0  2630EF00000100076C0000020006E0000003000771  5.278414 -5.459057
    # 1           0  14ED6B00000100076C0000020006E0000003000772  4.695608 -3.467254
    # 2           1  B9C00C00000100076C0000020006E0000003000773 -3.868806  4.992268
    # 3           1  5F2C9B00000100076C0000020006E0000003000774 -6.403026  3.706108
    # 4           0                                         C1   5.432918 -4.133372
    # 5           1                                         C2  -5.135109  4.361306

    return all_position


def format_kmeans(all_positions, n_cluster):
    """
    Build the output of the algorithm (dict) according to the catalog.

    :param all_positions: All the point / centroid positions, stored into a DataFrame (columns:
    (CLUSTER_ID, TSUID, COMP_1, COMP_2)
    :type all_positions: pandas.core.frame.DataFrame

    :param n_cluster: the number of clusters to form
    :type n_cluster: int

    :return: dict formatted as shown below, with: tsuid, mds new coordinates of each point, mds centroid coordinates
    :rtype: dict of dicts

    ..Example:
    {
      'C1': {
       'centroid': [x, y],
       '*tsuid1*': [x, y],
       '*tsuid2*': [x, y]
      },
      'C2': {
      ...
      }
    }

    """

    LOGGER.info("--- Exporting results to the wanted format ---")
    n_row = len(all_positions)

    centroids_positions = all_positions.iloc[(n_row - n_cluster):n_row]
    # Example:
    #    CLUSTER_ID TSUID    COMP_1    COMP_2
    # 4           0    C1  5.432918 -4.133372
    # 5           1    C2 -5.135109  4.361306

    points_positions = all_positions.drop(range(n_row - n_cluster, n_row))
    # Example :
    #    CLUSTER_ID                                       TSUID    COMP_1    COMP_2
    # 0           0  2630EF00000100076C0000020006E0000003000771  5.278414 -5.459057
    # 1           0  14ED6B00000100076C0000020006E0000003000772  4.695608 -3.467254
    # 2           1  B9C00C00000100076C0000020006E0000003000773 -3.868806  4.992268
    # 3           1  5F2C9B00000100076C0000020006E0000003000774 -6.403026  3.706108

    #  Init result
    result = {}

    # For each centroid
    for c in centroids_positions['TSUID']:  # "C1", "C2", ...

        # Purpose: CREATE DICT :
        # {'centroid': [x, y],
        #  '*tsuid1*': [x, y],
        #  '*tsuid2*': [x, y]}
        #  Into result['C1']

        result[c] = {}

        # Get the row of `centroids_positions` containing data about the current centroid
        current_centroid = centroids_positions[centroids_positions["TSUID"]==c]
        # Example :
        #    CLUSTER_ID TSUID    COMP_1    COMP_2
        # 4           0    C1  5.432918 -4.133372

        # Retrieve the position [x, y] of the current centroid
        result[c]['centroid'] = current_centroid[['COMP_1', 'COMP_2']].values[0]
        # Example: array([ 5.43291802, -4.13337188])

        # Retrieve current CLUSTER ID
        current_cluster_id = current_centroid['CLUSTER_ID'].values[0]
        # Example: 0

        # Retrieve TS corresponding to `current_cluster_id`
        current_lines = points_positions[points_positions['CLUSTER_ID'] == current_cluster_id]
        # Example:
        #    CLUSTER_ID                                       TSUID    COMP_1    COMP_2
        # 0           0  2630EF00000100076C0000020006E0000003000771  5.278414 -5.459057
        # 1           0  14ED6B00000100076C0000020006E0000003000772  4.695608 -3.467254
        for index in  current_lines.index:  # 0, 1
            # tsuid TO STORE
            current_tsuid = current_lines.loc[index, "TSUID"]
            # Example: "2630EF00000100076C0000020006E0000003000771"

            # [x, y] of the current TS to store
            values = current_lines.loc[index, ['COMP_1', 'COMP_2']].values
            # Example: array([5.278414219776018, -5.459056529052045], dtype=object)

            # UPDATE `result[c]`
            result[c][current_tsuid] = values
            # `tsuid`: [x, y]

    LOGGER.info("--- Finished to import the results ---")
    LOGGER.info("--- Finished to run format_kmeans() ---")
    return result


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
    # Retrieve tsuid list
    try:
        tsuid_list = [x['tsuid'] for x in ts_list]
    except Exception:
        raise ValueError('Impossible to retrieve the tsuid list')

    # Argument `nb_clusters`
    if type(nb_clusters) is not int:
        raise TypeError("TYPE ERROR: type of argument `nb_clusters` is {}, expected 'int'".format(type(nb_clusters)))
    elif not nb_clusters or nb_clusters < 1:
        raise ValueError("VALUE ERROR: argument `nb_clusters` must be an integer greater than 1")
    # Argument `random_state`
    if type(random_state) is not int and random_state is not None:
        raise TypeError("TYPE ERROR: type of argument `random_state` is {}, expected 'int' or 'NoneType'"
                        .format(type(random_state)))
    elif type(random_state) is int and random_state < 0:
        raise ValueError("VALUE ERROR: argument `random_state` must be a positive integer")

    if random_state is not None:
        # Set the seed: making results reproducible
        np.random.seed(random_state)

    # Argument `nb points by chunk`
    if type(nb_points_by_chunk) is not int:
        raise TypeError("TYPE ERROR: type of argument `nb_points_by_chunk` is {}, expected 'int'"
                        .format(nb_points_by_chunk))
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
    if spark or (spark is None and SparkUtils.check_spark_usage(tsuid_list=tsuid_list, nb_ts_criteria=100,
                                                                nb_points_by_chunk=nb_points_by_chunk)):

        # TODO: il faut convertir nb_points_by_chunk en nb_ts_by_chunk -> les TS sont extraites intégralement
        result_df = fit_kmeans_spark_internal(tsuid_list=tsuid_list,
                                              n_cluster=nb_clusters,
                                              nb_ts_by_chunks=nb_points_by_chunk,
                                              random_state_kmeans=random_state)
    else:
        result_df = fit_kmeans_sklearn_internal(tsuid_list=tsuid_list,
                                                n_cluster=nb_clusters,
                                                random_state_kmeans=random_state)
    # ------------------------------------
    # 2 - Compute the MDS (Multidimensional scaling) (purpose : 2 dimensional visualisation)
    # ------------------------------------
    # Note that the seed (random_state_mds) is the same
    all_positions = mds_representation_kmeans(data=result_df, random_state_mds=random_state)
    # -----------------------
    # 3 - Prepare the outputs
    # -----------------------
    result = format_kmeans(all_positions=all_positions, n_cluster=nb_clusters)

    # For now, model is not outputed
    return result  #, model
