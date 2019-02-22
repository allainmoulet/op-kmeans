"""
Copyright 2018-2019 CS Systèmes d'Information

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
import sklearn
from collections import defaultdict
from ikats.core.resource.api import IkatsApi
import logging

LOGGER = logging.getLogger(__name__)
"""
    K-Means Predict on Time Series
    ================================
    The function kmeans_on_ts_predict() checks the input given and calls kmeans_predict_sklearn_internal(). The latter
    is a wrapper of the sklearn.cluster.KMeans.predict() scikit-learn function. It is designed in 4 parts:
        * 1 - The call to _check_period_and_nb_points() function to check that the TS provided have the same number of
        points and the same period between themselves.
        * 2 - A check to see if the TS provided and the centroids calculated from the previous K-Means step have the
        same number of points and the same period.
        * 3 - The call to the sklearn.cluster.KMeans.predict() function.
        * 4 - A format step to get the following visualisation in IKATS: a table with TS in rows and links to the
        `curve` visualisation with all the curves of the considered cluster.
        
    ..note::
        The 'qual_ref_period' and the 'qual_nb_points' metadatas must be in every time serie's metadatas. That means
        that the user may have to run the 'time' part of the 'Quality Indicators' operator before using this one.
        Besides, time series can start (/end) from (/to) different points (no checks on start/end_date metadatas).
        
    ..note::
        We don't use normalisation on the time series, because we use Euclidian distance to calculate distances between
        them and the centroids obtained through the K-means algorithm.    
 
    .. note::
        Warning: all TS are named by their TSUID

    Here are the improvements that can be performed:
        * DTW distance can replace Euclidian one. In that case, we don't need anymore the TS to have the same number of
        points. 
        * When Spark models' storage in IKATS will be available, the Spark part of the 'K-Means on TS' operator will be
        implemented. This 'K-Means on TS Predict' operator as well. The choice of usage of Spark here will be totally
        determined by the class model used in the 'K-Means on TS'.
    """


def _check_period_and_nb_points(tsuid_list):
    """
    Check that the provided TS list (`tsuid_list`) have the same period and the same number of points.
    This function needs the metadatas `qual_ref_period` and 'qual_nb_points` which can be obtained by launching the
    IKATS operator `Quality Indicators`.

    :param tsuid_list: list of tsuid
    :type tsuid_list: list of str
    ..Example: ['tsuid1', 'tsuid2', ...]

    :return: Tuple composed by:
        * period (int)
        * number of points (int)

    :raises:
        * ValueError: TS are not aligned
        * ValueError: Some metadata are missing (`qual_ref_period` and `qual_nb_points`)

    ..Note: Comparisons are all made with the first TS
    """
    # Initialisation (avoid warnings)
    ref_period = None
    ref_nb_points = None
    # Read metadatas
    meta_list = IkatsApi.md.read(tsuid_list)
    # Perform operation iteratively on each TS
    for tsuid in tsuid_list:
        # ---------------------------------------------
        # 1 - Retrieve metadatas and check availability
        # ---------------------------------------------
        # Retrieve the metadatas
        md = meta_list[tsuid]
        # Check if the metadata 'qual_ref_period' is available
        if 'qual_ref_period' not in md.keys():
            raise ValueError(
                "No metaData `qual_ref_period` with tsuid {}... Please launch `Quality Indicator`".format(tsuid)
            )
        # Check if the metadata 'nb_points' is available
        elif 'qual_nb_points' not in md.keys():
            raise ValueError(
                "No metaData `nb_points` with tsuid {}... Please launch `Quality Indicator`".format(tsuid)
            )
        # If both metadatas are available, OK
        else:
            period = int(float(md['qual_ref_period']))
            nb_points = int(md['qual_nb_points'])
        # -----------------------------------------------------------
        # 2 - Check if the metadatas are the same (period, nb_points)
        # -----------------------------------------------------------
        # Set the metadatas of the 1st TS as references
        if tsuid is tsuid_list[0]:
            ref_period = period
            ref_nb_points = nb_points
        # Compare the metadatas of all the other TS with those of the 1st one
        else:
            # Compare `period`
            if period != ref_period:
                raise ValueError(
                    "TS {}: metadata `qual_ref_period` {} is different from those of the other TS (expected {})".format(
                        tsuid, period, ref_period)
                )
            # Compare `period`
            elif nb_points != ref_nb_points:
                raise ValueError(
                    "TS {}: metadata `qual_nb_points` {} is different from those of the other TS (expected {})".format(
                        tsuid, nb_points, ref_nb_points)
                )
    return ref_period, ref_nb_points


def kmeans_predict_sklearn_internal(result, model, tsuid_list):
    """
    Performs prediction on time series according to the sklearn K-Means model given in arguments. This function calls
    the corresponding function implemented in scikit learn.

    :param result: The result obtained as output of the K-Means on TS operator. Used for visualisation.
    :type result: dict of dict
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

    :param model: The sklearn K-Means model to be used to cluster the new time series
    :type model: sklearn.cluster.k_means

    :param tsuid_list: The list of tsuid to use
    :type tsuid_list: list of str

    :return predictions_table: dict formatted as awaited by functional type table (2 columns)
    :rtype predictions_table: collections.defaultdict
    ..Example:
    {'content': {
        'cells': [['2'], ['1']]
        },
    'headers': {
        'col': {'data': ['CLUSTER']},
        'row': {'data': ['FID', 'FID_1', 'FID_2']}
        },
    'table_desc': {
        'desc': "Description of the table in the menu 'Data Management' then 'Tables'",
        'name': "Name of the table in the menu 'Data Management' then 'Tables'"
        }
    }
    Resulting visualisation in IKATS:
    |   FID    |     CLUSTER      |
    |----------|------------------|
    |FID_1     | 2                |
    |FID_2     | 1                |
    """
    LOGGER.info(" --- Starting K-Means predict with scikit-learn --- ")
    try:
        start_loading_time = time.time()

        # ---------------------------------------------------------------------------------------------------------
        # 0 - Check that all the provided TS have the same period and number of points (raise ValueError otherwise)
        # ---------------------------------------------------------------------------------------------------------
        period, nb_points = _check_period_and_nb_points(tsuid_list)

        # -------------------------------------------------------------------------------------
        # 0bis - Check that the cendroids have the same number of points as well
        # -------------------------------------------------------------------------------------
        # We take one reference TSUID in order to compare its metadatas with them of the new TS in `tsuid_list`.
        # As we have already checked in the 'K-Means on TS' step that all the TS provided are aligned, taking the metas
        # of one reference tsuid is the same as taking those from the centroids. We do that because the centroids are
        # not recorded in IKATS.
        ref_tsuid = [i for i in result['C1'] if i != 'centroid'][0]
        # Check for the 'qual_ref_period' metadata
        if int(IkatsApi.md.read(ref_tsuid)[ref_tsuid]['qual_ref_period']) != period:
            raise ValueError(
                "The time series provided don't have the same period as the centroids. Get {} instead of {}"
                .format(int(IkatsApi.md.read(ref_tsuid)[ref_tsuid]['qual_ref_period']), period))
        # Check for the 'qual_nb_points' metadata
        if int(IkatsApi.md.read(ref_tsuid)[ref_tsuid]['qual_nb_points']) != nb_points:
            raise ValueError(
                "The time series provided don't have the same number of points as the centroids. Get {} instead of {}"
                .format(int(IkatsApi.md.read(ref_tsuid)[ref_tsuid]['qual_nb_points']), nb_points))

        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        # Extract input data (just data values, not timestamps)
        data_sklearn = np.array(IkatsApi.ts.read(tsuid_list))[:, :, 1]
        # Shape = (n_ts, n_times)

        # ------------------------------------------------
        # 2 - Predict the clusters for the new TS provided
        # ------------------------------------------------
        predictions = model.predict(data_sklearn)
        # Shape = (n_ts,)

        # ------------------------------------------------
        #  3- Format the result to display the table of predictions with links to visualisation
        # ------------------------------------------------
        table = defaultdict(dict)
        # 3.1 - Fill description fields
        table['table_desc']['title'] = 'K-Means Predictions on New TS'
        table['table_desc']['name'] = 'K-Means_table_pred'
        table['table_desc']['desc'] = 'Table of the K-Means predictions'
        # 3.2 - Fill content fields
        table['content']['cells'] = []
        table['content']['cells'].extend([[str(i + 1)] for i in predictions.tolist()])
        # 3.3 - Fill the headers for the columns
        table['headers']['col'] = dict()
        table['headers']['col']['data'] = ['Functional Id', 'Cluster Id']
        # 3.4 - Fill the headers for the rows with the FID of the TS and make links to visualisation
        table['headers']['row'] = dict()
        table['headers']['row']['data'] = [None]
        table['headers']['row']['data'].extend([IkatsApi.ts.fid(i) for i in tsuid_list])
        # Make links to `curve visualisation` when clicking on rows
        # This line is mandatory to have functional links
        table['headers']['row']['default_links'] = {'type': 'bucket_ts', 'context': 'raw'}
        table['headers']['row']['links'] = [None]
        for ind, tsuid in enumerate(tsuid_list):
            # Get the cluster predicted (0, 1, 2...) and add 1 to have the same as in 'result' argument (1, 2, 3...)
            cluster_id = predictions[ind] + 1
            # Get the tsuid of all the TS of the train step for the cluster_id of the current TS
            current_tsuid_list = list(result['C' + str(cluster_id)].keys())
            # Centroid is not a TS, it can't be plotted
            current_tsuid_list.remove('centroid')
            # Add the tsuid of the current TS
            current_tsuid_list.append(tsuid)
            # Get the FID of the tsuid to be plotted
            current_fid_list = [IkatsApi.ts.fid(i) for i in current_tsuid_list]
            table['headers']['row']['links'].extend([
                {
                    'val': {
                        # Start of comprehension list
                        'data': [
                            {
                                'tsuid': current_tsuid_list[x],
                                'funcId': current_fid_list[x]
                            } for x in range(len(current_tsuid_list))],
                        # End of comprehension list
                        'flags': [
                            {
                            'timestamp': int(IkatsApi.md.read(tsuid)[tsuid]['ikats_start_date'])
                            }]
                    }
                }
            ])
        return table
    finally:
        LOGGER.info(" --- Finished K-Means predict in: %.3f seconds --- ", time.time() - start_loading_time)


def kmeans_on_ts_predict(result, model, ts_list):
    """
    Performs prediction on time series data according to the sklearn K-Means model given in arguments. This function
    performs checks on the inputs and then calls kmeans_predict_sklearn_internal().

    :param result: The result obtained as output of the K-Means on TS operator. Used for visualisation.
    :type result: dict of dicts
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

    :param model: The K-Means model to be used to cluster the new time series
    :type model: sklearn.cluster.k_means

    :param ts_list: The time series to be clustered
    :type ts_list: list of dicts

    :return predictions_table: The table of results as needed for the visualisation of the results in IKATS
    :rtype predictions_table: collections.defaultdict
    """
    # ----------------
    # Check the inputs
    # ----------------
    # Argument `result`
    if type(result) is not dict:
        raise TypeError(
            "Type of argument `result` is {}, expected 'dict'".format(type(result)))
    elif not result:
        raise TypeError("Argument `result` is empty")
    # Argument `model`
    if not isinstance(model, sklearn.cluster.k_means_.KMeans):
        raise TypeError("Class of argument `model` is {}, expected sklearn.cluster.k_means_.KMeans".format(type(model)))
    elif not model:
        raise TypeError("Argument `model` is empty")
    # Argument `ts_list`
    if type(ts_list) is not list:
        raise TypeError("Type of argument `ts_list` is {}, expected 'list'".format(type(ts_list)))
    elif not ts_list:
        raise ValueError("Argument `ts_list` is empty")
    # Retrieve tsuid list
    try:
        tsuid_list = [x['tsuid'] for x in ts_list]
    except Exception:
        raise ValueError('Impossible to retrieve the tsuid list')
    # -----------------------------
    # Prediction on new time series
    # -----------------------------
    # Scikit learn case
    if model.__module__ == 'sklearn.cluster.k_means_':
        predictions_table = kmeans_predict_sklearn_internal(result=result, model=model, tsuid_list=tsuid_list)
    # elif model.__module__ == 'pyspark.ml.clustering.KMeansModel':
    #     predictions_table = kmeans_predict_spark_internal(model=model, tsuid_list=tsuid_list)
    else:
        raise NotImplementedError("Version using SPARK mode not yet implemented")
    return predictions_table
