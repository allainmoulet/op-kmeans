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
import sklearn

from ikats.core.resource.api import IkatsApi

import logging

LOGGER = logging.getLogger(__name__)
"""
    K-Means Predict on Time Series
    ================================
    ..note::
    ..note:: 
    ..note::

    Here are the improvements that can be performed:
        * 
        * 
        *
"""


def _check_period_and_nb_points(tsuid_list):
    """
    Check that the provided TS list (`tsuid_list`) have the same `period` and `nb_points`.
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
        # If both metadatas are available, we continue
        else:
            period = int(float(md['qual_ref_period']))
            nb_points = int(md['qual_nb_points'])
        # -----------------------------------------------------------
        # 2 - Check if the metadatas are the same (period, nb_points)
        # -----------------------------------------------------------
        # Set the metadatas of the 1st TS as references
        if tsuid == tsuid_list[0]:
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
            elif nb_points != ref_nb_points:
                raise ValueError(
                    "TS {}: metadata `qual_nb_points` {} is different from those of the other TS (expected {})".format(
                        tsuid, period, ref_period)
                )
    return ref_period, ref_nb_points


def kmeans_predict_sklearn_internal(model, tsuid_list):
    """
    Performs prediction on time series regarding the sklearn K-Means model given in arguments.

    :param model: The sklearn K-Means model to be used to cluster the new time series
    :type model: sklearn.cluster.k_means

    :param tsuid_list: The list of tsuid to use
    :type tsuid_list: list of str

    :return predictions_table: dict formatted as awaited by functional type table (2 columns)
    :rtype predictions_table: dict of dicts

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
    Resulting viz:
    |   FID \  |     CLUSTER      |
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
        # TODO : Rajouter un attribut (dict) metadata au modèle lors de l'étape K-means.
        # TODO : On a besoin de tsuid, du nombre de points et de la période.
        if model.metadatas['qual_ref_period'] is not period:
            raise ValueError("The time series provided don't have the same period as the centroids")
        if model.metadatas['qual_nb_points'] is not nb_points:
            raise ValueError("The time series provided don't have the same number of points as the centroids")
        # -------------------------------------------------------------
        # 1 - Process the data in the shape needed by sklearn's K-Means
        # -------------------------------------------------------------
        # Extract data (just data values, not timestamps)
        data_sklearn = np.array(IkatsApi.ts.read(tsuid_list))[:, :, 1]
        # Shape = (n_ts, n_times)
        # ------------------------------------------------
        # 2 - Predict the clusters for the new TS provided
        # ------------------------------------------------
        predictions = model.predict(data_sklearn)
        # ------------------------------------------------
        # 3 - Format the result
        # ------------------------------------------------
        predictions_table = {
            'content': {
                # str(i + 1) : First class in K-Means is '0', and we want it to be '1'
                'cells': [[str(i + 1)] for i in predictions.tolist()]
            },
            'headers': {
                'col': {'data': ['CLUSTER']},
                'row': {'data': ['FID'] + [IkatsApi.fid.read(tsuid=i) for i in tsuid_list]}
            },
            'table_desc': {
                'desc': "Table of the K-Means predictions",
                'name': "K-Means_table_pred"
            }
        }
        LOGGER.debug(" --- Finished K-Means predict in: %.3f seconds --- ", time.time() - start_loading_time)
        return predictions_table
    finally:
        LOGGER.info("--- Finished to run fit_kmeans_spark_internal() function ---")


# EN STAND-BY EN ATTENDANT LE STOCKAGE DES MODÈLES SPARK
# def kmeans_predict_spark_internal(model, tsuid_list):
#     """
#     Performs prediction on time series regarding the sklearn K-Means model given in arguments.
#
#     :param model: The Spark K-Means model to be used to cluster the new time series
#     :type model: pyspark.ml.clustering.KMeansModel
#
#     :param tsuid_list: The time series to be clustered
#     :type tsuid_list: list of dicts
#
#     :return predictions_table:
#     :rtype predictions_table:
#     """
#     pass


def kmeans_on_ts_predict(model, ts_list):
    """
    Performs prediction on time series regarding the sklearn K-Means model given in arguments.

    :param model: The K-Means model to be used to cluster the new time series
    :type model: sklearn.cluster.k_means OR pyspark.ml.clustering.KMeansModel

    :param ts_list: The time series to be clustered
    :type ts_list: list of dicts

    :return predictions_table: The K-Means model used
    :rtype predictions_table: dict of dicts

    ..Example:
    {
        'content': {'cells': [['0'], ['2'], ['1'], ['0']]},
        'headers': {
            'col': {'data': ['CLUSTER']},
            'row': {'data': ['FID', 'UT_KMEANS_PRED_TS_1', ..., 'UT_KMEANS_PRED_TS_4']}
        },
        'table_desc': {
            'desc': 'Table of the K-Means predictions',
            'name': 'K-Means_table_pred'
        }
    }
    """
    # ----------------
    # Check the inputs
    # ----------------
    # Argument `model`
    if not isinstance(model, sklearn.cluster.k_means_.KMeans):
        raise TypeError(
            "TYPE ERROR: class of argument `model` is {}, expected sklearn.cluster.k_means_.KMeans".format(type(model)))
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
    # -----------------------------
    # Prediction on new time series
    # -----------------------------
    if model.__module__ == 'sklearn.cluster.k_means_':
        predictions_table = kmeans_predict_sklearn_internal(model=model, tsuid_list=tsuid_list)
    # elif model.__module__ == 'pyspark.ml.clustering.KMeansModel':
    #     predictions_table = kmeans_predict_spark_internal(model=model, tsuid_list=tsuid_list)
    else:
        raise NotImplementedError
    return predictions_table
