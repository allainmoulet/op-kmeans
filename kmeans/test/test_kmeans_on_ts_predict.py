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
import unittest
import numpy as np
import logging

# IKATS import
# from ikats.core.library.spark import SSessionManager
from ikats.core.resource.api import IkatsApi
from ikats.algo.kmeans.kmeans_on_ts_predict import kmeans_on_ts_predict

from sklearn.cluster import KMeans

# from pyspark.ml.clustering import KMeans as KMeansSpark
# from pyspark.ml.linalg import Vectors

LOGGER = logging.getLogger()
# Log format. Used to be DEBUG
LOGGER.setLevel(logging.WARNING)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
# Used to be DEBUG
STREAM_HANDLER.setLevel(logging.WARNING)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


def gen_result_kmeans_on_ts(case):
    """
    Generate results of the shape of kmeans_on_ts in order to test
    """
    # -----------------------------------------------
    # CASE 1: SKLEARN - 4 TS DIVIDED IN 2 GROUPS OF 2
    # -----------------------------------------------
    if case == 1:
        result_kmeans_on_ts = {
            'C1': {
                'centroid': [3, 5],
                '*tsuid1*': [2, 4],
                '*tsuid2*': [4, 6]
            },
            'C2': {
                'centroid': [8, 7],
                '*tsuid3*': [9, 8],
                '*tsuid4*': [7, 6]
            }
        }
        # Fit the K-Means model
        x = np.array([[2, 4], [4, 6], [9, 8], [7, 6]])
        model_kmeans_on_ts = KMeans(n_clusters=2, random_state=13).fit(x)
    # -----------------------------------------------
    # CASE 2: SKLEARN - 9 TS DIVIDED IN 3 GROUPS OF 3
    # -----------------------------------------------
    elif case == 2:
        result_kmeans_on_ts = {
            'C1': {
                'centroid': [8, 7],
                '*tsuid4*': [9, 7],
                '*tsuid5*': [7, 6],
                '*tsuid6*': [8, 8]
            },
            'C2': {
                'centroid': [16, 20],
                '*tsuid7*': [15, 19],
                '*tsuid8*': [19, 23],
                '*tsuid9*': [14, 18]
            },
            'C3': {
                'centroid': [3, 5],
                '*tsuid1*': [2, 4],
                '*tsuid2*': [5, 6],
                '*tsuid3*': [2, 5]
            }
        }
        # Fit the K-Means model
        x = np.array([[9, 7], [7, 6], [8, 8], [15, 19], [19, 23], [14, 18], [2, 4], [5, 6], [2, 5]])
        model_kmeans_on_ts = KMeans(n_clusters=3, random_state=13).fit(x)
    # # ---------------------------------------------
    # # CASE 3: SPARK - 4 TS DIVIDED IN 2 GROUPS OF 2
    # # ---------------------------------------------
    # elif case == 3:
    #     result_kmeans_on_ts = {
    #         'C1': {
    #             'centroid': [3, 5],
    #             '*tsuid1*': [2, 4],
    #             '*tsuid2*': [4, 6]
    #         },
    #         'C2': {
    #             'centroid': [8, 7],
    #             '*tsuid3*': [9, 8],
    #             '*tsuid4*': [7, 6]
    #         },
    #     }
    #     # There is no equivalent predict method in Spark so we do the following:
    #     #   - store the centroids got with the Spark K-Means
    #     #   - calculate the distance between those the centroids and the new TS to predict
    #     #   - assign the TS to the group with the closest centroid
    #
    #     # The centroids obrained are hard-coded as the result of those following lines:
    # session = SSessionManager.get()
    # data = [(Vectors.dense([2, 4]),), (Vectors.dense([4, 6]),), (Vectors.dense([9, 8]), ), (Vectors.dense([7, 6]),)]
    # df = session.createDataFrame(data, ["time_series"])
    # model_kmeans_on_ts = KMeansSpark(featuresCol="time_series", k=2, seed=2).fit(df)
    # centroids = model_kmeans_on_ts.clusterCenters()
    # SSessionManager.stop()
    # centroids = [array([3, 5.]), array([8., 7.])]
    else:
        raise NotImplementedError
    return result_kmeans_on_ts, model_kmeans_on_ts


def gen_ts():
    """
    Generate a TS in the database in order to perform testing where id is defined.

    :return: the TSUID, funcId and all expected result (one per scaling)
    :rtype: list of dict
    """
    ts_content = np.array(
        [
            np.array([
                [14879030000, 11],
                [14879031000, 10],
            ]),
            np.array([
                [14879030000, 1],
                [14879031000, 3],
            ]),
            np.array([
                [14879030000, 18],
                [14879031000, 22],
            ]),
            np.array([
                [14879030000, 12],
                [14879031000, 11],
            ])
        ], np.float64)
    # ---------------------------------------------------------------------------------
    # Creation of the TS in IKATS
    # ---------------------------------------------------------------------------------
    result = []
    for i in range(np.array(ts_content).shape[0]):
        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_TS_' + str(i + 1)
        # Create TS
        my_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_content)[i, :, :])
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='qual_ref_period', value=100, force_update=True)
        # Check the 'status'
        if not my_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        result.append({'tsuid': my_ts['tsuid'], 'funcId': my_ts['funcId'], 'ts_content': ts_content[i]})
    return result


class TestKmeansOnTsPredict(unittest.TestCase):
    """
    Test of K-Means predict on time series
    """
    @staticmethod
    def clean_up_db(ts_info):
        """
        Clean up the database by removing created TS
        :param ts_info: list of TS to remove
        """
        for ts_item in ts_info:
            # Delete created TS
            IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_arguments_kmeans_on_ts_predict(self):
        """
        Test the behaviour when wrong arguments are given on kmeans_on_ts_predict()
        """
        _, model_kmeans_on_ts = gen_result_kmeans_on_ts(case=1)
        # ------------------
        # Argument `model`
        # ------------------
        # Wrong type (not a sklearn KMeans model)
        msg = "Testing arguments: Error in testing `model` type"
        with self.assertRaises(TypeError, msg=msg):
            kmeans_on_ts_predict(model=42, ts_list=[1, 2])
        # Empty
        msg = "Testing arguments: Error in testing `nb_clusters` not int"
        with self.assertRaises(TypeError, msg=msg):
            kmeans_on_ts_predict(model=None, ts_list=[1, 2])
        # ------------------
        # Argument `ts_list`
        # ------------------
        # Wrong type (not a list)
        msg = "Testing arguments: Error in testing `ts_list` type"
        with self.assertRaises(TypeError, msg=msg):
            # noinspection PyTypeChecker
            kmeans_on_ts_predict(model=model_kmeans_on_ts, ts_list=42)
        # Empty
        msg = "Testing arguments: Error in testing `ts_list` as empty list"
        with self.assertRaises(ValueError, msg=msg):
            kmeans_on_ts_predict(model=model_kmeans_on_ts, ts_list=[])

    # ------------------ #
    # OUTPUTS TYPE TESTS #
    # ------------------ #
    def test_kmeans_predict_sklearn_output_type(self):
        """
        Test the 'type' of the results for the sklearn version of the sklearn K-means predict on time series
        """
        # Generate results in order to test the wrapper
        _, model_kmeans_on_ts = gen_result_kmeans_on_ts(case=1)
        # Generate ts
        my_ts = gen_ts()
        try:
            # Test the wrapper
            result = kmeans_on_ts_predict(model=model_kmeans_on_ts, ts_list=my_ts)
            # Checks of output's type (dict)
            self.assertEqual(type(result), dict, msg="ERROR: This output's type is not 'dict'")
        finally:
            self.clean_up_db(my_ts)

    # def test_kmeans_predict_spark_output_type(self):
    #     """
    #     Test the 'type' of the results for the sklearn version of the Spark K-means predict on time series
    #     """
    #     pass

    # ------------- #
    # RESULTS TESTS #
    # ------------- #
    def test_kmeans_predict_sklearn_result(self):
        """
        Test the result obtained for the sklearn version of the sklearn K-means algorithm on time series
        """
        # Generate results in order to test the wrapper
        _, model_kmeans_on_ts = gen_result_kmeans_on_ts(case=2)
        # Generate ts
        my_ts = gen_ts()
        # Expected result
        predictions_table = {
            'content': {
                'cells': [['1'], ['3'], ['2'], ['1']]
            },
            'headers': {
                'col': {'data': ['CLUSTER']},
                'row': {'data': ['FID', 'FID_1', 'FID_2']}
            },
            'table_desc': {
                'desc': "Table of the K-Means predictions",
                'name': "K-Means_table_pred"
            }
        }
        try:
            # Test the wrapper
            result = kmeans_on_ts_predict(model=model_kmeans_on_ts, ts_list=my_ts)
            # We only compare the predictions (not the FID)
            self.assertEqual(result['content']['cells'], predictions_table['content']['cells'],
                             msg="ERROR: The output is not the one expected")
        finally:
            self.clean_up_db(my_ts)

    # def test_kmeans_predict_spark_result(self):
    #     """
    #     Test the result obtained for the sklearn version of the Spark K-means algorithm on time series
    #     """
    #     pass
