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
from ikats.core.resource.api import IkatsApi
from ikats.algo.kmeans.kmeans_on_ts import fit_kmeans_on_ts

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


def gen_ts(ts_id):
    """
    Generate a TS in the database in order to perform testing where id is defined.

    :param ts_id: Identifier of the TS to generate. See content below for the structure.
    :type ts_id: int

    :return: the TSUID, funcId and all expected result (one per scaling)
    :rtype: list of dict
    """
    # Build TS identifier
    fid = 'UNIT_TEST_K-MEANS_%s' % ts_id
    # -----------------------------------------------------------------------------
    # CASE 1: 4 TS divided in 2 groups of 2 with 2 points per TS. shape = (4, 2, 2)
    # -----------------------------------------------------------------------------
    if ts_id == 1:
        ts_a1 = np.array(([10000, 7], [11000, 3]))
        ts_a2 = np.array(([10000, 9], [11000, 4]))
        ts_b1 = np.array(([10000, 14], [11000, 15]))
        ts_b2 = np.array(([10000, 12], [21000, 17]))
        ts_content = np.array([ts_a1, ts_a2, ts_b1, ts_b2], np.float64)
    # -------------------------------------------------------------------------------
    # CASE 2: 4 TS divided in 2 groups of 2 with 10 points per TS. shape = (4, 10, 2)
    # -------------------------------------------------------------------------------
    elif ts_id == 2:
        ts_a1 = np.array(([10000, 7], [11000, 3], [12000, 4], [13000, 9], [14000, 5], [15000, 6], [16000, 1],
                          [17000, 0], [18000, 1], [19000, 2]))
        ts_a2 = np.array(([10000, 6], [11000, 5], [12000, 5], [13000, 7], [14000, 6], [15000, 5], [16000, 2],
                          [17000, 1], [18000, 2], [19000, 1]))
        ts_b1 = np.array(([10000, 14], [11000, 13], [12000, 15], [13000, 15], [14000, 20], [15000, 30], [16000, 42],
                          [17000, 43], [18000, 47], [19000, 50]))
        ts_b2 = np.array(([10000, 12], [11000, 10], [12000, 14], [13000, 18], [14000, 19], [15000, 26], [16000, 41],
                          [17000, 45], [18000, 46], [19000, 51]))
        ts_content = np.array([ts_a1, ts_a2, ts_b1, ts_b2], np.float64)
    # -----------------------------------------------------------------------------
    # CASE 3: 6 TS divided in 3 groups of 2 with 2 points per TS. shape = (6, 2, 2)
    # -----------------------------------------------------------------------------
    elif ts_id == 3:
        ts_a1 = np.array(([10000, 7], [11000, 3]))
        ts_a2 = np.array(([10000, 9], [11000, 4]))
        ts_b1 = np.array(([10000, 14], [11000, 15]))
        ts_b2 = np.array(([10000, 12], [11000, 17]))
        ts_c1 = np.array(([10000, 26], [11000, 4]))
        ts_c2 = np.array(([10000, 24], [11000, 2]))
        ts_content = np.array([ts_a1, ts_a2, ts_b1, ts_b2, ts_c1, ts_c2], np.float64)
    # ---------------------------------------------------------------------------------
    # CASE 4: non aligned TS to be used in the test test_alignment(). shape = (4, 2, 2)
    # ---------------------------------------------------------------------------------
    elif ts_id == 4:
        ts_a1 = np.array(([10000, 7], [11000, 3]))
        ts_a2 = np.array(([20000, 9], [21000, 4]))
        ts_b1 = np.array(([10000, 14], [11000, 15]))
        ts_b2 = np.array(([40000, 12], [50000, 17]))
        ts_content = np.array([ts_a1, ts_a2, ts_b1, ts_b2], np.float64)
    else:
        raise NotImplementedError
    # ---------------------------------------------------------------------------------
    # Creation of the TS in IKATS
    # ---------------------------------------------------------------------------------
    result = []
    for i in range(np.array(ts_content).shape[0]):

        # `fid` must be unique for each TS
        current_fid = fid + '_TS_' + str(i + 1)
        # Create TS
        my_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_content)[i, :, :])
        # Create metadatas 'qual_nb_points', 'name' and 'funcId'
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='qual_nb_points', value=len(ts_content), force_update=True)
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='qual_ref_period', value=1, force_update=True)
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='funcId', value=current_fid, force_update=True)
        if not my_ts['status']:
            raise SystemError("Error while creating TS %s" % ts_id)
        # Create a list of lists of TS (dicts)
        result.append({'tsuid': my_ts['tsuid'], 'funcId': my_ts['funcId'], 'ts_content': ts_content[i]})
    return result


class TestKmeansOnTS(unittest.TestCase):
    """
    Test of K-Means algorithm on time series
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

    @unittest.skip
    def test_arguments_fit_kmeans_on_ts(self):
        """
        Test the behaviour when wrong arguments are given on fit_fit_kmeans_on_ts()
        """
        # Get the TSUID of the saved TS
        my_ts = gen_ts(1)
        try:
            # ----------------------------
            # Argument `ts_list`
            # ----------------------------
            # Wrong type (not a list)
            msg = "Testing arguments: Error in testing `ts_list` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=0.5, nb_clusters=2)
            # Empty
            msg = "Testing arguments: Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                fit_kmeans_on_ts(ts_list=[], nb_clusters=2)
            # ----------------------------
            # Argument `nb_clusters`
            # ----------------------------
            # Wrong type (not an int)
            msg = "Testing arguments: Error in testing `nb_clusters` type"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=-42)
            # Empty
            msg = "Testing arguments: Error in testing `nb_clusters` not int"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters="a")
            # ----------------------------
            # Argument `random state`
            # ----------------------------
            # Wrong type (not int or None)
            msg = "Testing arguments : Error in testing `random_state` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, random_state="a")
            # Negative number
            msg = "Testing arguments : Error in testing `random_state` type"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, random_state=-42)
            # ----------------------------
            # Argument `nb_points_by_chunk`
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, nb_points_by_chunk='a')
            # Negative number
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, nb_points_by_chunk=-42)
            # ----------------------------
            # Argument `spark`
            # ----------------------------
            # Wrong type (not NoneType or bool)
            msg = "Testing arguments : Error in testing `spark` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark='42')
        finally:
            # Clean up database
            self.clean_up_db(my_ts)

    # ------------------ #
    # OUTPUTS TYPE TESTS #
    # ------------------ #
    @unittest.skip
    def test_kmeans_sklearn_output_type(self):
        """
        Test the 'type' of the results for the sklearn version of the K-means algorithm on time series
        """
        rand_state = 1
        # TS creation
        # Test on non aligned TS
        my_ts = gen_ts(1)

        try:
            # Fit the model
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)

            # Checks of output's type (dict)
            self.assertEqual(type(result_kmeans), dict, msg="ERROR: This output's type is not 'dict'")
        finally:
            self.clean_up_db(my_ts)

    def test_kmeans_spark_output_type(self):
        """
        Test the 'type' of the results for the Spark version of the K-means algorithm on time series
        """
        rand_state = 1
        # TS creation
        my_ts = gen_ts(1)

        try:
            # Fit the model
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)
            # Checks of outputs' type (dict)
            self.assertEqual(type(result_kmeans), dict, msg="ERROR: This output's type is not 'dict'")
        finally:
            self.clean_up_db(my_ts)

    # ------------- #
    # RESULTS TESTS #
    # ------------- #
    def test_kmeans_sklearn_result(self):
        """
        Test the result obtained for the sklearn version of the K-means algorithm on time series
        """
        # Used for reproducible results
        rand_state = 1
        # TS creation
        my_ts = gen_ts(2)

        # Get the tsuid list
        tsuid_list = [x['tsuid'] for x in my_ts]

        try:
            # Fit the model
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)

            # TSUID associated to the first centroid
            tsuid_group = list(result_kmeans.get("C1").keys())
            # Example:
            # ['centroid', '14ED6B00000100076C0000020006E0000003000772', '2630EF00000100076C0000020006E0000003000771']

            # Check what TS are contained in the 1st cluster
            condition = all(x in tsuid_group for x in tsuid_list[0:2]) or \
                all(x in tsuid_group for x in tsuid_list[2:4])
            self.assertTrue(condition, msg="ERROR: The obtained clustering is not the one expected")

            # Check if each cluster contains 2 time series
            condition = len(result_kmeans['C1']) == len(result_kmeans['C2'])
            self.assertTrue(condition, msg="ERROR: The obtained clustering is un-balanced")

        finally:
            self.clean_up_db(my_ts)

    # @unittest.skip
    def test_kmeans_spark_result(self):
        """
        Test the result obtained for the sklearn version of the K-means algorithm on time series
        """
        # Used for reproducible results
        # Used for reproducible results
        rand_state = 1
        # TS creation
        my_ts = gen_ts(2)

        # Get the tsuid list
        tsuid_list = [x['tsuid'] for x in my_ts]

        try:
            # Fit the model
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)

            # TSUID associated to the first centroid
            tsuid_group = list(result_kmeans.get("C1").keys())

            # Check what TS are contained in the 1st cluster
            condition = all(x in tsuid_group for x in tsuid_list[0:2]) or \
                all(x in tsuid_group for x in tsuid_list[2:4])
            self.assertTrue(condition, msg="ERROR: The obtained clustering is not the one expected")

            # Check if each cluster contains 2 time series
            condition = len(result_kmeans['C1']) == len(result_kmeans['C2'])
            self.assertTrue(condition, msg="ERROR: The obtained clustering is un-balanced")

        finally:
            self.clean_up_db(my_ts)

    def test_diff_sklearn_spark(self):
        """
        Test the difference of result between the functions kmeans_spark() and kmeans_sklearn() on time series.
        The same result should be obtained with both ways.
        Just test clustering, not MDS 2-dimensional representation.
        ..Note: MDS 2-dimensional transformation result is not tested.
        """
        # Used for reproducible results
        rand_state = 1
        # TS creation
        my_ts = gen_ts(1)
        try:
            # Fit the model
            result_sklearn = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)
            result_spark = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)

            # Comparison of the results: CASE 1: `result_sklearn['C1']`
            # Because MDS transformation is sensitive to the order, a label switching can cause differences in
            # results
            condition_same_labels = (result_sklearn['C1'].keys()) == set(result_spark['C1'].keys())
            condition_switched_labels = (result_sklearn['C1'].keys()) == set(result_spark['C2'].keys())

            # Case of labels switching: we have the same clustering but different labels.
            # Example:
            # result_sklearn = {{'C1': {'tsuid_1':[...], 'tsuid_2': [...]}, {'C2': {'tsuid_3': [...], 'tsuid_4': [...]}}
            # result_spark = {{'C2': {'tsuid_1': [...], 'tsuid_2': [...]}, {'C1': {'tsuid_3': [...], 'tsuid_4': [...]}}
            condition = condition_same_labels or condition_switched_labels
            msg = "ERROR: Spark clustering and scikit-learn clustering are different\n" \
                  "Result sklearn = {} \n" \
                  "Result Spark ={}".format(result_sklearn, result_spark)
            self.assertTrue(condition, msg=msg)

            # CASE 2: `result_sklearn['C2']`
            condition_same_labels = (result_sklearn['C2'].keys()) == set(result_spark['C2'].keys())
            condition_switched_labels = (result_sklearn['C2'].keys()) == set(result_spark['C1'].keys())
            condition = condition_same_labels or condition_switched_labels
            self.assertTrue(condition, msg=msg)

        finally:
            self.clean_up_db(my_ts)

    def test_alignment_time_series(self):
        """
        Test the behavior of the function fit_kmeans_on_ts() when the time series are not aligned. It means they don't
        have the same start date and/or end date and/or period.
        """
        # Used for reproducible results
        rand_state = 1
        # TS creation
        my_ts = gen_ts(4)
        try:
            msg = "ERROR: The time series are not aligned - {} mode"
            # Test on scikit-learn mode
            with self.assertRaises(ValueError, msg=msg.format("scikit-learn")):
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)
            # Test on Spark mode
            with self.assertRaises(ValueError, msg=msg.format("Spark")):
                fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)
        finally:
            self.clean_up_db(my_ts)
