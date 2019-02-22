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
import unittest
import numpy as np
import logging
import collections

from ikats.core.resource.api import IkatsApi
from ikats.algo.kmeans.kmeans_on_ts_predict import kmeans_on_ts_predict
from sklearn.cluster import KMeans

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


def gen_scenario(case):
    """
    Generate scenarios in order to perform UT.

    :param case: Allow to choose the generated scenario
    :type case: int

    :return:
        * my_clusters: A mock of the clustering obtained in the previous step
        * my_model: A mock of the K-Means model obtained in the previous step
        * my_old_ts: A mock of the TS used to build the K-Means model in the previous step
        * my_new_ts: The new set of TS to be assigned according to the K-Means model
    :rtype:
        * my_clusters: dict of dicts
        * my_model: sklearn.cluster.k_means_.KMeans
        * my_old_ts: list of dicts
        * my_new_ts: list of dicts
    """
    # -------------------
    # CASE 1: Normal case
    # -------------------
    if case == 1:
        # --- Generate the 4 new TS to be clustered according to the previous 'K-Means on TS' step ---
        ts_new = np.array(
            [
                np.array([
                    [14879030000, 10],
                    [14879031000, 9],
                ]),
                np.array([
                    [14879030000, 1],
                    [14879031000, 5],
                ]),
                np.array([
                    [14879030000, 2],
                    [14879031000, 6],
                ]),
                np.array([
                    [14879030000, 12],
                    [14879031000, 11],
                ])
            ], np.float64)

        # -------------------------------
        # Creation of the new TS in IKATS
        # -------------------------------
        my_new_ts = []
        for i in range(np.array(ts_new).shape[0]):
            # `fid` must be unique for each TS
            current_fid = 'UT_KMEANS_PRED_NEW_TS_' + str(i + 1)
            # Create TS
            created_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_new)[i, :, :])
            # Create metadata 'qual_ref_period'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
            # Create metadata 'qual_nb_points'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=2, force_update=True)
            # Check the 'status'
            if not created_ts['status']:
                raise SystemError("Error while creating TS %s" % current_fid)
            # Create a list of lists of TS (dicts)
            my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_new[i]})

    # ------------------------------------------------------------------------
    # CASE 2: Abnormal case - TS don't have the same period between themselves
    # ------------------------------------------------------------------------
    elif case == 2:
        ts_1 = np.array(
            [
                [14879030000, 12],
                [14879031000, 10],
            ], np.float64)
        ts_2 = np.array(
            # Not the same period
            [
                [14879030000, 15],
                [14879041000, 14],
            ], np.float64)
        # -------------------------------
        # Creation of the new TS in IKATS
        # -------------------------------
        my_new_ts = []
        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_TS_1'
        # Create TS
        created_ts = IkatsApi.ts.create(fid=current_fid, data=ts_1)
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
        # Check the 'status'
        if not created_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_1})

        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_TS_2'
        # Create TS
        created_ts = IkatsApi.ts.create(fid=current_fid, data=ts_2)
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=11000, force_update=True)
        # Check the 'status'
        if not created_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_2})

    # ----------------------------------------------------------------------------------
    # CASE 3: Abnormal case - TS don't have the same number of points between themselves
    # ----------------------------------------------------------------------------------
    elif case == 3:
        ts_1 = np.array(
            [
                [14879030000, 11],
                [14879031000, 10],
            ], np.float64)
        ts_2 = np.array(
            [
                [14879030000, 11],
                [14879031000, 10],
                [14879032000, 9],
            ], np.float64)
        # -------------------------------
        # Creation of the new TS in IKATS
        # -------------------------------
        my_new_ts = []
        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_TS_1'
        # Create TS
        created_ts = IkatsApi.ts.create(fid=current_fid, data=ts_1)
        # Create metadata 'qual_nb_points'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=2, force_update=True)
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
        # Check the 'status'
        if not created_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_1})

        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_TS_2'
        # Create TS
        created_ts = IkatsApi.ts.create(fid=current_fid, data=ts_2)
        # Create metadata 'qual_nb_points'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=3, force_update=True)
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
        # Check the 'status'
        if not created_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_2})

    # ----------------------------------------------------------------------
    # CASE 4: Abnormal case - TS don't have the same period as the centroids
    # ----------------------------------------------------------------------
    elif case == 4:
        ts_new = np.array(
            [
                np.array([
                    [14879030000, 11],
                    [14879041000, 10],
                ]),
                np.array([
                    [14879030000, 1],
                    [14879041000, 5],
                ]),
                np.array([
                    [14879030000, 18],
                    [14879041000, 22],
                ]),
                np.array([
                    [14879030000, 12],
                    [14879041000, 11],
                ])
            ], np.float64)

        # -------------------------------
        # Creation of the new TS in IKATS
        # -------------------------------
        my_new_ts = []
        for i in range(np.array(ts_new).shape[0]):
            # `fid` must be unique for each TS
            current_fid = 'UT_KMEANS_PRED_NEW_TS_' + str(i + 1)
            # Create TS
            created_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_new)[i, :, :])
            # Create metadata 'qual_ref_period'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=11000, force_update=True)
            # Create metadata 'qual_nb_points'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=2, force_update=True)
            # Check the 'status'
            if not created_ts['status']:
                raise SystemError("Error while creating TS %s" % current_fid)
            # Create a list of lists of TS (dicts)
            my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_new[i]})

    # ----------------------------------------------------------------------
    # CASE 5: Abnormal case - TS don't have the same period as the centroids
    # ----------------------------------------------------------------------
    elif case == 5:
        ts_new = np.array(
            [
                np.array([
                    [14879030000, 11],
                    [14879031000, 10],
                    [14879032000, 9],
                ]),
                np.array([
                    [14879030000, 1],
                    [14879031000, 5],
                    [14879032000, 9],
                ]),
                np.array([
                    [14879030000, 18],
                    [14879031000, 22],
                    [14879032000, 26],
                ]),
                np.array([
                    [14879030000, 12],
                    [14879031000, 11],
                    [14879032000, 10],
                ])
            ], np.float64)

        # -------------------------------
        # Creation of the new TS in IKATS
        # -------------------------------
        my_new_ts = []
        for i in range(np.array(ts_new).shape[0]):
            # `fid` must be unique for each TS
            current_fid = 'UT_KMEANS_PRED_NEW_TS_' + str(i + 1)
            # Create TS
            created_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_new)[i, :, :])
            # Create metadata 'qual_ref_period'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
            # Create metadata 'qual_nb_points'
            IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=3, force_update=True)
            # Check the 'status'
            if not created_ts['status']:
                raise SystemError("Error while creating TS %s" % current_fid)
            # Create a list of lists of TS (dicts)
            my_new_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_new[i]})

    else:
        raise NotImplementedError

    # --- Generate the 'Model' output from the 'K-Means on TS' operator ---
    # Fit the K-Means model
    x = np.array([[2, 6], [4, 8], [9, 8], [7, 6]])
    # The clustering obtained with this seed is `array([1, 1, 0, 0], dtype=int32)`
    my_model = KMeans(n_clusters=2, random_state=13).fit(x)
    # Add the metadatas to the model
    # my_model.metadatas = {'qual_ref_period': 1000, 'qual_nb_points': 2}

    # --- Generate the corresponding TS ---
    ts_old = np.array(
        [
            np.array([
                [14879030000, 2],
                [14879031000, 6],
            ]),
            np.array([
                [14879030000, 4],
                [14879031000, 8],
            ]),
            np.array([
                [14879030000, 9],
                [14879031000, 8],
            ]),
            np.array([
                [14879030000, 7],
                [14879031000, 6],
            ])
        ], np.float64)

    # -------------------------------
    # Creation of the old TS in IKATS
    # -------------------------------
    # For the old TS to be clustered
    my_old_ts = []
    for i in range(np.array(ts_old).shape[0]):
        # `fid` must be unique for each TS
        current_fid = 'UT_KMEANS_PRED_OLD_TS_' + str(i + 1)
        # Create TS
        created_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_old)[i, :, :])
        # Create metadata 'qual_ref_period'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_ref_period', value=1000, force_update=True)
        # Create metadata 'qual_nb_points'
        IkatsApi.md.create(tsuid=created_ts['tsuid'], name='qual_nb_points', value=2, force_update=True)
        # Check the 'status'
        if not created_ts['status']:
            raise SystemError("Error while creating TS %s" % current_fid)
        # Create a list of lists of TS (dicts)
        my_old_ts.append({'tsuid': created_ts['tsuid'], 'funcId': created_ts['funcId'], 'ts_content': ts_old[i]})

    # --- Generate the 'Clusters' output from the 'K-Means on TS' operator ---
    my_clusters = {
        'C1': {
            'centroid': [3, 7],
            my_old_ts[0]['tsuid']: [2, 6],
            my_old_ts[1]['tsuid']: [4, 8]
        },
        'C2': {
            'centroid': [8, 7],
            my_old_ts[2]['tsuid']: [9, 8],
            my_old_ts[3]['tsuid']: [7, 6]
        }
    }

    return my_clusters, my_model, my_old_ts, my_new_ts


class TestKmeansOnTsPredict(unittest.TestCase):
    """
    Test of K-Means Predict on time series
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
        # Scenario generation to perform unit test
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=1)
        try:
            # -----------------
            # Argument `result`
            # -----------------
            # Wrong type (not a dict)
            with self.assertRaises(TypeError, msg="Testing arguments: Error in testing `result` type --- Wrong type"):
                # noinspection PyTypeChecker
                kmeans_on_ts_predict(result=42, model=my_model, ts_list=my_new_ts)
            # # Empty
            with self.assertRaises(TypeError, msg="Testing arguments: Error in testing `result` type --- Empty"):
                # noinspection PyTypeChecker
                kmeans_on_ts_predict(result=None, model=my_model, ts_list=my_new_ts)
            # ----------------
            # Argument `model`
            # ----------------
            # Wrong type (not a sklearn KMeans model)
            with self.assertRaises(TypeError, msg="Testing arguments: Error in testing `model` type --- Wrong type"):
                kmeans_on_ts_predict(result=my_clusters, model=42, ts_list=my_new_ts)
            # Empty
            with self.assertRaises(TypeError, msg="Testing arguments: Error in testing `model` type --- Empty"):
                kmeans_on_ts_predict(result=my_clusters, model=None, ts_list=my_new_ts)
            # ------------------
            # Argument `ts_list`
            # ------------------
            # Wrong type (not a list)
            with self.assertRaises(TypeError, msg="Testing arguments: Error in testing `ts_list` type --- Wrong type"):
                # noinspection PyTypeChecker
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=42)
            # Empty
            with self.assertRaises(ValueError, msg="Testing arguments: Error in testing `ts_list` type --- Empty"):
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=[])
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    # ------------------ #
    # OUTPUTS TYPE TESTS #
    # ------------------ #
    def test_kmeans_predict_output_type(self):
        """
        Test the type of the output of the kmeans_on_ts_predict() function.
        """
        # Scenario generation to perform unit test
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=1)
        try:
            # Test the wrapper
            table = kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
            # Checks of output's type (dict)
            self.assertEqual(type(table), collections.defaultdict,
                             msg="ERROR: The output's type is not 'collections.defaultdict'")
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    # ------------- #
    # RESULTS TESTS #
    # ------------- #
    def test_kmeans_predict_result(self):
        """
        Test the result of the kmeans_on_ts_predict() function on a basic example.
        """
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=1)
        # Expected result
        expected_result = [['1'], ['2'], ['2'], ['1']]
        try:
            # Test the wrapper
            table = kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
            # We only compare the predictions (not the FID)
            self.assertEqual(expected_result, table['content']['cells'],
                             msg="ERROR: The output is not the one expected")
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    # --------------- #
    # ALIGNMENT TESTS #
    # --------------- #
    def test_period_time_series(self):
        """
        Test the behavior of the function kmeans_on_ts_predict() when the time series provided don't have the same
        number of points.
        """
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=2)
        try:
            with self.assertRaises(ValueError, msg="The time series provided don't have the same period"):
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    def test_number_points_time_series(self):
        """
        Test the behavior of the function kmeans_on_ts_predict() when the time series provided don't have the same
        number of points.
        """
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=3)
        try:
            with self.assertRaises(
                    ValueError,
                    msg="ERROR: The time series provided don't have the same number of points"):
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    def test_period_with_centroids(self):
        """
        Test the behavior of the function kmeans_on_ts_predict() when the time series provided don't have the same
        period as the centroids resulting from the previous K-Means step.
        """
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=4)
        try:
            with self.assertRaises(ValueError,
                                   msg="The time series provided don't have the same period as the centroids"):
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)

    def test_number_points_with_centroids(self):
        """
        Test the behavior of the function kmeans_on_ts_predict() when the time series provided don't have the same
        number of points as the centroids resulting from the previous K-Means step.
        """
        my_clusters, my_model, my_old_ts, my_new_ts = gen_scenario(case=5)
        try:
            with self.assertRaises(ValueError,
                                   msg="The time series don't have the same number of points as the centroids"):
                kmeans_on_ts_predict(result=my_clusters, model=my_model, ts_list=my_new_ts)
        finally:
            # Clean up database
            self.clean_up_db(my_old_ts)
            self.clean_up_db(my_new_ts)
