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
import logging

import numpy as np

from ikats.core.resource.api import IkatsApi
from ikats.core.library.exception import IkatsInputTypeError, IkatsInputContentError
from ikats.algo.kmeans.kmeans_on_ts import fit_kmeans_sklearn_internal, fit_kmeans_on_ts

# Set LOGGER
LOGGER = logging.getLogger()
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

def gen_ts(ts_id):
    """
    Generate a TS in the database in order to perform testing where id is defined.

    :param ts_id: Identifier of the TS to generate. See content below for the structure.
    :type ts_id: int

    :return: the TSUID, funcId and all expected result (one per scaling)
    :rtype: dict
    """
    # Build TS identifier
    fid = 'UNIT_TEST_K-MEANS_%s' % ts_id
    # -----------------------------------------------------------------------------
    # CASE 1: 4 TS divided in 2 groups of 2 with 2 points per TS. shape = (4, 2, 2)
    # -----------------------------------------------------------------------------
    if ts_id == 1:
        time = list(range(1, 3))
        ts_A1 = [7, 3]
        ts_A2 = list(ts_A1 + np.random.normal(0, 1, size=len(ts_A1)))
        ts_B1 = [14, 13]
        ts_B2 = list(ts_B1 + np.random.normal(0, 1, size=len(ts_B1)))
        ts_content = np.array([np.array([time, ts_A1]).T,
                               np.array([time, ts_A2]).T,
                               np.array([time, ts_B1]).T,
                               np.array([time, ts_B2]).T], np.float64)
    # -------------------------------------------------------------------------------
    # CASE 2: 4 TS divided in 2 groups of 2 with 10 points per TS. shape = (4, 10, 2)
    # -------------------------------------------------------------------------------
    elif ts_id == 2:
        time = list(range(1, 11))
        ts_A1 = [7, 3, 4, 9, 5, 6, 1, 0, 1, 2]
        ts_A2 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_B1 = [14, 13, 15, 15, 20, 30, 42, 43, 47, 50]
        ts_B2 = ts_B1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_content = np.array([np.array([time, ts_A1]).T,
                               np.array([time, ts_A2]).T,
                               np.array([time, ts_B1]).T,
                               np.array([time, ts_B2]).T], np.float64)
    # -------------------------------------------------------------------------------
    # CASE 3: 15 TS divided in 3 groups of 5 with 2 points per TS. shape = (15, 2, 2)
    # -------------------------------------------------------------------------------
    elif ts_id == 3:
        time = list(range(1, 3))
        ts_A1 = [7, 3]
        ts_A2 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A3 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A4 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A5 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_B1 = [14, 13]
        ts_B2 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B3 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B4 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B5 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_C1 = [50, 55]
        ts_C2 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C3 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C4 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C5 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_content = np.array([np.array([time, ts_A1]).T, np.array([time, ts_A2]).T, np.array([time, ts_A3]).T,
                               np.array([time, ts_A4]).T, np.array([time, ts_A5]).T, np.array([time, ts_B1]).T,
                               np.array([time, ts_B2]).T, np.array([time, ts_B3]).T, np.array([time, ts_B4]).T,
                               np.array([time, ts_B5]).T, np.array([time, ts_C1]).T, np.array([time, ts_C2]).T,
                               np.array([time, ts_C3]).T, np.array([time, ts_C4]).T, np.array([time, ts_C5]).T],
                              np.float64)
    # ---------------------------------------------------------------------------------
    # CASE 4: 15 TS divided in 3 groups of 5 with 10 points per TS. shape = (15, 10, 2)
    # ---------------------------------------------------------------------------------
    elif ts_id == 4:
        time = list(range(1, 11))
        ts_A1 = [7, 3, 4, 9, 5, 6, 1, 0, 1, 2]
        ts_A2 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A3 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A4 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_A5 = ts_A1 + np.random.normal(0, 1, size=len(ts_A1))
        ts_B1 = [14, 13, 15, 15, 20, 30, 42, 43, 47, 50]
        ts_B2 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B3 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B4 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_B5 = ts_B1 + np.random.normal(0, 1, size=len(ts_B1))
        ts_C1 = [50, 55, 54, 52, 59, 57, 51, 55, 52, 58]
        ts_C2 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C3 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C4 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_C5 = ts_C1 + np.random.normal(0, 1, size=len(ts_C1))
        ts_content = np.array([np.array([time, ts_A1]).T, np.array([time, ts_A2]).T, np.array([time, ts_A3]).T,
                               np.array([time, ts_A4]).T, np.array([time, ts_A5]).T, np.array([time, ts_B1]).T,
                               np.array([time, ts_B2]).T, np.array([time, ts_B3]).T, np.array([time, ts_B4]).T,
                               np.array([time, ts_B5]).T, np.array([time, ts_C1]).T, np.array([time, ts_C2]).T,
                               np.array([time, ts_C3]).T, np.array([time, ts_C4]).T, np.array([time, ts_C5]).T],
                              np.float64)
    else:
        raise NotImplementedError
    # ---------------------------------------------------------------------------------
    # Creation of the TS in IKATS
    # ---------------------------------------------------------------------------------
    result = []
    for i in range(np.array(ts_content).shape[0]):
        # `fid` must be unique for each TS
        current_fid = fid + '_TS_' + str(i + 1)
        # Create a TS, dict with keys: 'tsuid', 'funcId', 'status', 'reponseStatus', 'summary', 'errors' and 'numberOfSuccess'
        myTS = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_content)[i, :, :])
        # Create metadatas 'qual_nb_points', 'name' and 'funcId'
        IkatsApi.md.create(tsuid=myTS['tsuid'], name='qual_nb_points', value=len(ts_content), force_update=True)
        IkatsApi.md.create(tsuid=myTS['tsuid'], name='metric', value='metric_%s' % ts_id, force_update=True)
        IkatsApi.md.create(tsuid=myTS['tsuid'], name='funcId', value=current_fid, force_update=True)
        if not myTS['status']:
            raise SystemError("Error while creating TS %s" % ts_id)
        # Create a list of lists of TS (dicts)
        result.append({'tsuid': myTS['tsuid'], 'funcId': myTS['funcId'], 'ts_content': ts_content[i]})
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

    def test_arguments_fit_kmeans_on_ts(self):
        """
        Test the behaviour when wrong arguments are given on fit_kmeans_on_ts()
        """
        # TODO : Je ne dois tester que le wrapper : fit_kmeans_on_ts(). Code à réaménager en conséquence.
        # Get the TSUID of the saved TS
        tsuid_list = gen_ts(1)
        try:
            # ----------------------------
            # Argument `ts_list`
            # ----------------------------
            # Wrong type (not list)
            msg = "Testing arguments : Error in testing `ts_list` type"
            with self.assertRaises(TypeError, msg=msg):
                kmeans_on_ts(ts_list=0.5)
            # Empty
            msg = "Testing arguments : Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                kmeans_on_ts(ts_list=[])
            # ----------------------------
            # Argument `nb_clusters`
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_clusters` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                kmeans_on_ts(ts_list=tsuid_list, nb_clusters=-4)
            # Empty
            msg = "Testing arguments : Error in testing `nb_clusters` as empty int"
            with self.assertRaises(ValueError, msg=msg):
                kmeans_on_ts(ts_list=tsuid_list)
            # ----------------------------
            # Argument `nb_points_by_chunk`
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                kmeans_on_ts(ts_list=tsuid_list, nb_clusters=2, nb_points_by_chunk='a')
            # ----------------------------
            # Argument `spark`
            # ----------------------------
            # Wrong type (not NoneType or bool)
            msg = "Testing arguments : Error in testing `spark` type"
            with self.assertRaises(TypeError, msg=msg):
                kmeans_on_ts(ts_list=tsuid_list, nb_clusters=2, spark='a')
        finally:
            # Clean up database
            self.clean_up_db(tsuid_list)

    # ------------------ #
    # OUTPUTS TYPE TESTS #
    # ------------------ #
    def test_kmeans_sklearn_output_type(self):
        """
        Test the 'type' of the results for the sklearn version of the K-means algorithm on time series
        """
        try:
            # Used for reproducible results
            rand_state = 1
            # TS creation
            ts_list = gen_sax(1)
            # A typical K-Means model (for 'type' comparison)
            ref_model = KMeans(n_clusters=2, random_state=rand_state)

            # TODO : Appeler le wrapper
            model = fit_kmeans_sklearn_internal(data=ts_list, nb_clusters=2, random_state=rand_state)
            self.assertEqual(type(model[1]), dict, msg="ERROR: the output is not a dict")
            self.assertTrue((type(model[0]), type(ref_model)),
                            msg="ERROR: The output's model's type is not sklearn.cluster.k_means_")
        finally:
            self.clean_up_db(ts_list)

    def test_kmeans_spark_output_type(self):
        """
        Test the 'type' of the results for the Spark version of the K-means algorithm on time series
        """
        pass

    # ------------- #
    # RESULTS TESTS #
    # ------------- #
    # TODO : Je ne dois tester que le wrapper : fit_kmeans_on_ts(). Code à réaménager en conséquence.
    def test_kmeans_sklearn_result(self):
        """
        Test the result obtained by the function kmeans_sklearn() on time series
        """
        # Used for reproducible results
        rand_state = 1
        try:
            # TS creation
            ts_list = gen_ts(1)

            # TODO : Je ne dois tester que le wrapper : fit_kmeans_on_ts(). Code à réaménager en conséquence.
            model = fit_kmeans_sklearn_internal(data=ts_list, nb_clusters=2, random_state=rand_state)

            condition = (model.labels_[0] == model.labels_[1] and model.labels_[2] == model.labels_[3])
            self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")

        finally:
            self.clean_up_db(ts_list)

    def test_kmeans_spark_result(self):
        """
        Test the results for the Spark version of the K-means algorithm on time series
        """
        # The same *random_state* is used for reproducible results
        # random_state = 1
        #
        # # 1/ Small data sets (for gen_sax(0) or gen_sax(1))
        # # -----------------------------------------------------------
        #
        # sax = gen_sax(0)
        # result = fit_kmeans(sax=sax, n_cluster=2, random_state=random_state)
        #
        # # We want the clustering {(a,b) ; (c,d)}
        #
        # # the result of the clustering for group 1
        # tsuid_group = result[1].get("C1").keys()  # ex: ['centroid_1', 'b', 'a']
        # # condition : the current group is (a,b) or (c,d)
        # condition = (("a" in tsuid_group) and ("b" in tsuid_group)) or (("c" in tsuid_group) and ("d" in tsuid_group))
        #
        # self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")
        #
        # # idem on group #2
        # tsuid_group = result[1].get("C2").keys()  # ex :['centroid_2', 'd', 'c']
        # condition = (("a" in tsuid_group) and ("b" in tsuid_group)) or (("c" in tsuid_group) and ("d" in tsuid_group))
        #
        # self.assertTrue(condition, msg="Error, the clustering is not efficient in trivial situations")
        #
        # # 2/ Test with a (huge) trivial data-set (for gen_sax(2))
        # # -----------------------------------------------------------
        # # We want the clustering {(a:i) ; (j:r) ; (s:1)}
        # # {alphabet[0:9] ; alphabet[9:18] ; alphabet[18:27] }
        #
        # sax = gen_sax(2)
        # n_cluster = 3
        # alphabet = list("abcdefghijklmnopqrstuvwxyz1")
        #
        # result = fit_kmeans(sax=sax, n_cluster=n_cluster, random_state=random_state)
        #
        # # For each group
        # for group in range(1, n_cluster):
        #     # List of the TSUID in the current group
        #     tsuid_group = result[1].get("C" + str(group)).keys()  # ex :['centroid_1', 'a', 'b',...,"i"]
        #
        #     # The group is the same than expected ?
        #     condition = (
        #         all(x in tsuid_group for x in alphabet[0:9]) or
        #         all(x in tsuid_group for x in alphabet[9:18]) or
        #         all(x in tsuid_group for x in alphabet[18:27])
        #     )
        #
        #     self.assertTrue(condition,
        #                     msg="Error, the clustering is not efficient in trivial situations (case n_cluster=3)")

    def test_diff_sklearn_spark(self):
        """
        Test the difference of result between the functions kmeans_spark() and kmeans_sklearn() on time series.
        The same result should be obtained with both ways.
        """
        pass

    # ---------------- #
    # ROBUSTNESS TESTS #
    # ---------------- #
    def test_kmeans_sklearn_robustness(self):
        """
        Test the robustness of the function kmeans_sklearn() on time series
        """
        pass

    def test_kmeans_spark_robustness(self):
        """
        Test the robustness of the function kmeans_spark() on time series
        """
        # The same *random_state* is used for reproducible results
        # random_state = 1
        # sax = gen_sax(1)
        #
        # # invalid sax type
        # with self.assertRaises(IkatsInputTypeError, msg="Error, invalid sax type."):
        #     fit_kmeans(sax=[1, 2], n_cluster=2, random_state=random_state)
        #     fit_kmeans(sax={"a": [1, 2], "b": [1, 2]}, n_cluster=2, random_state=random_state)
        #     fit_kmeans(sax={"a": {"paa": [1, 2]}, "b": [2, 3]}, n_cluster=2, random_state=random_state)
        #     fit_kmeans(sax={"a": {"paa": [1, 2]}, "b": {"paa": [2, 3, 3]}}, n_cluster=2, random_state=random_state)
        #     fit_kmeans(sax="paa", n_cluster=2, random_state=random_state)
        #
        # # invalid n_cluster type
        # with self.assertRaises(IkatsInputTypeError, msg="Error, invalid n_cluster type."):
        #     fit_kmeans(sax=sax, n_cluster="2", random_state=random_state)
        #     fit_kmeans(sax=sax, n_cluster=[2, 3, 4], random_state=random_state)
        #
        # # invalid n_cluster value
        # with self.assertRaises(IkatsInputTypeError, msg="Error, invalid n_cluster value."):
        #     fit_kmeans(sax=sax, n_cluster=-2, random_state=random_state)
        #     # (referenced in the code as a TypeError)
        #
        # # invalid random_state type
        # with self.assertRaises(IkatsInputTypeError, msg="Error, invalid random_state type"):
        #     fit_kmeans(sax=sax, n_cluster=2, random_state="random_state")
        #     fit_kmeans(sax=sax, n_cluster=2, random_state=[1, 3])
