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
import time

# Import sk Kmeans
import sklearn.cluster
# Import pyspark kmeans
import pyspark.ml.clustering

from ikats.core.resource.api import IkatsApi
from ikats.core.library.exception import IkatsConflictError
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
    :rtype: dict
    """
    # Build TS identifier
    fid = 'UNIT_TEST_K-MEANS_%s' % ts_id
    # -----------------------------------------------------------------------------
    # CASE 1: 4 TS divided in 2 groups of 2 with 2 points per TS. shape = (4, 2, 2)
    # -----------------------------------------------------------------------------
    if ts_id == 1:
        time1 = list(range(1, 3))
        ts_a1 = [7, 3]
        ts_a2 = list(ts_a1 + np.random.normal(0, 1, size=len(ts_a1)))
        ts_b1 = [14, 13]
        ts_b2 = list(ts_b1 + np.random.normal(0, 1, size=len(ts_b1)))
        ts_content = np.array([np.array([time1, ts_a1]).T,
                               np.array([time1, ts_a2]).T,
                               np.array([time1, ts_b1]).T,
                               np.array([time1, ts_b2]).T], np.float64)
    # -------------------------------------------------------------------------------
    # CASE 2: 4 TS divided in 2 groups of 2 with 10 points per TS. shape = (4, 10, 2)
    # -------------------------------------------------------------------------------
    elif ts_id == 2:
        time1 = list(range(1, 11))
        ts_a1 = [7, 3, 4, 9, 5, 6, 1, 0, 1, 2]
        ts_a2 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_b1 = [14, 13, 15, 15, 20, 30, 42, 43, 47, 50]
        ts_b2 = ts_b1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_content = np.array([np.array([time1, ts_a1]).T,
                               np.array([time1, ts_a2]).T,
                               np.array([time1, ts_b1]).T,
                               np.array([time1, ts_b2]).T], np.float64)
    # -------------------------------------------------------------------------------
    # CASE 3: 15 TS divided in 3 groups of 5 with 2 points per TS. shape = (15, 2, 2)
    # -------------------------------------------------------------------------------
    elif ts_id == 3:
        time1 = list(range(1, 3))
        ts_a1 = [7, 3]
        ts_a2 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a3 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a4 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a5 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_b1 = [14, 13]
        ts_b2 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b3 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b4 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b5 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_c1 = [50, 55]
        ts_c2 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c3 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c4 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c5 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_content = np.array([np.array([time1, ts_a1]).T, np.array([time1, ts_a2]).T, np.array([time1, ts_a3]).T,
                               np.array([time1, ts_a4]).T, np.array([time1, ts_a5]).T, np.array([time1, ts_b1]).T,
                               np.array([time1, ts_b2]).T, np.array([time1, ts_b3]).T, np.array([time1, ts_b4]).T,
                               np.array([time1, ts_b5]).T, np.array([time1, ts_c1]).T, np.array([time1, ts_c2]).T,
                               np.array([time1, ts_c3]).T, np.array([time1, ts_c4]).T, np.array([time1, ts_c5]).T],
                              np.float64)
    # ---------------------------------------------------------------------------------
    # CASE 4: 15 TS divided in 3 groups of 5 with 10 points per TS. shape = (15, 10, 2)
    # ---------------------------------------------------------------------------------
    elif ts_id == 4:
        time1 = list(range(1, 11))
        ts_a1 = [7, 3, 4, 9, 5, 6, 1, 0, 1, 2]
        ts_a2 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a3 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a4 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_a5 = ts_a1 + np.random.normal(0, 1, size=len(ts_a1))
        ts_b1 = [14, 13, 15, 15, 20, 30, 42, 43, 47, 50]
        ts_b2 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b3 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b4 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_b5 = ts_b1 + np.random.normal(0, 1, size=len(ts_b1))
        ts_c1 = [50, 55, 54, 52, 59, 57, 51, 55, 52, 58]
        ts_c2 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c3 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c4 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_c5 = ts_c1 + np.random.normal(0, 1, size=len(ts_c1))
        ts_content = np.array([np.array([time1, ts_a1]).T, np.array([time1, ts_a2]).T, np.array([time1, ts_a3]).T,
                               np.array([time1, ts_a4]).T, np.array([time1, ts_a5]).T, np.array([time1, ts_b1]).T,
                               np.array([time1, ts_b2]).T, np.array([time1, ts_b3]).T, np.array([time1, ts_b4]).T,
                               np.array([time1, ts_b5]).T, np.array([time1, ts_c1]).T, np.array([time1, ts_c2]).T,
                               np.array([time1, ts_c3]).T, np.array([time1, ts_c4]).T, np.array([time1, ts_c5]).T],
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

        try:
            # Create a TS, dict with keys: 'tsuid', 'funcId', 'status', 'reponseStatus', 'summary', 'errors',
            # 'numberOfSuccess'
            IkatsApi.ts.create_ref(current_fid)
        except IkatsConflictError:
            # If the TS already exists, we add timestamp to have an unique FID
            current_fid = '%s_%s_' % (current_fid, int(time.time() * 1000))
            IkatsApi.ts.create_ref(current_fid)

        my_ts = IkatsApi.ts.create(fid=current_fid, data=np.array(ts_content)[i, :, :])
        # Create metadatas 'qual_nb_points', 'name' and 'funcId'
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='qual_nb_points', value=len(ts_content), force_update=True)
        IkatsApi.md.create(tsuid=my_ts['tsuid'], name='metric', value='metric_%s' % ts_id, force_update=True)
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
    def test_kmeans_sklearn_output_type(self):
        """
        Test the 'type' of the results for the sklearn version of the K-means algorithm on time series
        """
        try:
            rand_state = 1
            # TS creation
            my_ts = gen_ts(1)
            # Fit the model
            # noinspection PyTypeChecker
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)
            # Checks of outputs' type
            self.assertEqual(type(result_kmeans[1]), dict, msg="ERROR: This output's type is not 'dict'")
            self.assertTrue(type(result_kmeans[0]) is sklearn.cluster.k_means_.KMeans,
                            msg="ERROR: This output's type is not 'sklearn.cluster.k_means_'")
        finally:
            self.clean_up_db(my_ts)

    def test_kmeans_spark_output_type(self):
        """
        Test the 'type' of the results for the Spark version of the K-means algorithm on time series
        """
        try:
            rand_state = 1
            # TS creation
            my_ts = gen_ts(1)
            # Fit the model
            # noinspection PyTypeChecker
            result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)
            # Checks of outputs' type
            self.assertEqual(type(result_kmeans[1]), dict, msg="ERROR: This output's type is not 'dict'")
            self.assertTrue(type(result_kmeans[0]) is pyspark.ml.clustering.KMeansModel,
                            msg="ERROR: This output's type is not 'pyspark.ml.clustering.KMeansModel'")
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
        try:
            # TS creation
            my_ts = gen_ts(1)
            # Fit the model
            model, result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)

            # TODO: tester le fait que 2 TSUID doivent être ds le même dict
            condition = (result_kmeans == np.array([0, 0, 1, 1], dtype=np.int32))
            self.assertEqual(np.all(condition), True, msg="ERROR: The obtained clustering is not the one expected")
        finally:
            self.clean_up_db(my_ts)

    def test_kmeans_spark_result(self):
        """
        Test the result obtained for the sklearn version of the K-means algorithm on time series
        """
        # Used for reproducible results
        rand_state = 1
        try:
            # TS creation
            my_ts = gen_ts(1)
            # Fit the model
            # noinspection PyTypeChecker
            # TODO: cf fonction précédente
            model, result_kmeans = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)
            condition = (result_kmeans == np.array([0, 0, 1, 1], dtype=np.int32))
            self.assertEqual(np.all(condition), True, msg="ERROR: The obtained clustering is not the one expected")
        finally:
            self.clean_up_db(my_ts)

    def test_diff_sklearn_spark(self):
        """
        Test the difference of result between the functions kmeans_spark() and kmeans_sklearn() on time series.
        The same result should be obtained with both ways.
        """
        # Used for reproducible results
        rand_state = 1
        try:
            # TS creation
            my_ts = gen_ts(1)
            # Fit the model
            _, result_sklearn = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=False, random_state=rand_state)
            _, result_spark = fit_kmeans_on_ts(ts_list=my_ts, nb_clusters=2, spark=True, random_state=rand_state)
            # TODO: Problème éventuel de label switching. Le lancer plusieurs fois !
            condition = (result_sklearn == result_spark)
            # TODO: vérifier que MDS est bien reproductible avec la même seed
            msg = "ERROR: The obtained clustering is not the one expected.\n" \
                  "Result sklearn = {} \n" \
                  "Result Spark ={}".format(result_sklearn, result_spark)
            self.assertEqual(np.all(condition), True, msg=msg)
        finally:
            self.clean_up_db(my_ts)
