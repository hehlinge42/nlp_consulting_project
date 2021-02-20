# Databricks notebook source
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import DenseMatrix

from logzero import logger
import logzero
import logging

import numpy as np


def spark_lsi(input, output):

    logger.info('Reading textFile')
    text_file = sc.textFile(input)

    text_file_split = text_file.map(lambda x: x.split())
    text_file_cast = text_file_split.map(lambda x: (int(x[0]), {int(x[1]): float(x[2])}))
    grouped_by = text_file_cast.groupByKey().mapValues(list)
    grouped_by2 = grouped_by.map(lambda x: (x[0], 18005, dict(pair for d in x[1] for pair in d.items())))
    vectors = grouped_by2.map(lambda x: (x[0], Vectors.sparse(x[1], x[2])))
    vectors = vectors.sortByKey()
    vectors_no_key = vectors.map(lambda x: x[1])
    vectors_list = vectors_no_key.collect()

    rows = sc.parallelize(vectors_list)
    mat = RowMatrix(rows)

    svd = mat.computeSVD(517, computeU=True)
    U = svd.U       # The U factor is a RowMatrix.
    s = svd.s       # The singular values are stored in a local dense vector.
    V = svd.V       # The V factor is a local dense matrix.

    s_matrix = DenseMatrix(len(s), len(s), np.diag(s).ravel("F"))

    truncated_svd = U.multiply(s_matrix)

    spark_df = truncated_svd.rows.map(lambda v: v.toArray().tolist()).toDF()
    display(spark_df)

    pandas_df = spark_df.toPandas()
    pandas_df.to_csv(output)


if __name__ == 'main':

    spark_lsi('./cleaner/cleaned_data/restaurant_tfidf_sparse.txt', './embedder/embedded_data/spark_lsi_py.csv')