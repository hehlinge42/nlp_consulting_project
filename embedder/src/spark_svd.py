import sys
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext
import pandas as pd

from scipy.sparse import load_npz

from logzero import logger
import logzero
import logging

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# spark.conf.set("spark.sql.execution.arrow.enabled", "true")

logger.info("Loading sparse matrix")
document_term_sparse_matrix = load_npz('../../cleaner/cleaned_data/restaurant_tfidf_sparse.npz')

logger.info(f"Converting to dense matrix")
dense_matrix = document_term_sparse_matrix.todense()

logger.info(f"Creating pd.DataFrame")
review_id = pd.read_csv('../../cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv', index_col='review_id')
colnames = pd.read_csv('../../cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv')
document_term_matrix = pd.DataFrame(dense_matrix, index=review_id.index.values, columns=colnames['colnames'])
logger.info(f"Shape of DataFrame is {document_term_matrix.shape}")

logger.info(f"Converting to spark.DataFrame")
spark_df = spark.createDataFrame(document_term_matrix)
logger.info(f"Printing schema")
spark_df.show(30, truncate=False)