import dask
from dask_ml.decomposition import TruncatedSVD
import dask.array as da
from dask.distributed import Client, LocalCluster

import numpy as np

from scipy.sparse import load_npz

from logzero import logger
import logzero
import pandas as pd
import logging

# import dask.multiprocessing
# dask.config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler


def get_best_nb_component(da_dense_matrix, var_threshold=0.5, max_components=1000):
    
    total_variance = 0.0
    n_components = 0

    logger.info(f"Starting get_best_nb_component with {max_components} components")
    large_svd = TruncatedSVD(n_components=max_components)
    large_lsa = large_svd.fit_transform(da_dense_matrix)
    logger.info("Ending get_best_nb_component")

    for explained_variance in large_svd.explained_variance_ratio_:
        total_variance += explained_variance
        n_components += 1

        if total_variance >= var_threshold:
            break
            
    # Return the number of components
    return n_components


if __name__ == '__main__':

    cluster = LocalCluster()
    client = Client(cluster)
    logger.info(f"Cluster scheduler: {cluster.scheduler}")
    logger.info(f"Cluster workers: {cluster.workers}")
    logger.info(f"Client: {client}")
    logger.info(client.scheduler_info()['services'])

    logger.info(f"Loading npz")
    document_term_sparse_matrix = load_npz('../../cleaner/cleaned_data/restaurant_tfidf_sparse.npz')

    logger.info(f"Converting sparse into dense matrix")
    dense_matrix = document_term_sparse_matrix.todense()
    dense_matrix = np.array(dense_matrix)

    logger.info(f"Converting into dask array")
    da_dense_matrix = da.from_array(dense_matrix, chunks=({0: -1, 1: 'auto'}))
    logger.info(f"Dense matrix has shape {da_dense_matrix.shape}")
    
    # review_id = pd.read_csv('../../cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv', index_col='review_id')

    best_nb_component = get_best_nb_component(da_dense_matrix)
    logger.info(f"BEST COMPONENT is {best_nb_component}")

    logger.info(f"Starting TruncatedSVD")
    svd = TruncatedSVD(n_components=n_components)
    svd_output = svd.fit_transform(da_dense_matrix)

    logger.info(f"Converting to DataFrame")
    embedded_document = pd.DataFrame(svd_output)
    # embedded_document.index = review_id.index.values

    logger.info(f"Writing output file")
    embedded_document.to_csv('../embedded_data/dask_test.csv')


