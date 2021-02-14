### helper functions

from logzero import logger
import logging
import logzero
import glob
import pandas as pd

logzero.loglevel(logging.DEBUG)

def get_document_term_per_restaurant(directory, nb_resto=10):
    
    document_term_matrices = {}
    tfidfs_files = glob.glob(directory + '*.csv')

    for idx, tfidfs_file in enumerate(tfidfs_files):
        logger.info(f' > COLLECTING FILE {tfidfs_file}')
        tfidf_df = pd.read_csv(tfidfs_file, index_col='review_id')
        filename = tfidfs_file.split('/')[-1]
        id_resto = filename.split('_')[0]
        document_term_matrices[id_resto] = tfidf_df
        if idx >= nb_resto:
            break

    return document_term_matrices

    def get_representative_document(tfidf_dict, min_reviews=500):

        for idx, tfidf_df in enumerate(tfidf_dict):
            if len(tfidfs_df) >= 500:
                logger.info(f"FOUND TFIDF FILE OF SIZE {len(tfidfs_df)} FOR RESTAURANT ID {idx}")
                return tfidf_df

        logger.critical(f"FOUND NO TFIDF WITH SUFFICIENT LEN {min_reviews}")
        return None
