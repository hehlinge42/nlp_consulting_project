import pandas as pd
from logzero import logger
import logzero
import os

if __name__ == "__main__":

    logger.info(f"In generate_balanced_dataset.py, pwd returns {os.getcwd()}")

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

    logger.info(f"In generate_balanced_dataset.py, changing working directory to {os.getcwd()}")

    reviews = pd.read_json(os.path.join(os.getcwd(), 'scraper', 'scraped_data', 'merged_data', 'merged_reviews.json'), lines=True)
    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(by_rating['review_id'])
    balanced_reviews = reviews.groupby("rating").sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv(os.path.join(os.getcwd(), 'scraper', 'scraped_data', 'merged_data', 'balanced_reviews.csv'), sep='#', index_label='review_id')