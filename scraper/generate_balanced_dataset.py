import pandas as pd
from logzero import logger
import logzero
import os
import argparse

if __name__ == "__main__":

    logger.info(' > Generating balanced dataset.')

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

    parser = argparse.ArgumentParser(description="Generator of reviews sample with equally balanced ratings")
    parser.add_argument('-s', '--size', type=int, default=18000, help='Caps the number of reviews to be processed')
    args = parser.parse_args()

    reviews = pd.read_json(os.path.join(os.getcwd(), 'scraper', 'scraped_data', 'merged_data', 'merged_reviews.json'), lines=True)
    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(args.size // 5, min(by_rating['review_id']))
    balanced_reviews = reviews.groupby("rating").sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv(os.path.join(os.getcwd(), 'scraper', 'scraped_data', 'merged_data', 'balanced_reviews.csv'), sep='#', index_label='review_id')