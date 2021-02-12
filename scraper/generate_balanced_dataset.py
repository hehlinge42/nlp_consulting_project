import pandas as pd

if __name__ == "__main__":

    reviews = pd.read_json('scraped_data/merged_data/merged_reviews.json', lines=True)
    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(by_rating['review_id'])
    balanced_reviews = reviews.groupby("rating").sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv('scraped_data/merged_data/balanced_reviews.csv', sep='#', index_label='review_id')