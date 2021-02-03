from cleaner import Cleaner



if __name__ == "__main__":
    # execute only if run as a script
    filename = '../scraper/scraped_data/reviews.json'
    cleaner = Cleaner()
    cleaner.set_file(filename)
    cleaner.clean(ngram=2)
    cleaner.get_word_count_by_restaurant()
    cleaner.write_tokenized_reviews()
    cleaner.write_tfidfs()
    cleaner.save_wordclouds()