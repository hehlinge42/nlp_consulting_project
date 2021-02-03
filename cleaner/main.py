from cleaner import Cleaner



if __name__ == "__main__":
    # execute only if run as a script
    filename = '../scraper/scraped_data/reviews.json'
    cleaner = Cleaner()
    cleaner.set_file(filename)
    cleaner.clean(ngram=1)
    cleaner.get_word_count_by_restaurant()
    print(cleaner.df_word_frequency[20].head())
    cleaner.write_file()