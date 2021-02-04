from cleaner import Cleaner

if __name__ == "__main__":

    filename = '../scraper/scraped_data/reviews.json'
    cleaner = Cleaner()
    cleaner.set_file(filename)
    cleaner.preprocessing(ngram=2)
    cleaner.get_word_count_by_restaurant()
    cleaner.save_tokenized_corpus('./cleaned_data/')
    cleaner.write_tfidfs()
    cleaner.save_wordclouds()