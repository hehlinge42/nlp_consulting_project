import argparse
from cleaner import Cleaner
from helpers import save_wordcloud, save_tfidf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cleaner and tokenizer of raw text stored as json file")
    parser.add_argument('-f', '--files', nargs="*", type=str, help='path to the files to be cleaned')
    parser.add_argument('-d', '--debug', help="prints intermediary logs", action="store_true")
    parser.add_argument('-s', '--early_stop', type=int, default=-1, help='Caps the number of reviews to be processed')
    args = parser.parse_args()

    filenames = args.files

    if args.early_stop == -1:
        args.early_stop = None

    cleaner = Cleaner(debug=args.debug, early_stop=args.early_stop)

    for file in filenames:
        cleaner.set_file(file)
        cleaner.preprocessing(ngram=2)

        cleaner.save_tokenized_corpus('./cleaned_data/', 'tokenized_corpus.json', cleaner.tokenized_corpus)
        cleaner.save_files('./cleaned_data/restaurant_wordclouds/', save_wordcloud, mask_path='assets/capgemini.jpg')
        cleaner.save_files('./cleaned_data/restaurant_word_frequencies/', save_tfidf)
        cleaner.save_tokenized_corpus('./cleaned_data/', 'restaurants_tfidf.csv', cleaner.corpus_tfidf, file_type='csv')