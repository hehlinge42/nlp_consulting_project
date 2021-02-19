import argparse
from cleaner import Cleaner
from helpers import save_wordcloud, save_tfidf
import os

if __name__ == "__main__":

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

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
        cleaner.set_file(file, filetype=file.split('.')[-1])
        cleaner.preprocessing(ngram=2)

        cleaner.save_tokenized_corpus('./cleaner/cleaned_data/', 'tokenized_corpus.json', cleaner.tokenized_corpus)
        cleaner.save_files('./cleaner/cleaned_data/restaurants_wordclouds/', save_wordcloud, mask_path='cleaner/assets/capgemini.jpg')
        cleaner.save_files('./cleaner/cleaned_data/restaurants_tfidf/', save_tfidf)
        # cleaner.save_tokenized_corpus('./cleaned_data/', 'restaurants_tfidf.csv', cleaner.corpus_tfidf, file_type='csv')
        cleaner.save_sparse_matrix('./cleaner/cleaned_data/restaurant_tfidf_sparse.npz', 
                                    './cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv',
                                    './cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv',
                                    './cleaner/cleaned_data/restaurant_tfidf_sparse.txt')