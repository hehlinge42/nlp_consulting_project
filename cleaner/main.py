from cleaner import Cleaner



if __name__ == "__main__":
    # execute only if run as a script
    filename = '../scraper/scraped_data/reviews.json'
    cleaner = Cleaner()
    cleaner.set_file_info(filename)
    cleaner.clean()
    cleaner.write_file()