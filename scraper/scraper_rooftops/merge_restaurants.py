import glob
import json
from logzero import logger
import logzero
import pandas as pd
import os

path_list = os.getcwd().split(os.sep)
target_index = path_list.index('nlp_consulting_project')
path_list = path_list[:target_index + 1]
os.chdir(os.path.join(os.sep, *path_list))


def merge_files(directory, output_file):
    
    existing_files = glob.glob(directory + "/*.json")
    logger.info(f"Merging {len(existing_files)} existing_files")

    with open(output_file, "w+") as merged_file:
        for existing_file in existing_files:
            with open(existing_file, "r+") as child_file:
                for line in child_file:
                    merged_file.write(line)


scraped_data_dir = os.path.join(os.getcwd(), 'scraper', 'scraper_rooftops', 'scraped_data')
merged_data_dir = os.path.join(scraped_data_dir, 'merged_data')

if not os.path.exists(merged_data_dir):
    logger.warn(f"Creating directory {merged_data_dir}")
    os.makedirs(merged_data_dir)

logger.info("Merging restaurant files")
merge_files(os.path.join(scraped_data_dir, 'restaurants'), os.path.join(merged_data_dir, "merged_restaurants.json"))