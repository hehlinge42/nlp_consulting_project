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

merged_file = os.path.join(merged_data_dir, "merged_restaurants.json")

logger.info("Merging restaurant files")
merge_files(os.path.join(scraped_data_dir, 'restaurants'), merged_file)

already_scraped_list = [10216, 13639, 12755, 14787, 14657, 10068, 13460, 13836, 12125, 9796, 11264]
all_restaurant_path = os.path.join(os.getcwd(), 'scraper', 'scraper_restaurants', 'scraped_data', 'restaurants', 'restaurants_run_1.json')
all_restaurants = pd.read_json(all_restaurant_path, lines=True)
all_restaurants.set_index(['restaurant_id'], inplace=True)
rooftops = all_restaurants.loc[already_scraped_list, :]

logger.info(f"Concatenating with {len(rooftops)} already scraped restaurants")

merged_rooftops = pd.read_json(merged_file, lines=True)
merged_rooftops.set_index(['restaurant_id'], inplace=True)
merged_rooftops = pd.concat([merged_rooftops, rooftops])
merged_rooftops.to_csv(os.path.join(os.getcwd(), 'scraper', 'scraper_rooftops', 'scraped_data', 'merged_data', 'merged_rooftops.csv'))