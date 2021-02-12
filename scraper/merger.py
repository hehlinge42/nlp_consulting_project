import glob
import json
from logzero import logger
import logzero
import pandas as pd
import os

def offset_ids():

    existing_restaurants = glob.glob("./scraped_data/restaurants/*.json")
    existing_reviews = glob.glob("./scraped_data/reviews/*.json")

    logger.info(f"FINDING RESTAURANTS FILES: {existing_restaurants}")
    output_file_resto = "./scraped_data/restaurants/modified_restaurants_"
    output_file_review = "./scraped_data/reviews/modified_reviews_"

    resto_offset = 0
    nb_lines_file_resto = 0
    review_offset = 0
    nb_lines_file_review = 0

    couples = zip(existing_restaurants, existing_reviews)

    for idx, couple in enumerate(couples):
        resto_file, review_file = couple
        logger.warn(f"{resto_file} \n {review_file}")
        
        nb_lines_file_resto = 0
        nb_lines_file_review = 0
        modified_file_resto = open(output_file_resto + str(idx + 1) + '.json', "a+")
        modified_file_review = open(output_file_review + str(idx + 1) + '.json', "a+")
        file_resto = open(resto_file, "r+")
        file_review = open(review_file, "r+")
    
        for line in file_resto:
            data = json.loads(line)
            data["restaurant_id"
    ] += resto_offset
            json.dump(data, modified_file_resto)
            modified_file_resto.write("\n")
            nb_lines_file_resto += 1

        for line in file_review:
            data = json.loads(line)
            data["restaurant_id"] += resto_offset
            data["review_id"] += review_offset
            json.dump(data, modified_file_review)
            modified_file_review.write("\n")
            nb_lines_file_review += 1

        review_offset += nb_lines_file_review
        resto_offset += nb_lines_file_resto
        
        modified_file_resto.close()
        modified_file_review.close()
        file_resto.close()
        file_review.close()

def merge_files(directory, output_file):
    
    existing_files = glob.glob(directory + "*.json")

    with open(output_file, "w+") as merged_file:
        for existing_file in existing_files:
            with open(existing_file, "r+") as child_file:
                for line in child_file:
                    merged_file.write(line)


# try:
#     os.mkdir("./scraped_data/merged_data")
# except OSError:
#     logger.warn("OSError: directory already exists")

# # offset_ids()
# merge_files("./scraped_data/restaurants/", "./scraped_data/merged_data/merged_restaurants.json")
# merge_files("./scraped_data/reviews/", "./scraped_data/merged_data/merged_reviews.json")
# merge_files("./scraped_data/users/", "./scraped_data/merged_data/merged_users.json")