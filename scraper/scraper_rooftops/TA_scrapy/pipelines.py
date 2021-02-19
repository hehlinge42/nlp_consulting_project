# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# class TaScrapyPipeline(object):
#     def process_item(self, item, spider):
#         return item

import logging
import logzero
from logzero import logger

import json
from TA_scrapy.items import RestoItem, ReviewRestoItem, UserItem
from itemadapter import ItemAdapter

class TaScrapyPipeline(object):

    def __init__(self):
        
        self.restaurants_folder = 'restaurants/'
        self.reviews_folder = 'reviews/'
        self.users_folder = 'users/'
        
        logger.info(' > Init TaScrapyPipeline')

    def open_spider(self, spider):

        self.file_reviews = open(spider.directory +  self.reviews_folder + 'reviews_id_' + str(abs(spider.id_resto)) + '.json', 'w+')
        logger.info(' Open file reviews.json')

        self.file_restaurants = open(spider.directory + self.restaurants_folder + 'restaurants_id_' + str(abs(spider.id_resto)) + '.json', 'w+')
        logger.info('Open file restaurants.json')

        if spider.scrap_user != 0:
            self.file_users = open(spider.directory + self.users_folder + 'users_id_' + str(abs(spider.id_resto)) + '.json', 'w+')
            logger.info('Open file users.json')

    def close_spider(self, spider):

        self.file_reviews.close()
        logger.info('Close file reviews.json')

        self.file_restaurants.close()
        logger.info('Close file restaurants.json')

        if spider.scrap_user != 0:
            self.file_users.close()
            logger.info('Close file users.json')

    def process_item(self, item, spider):

        if isinstance(item, RestoItem):
            return self.handle_resto_item(item, spider)

        if isinstance(item, ReviewRestoItem):
            return self.handle_review_item(item, spider)

        if isinstance(item, UserItem):
            return self.handle_user_item(item, spider)


    def handle_resto_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file_restaurants.write(line)
        return item


    def handle_review_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file_reviews.write(line)
        return item

    
    def handle_user_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file_users.write(line)
        return item