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
        
        logzero.loglevel(logging.WARNING)
        logger.warn(' > Init TaScrapyPipeline')

    def open_spider(self, spider):

        logger.warn(' Open file reviews.json')
        self.file_reviews = open(spider.directory + 'reviews.json', 'w')

        logger.warn('Open file restaurants.json')
        self.file_restaurants = open(spider.directory + 'restaurants.json', 'w')

        if spider.scrap_user != 0:
            logger.warn('Open file users.json')
            self.file_users = open(spider.directory + 'users.json', 'w')

    def close_spider(self, spider):

        self.file_reviews.close()
        logger.warn('Close file reviews.json')

        self.file_restaurants.close()
        logger.warn('Close file restaurants.json')

        if spider.scrap_user != 0:
            self.file_users.close()
            logger.warn('Close file users.json')

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