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
from TA_scrapy.items import RestoItem
from itemadapter import ItemAdapter

class TaScrapyPipeline(object):

    def __init__(self):
        self.restaurants_folder = 'bulk_restaurants/'
        logger.info(' > Init TaScrapyPipeline')

    def open_spider(self, spider):
        self.file_restaurants = open(spider.directory + self.restaurants_folder + 'restaurants_run_' + str(spider.next_file_id) + '.json', 'w+')
        logger.info('Open file restaurants.json')

    def close_spider(self, spider):
        self.file_restaurants.close()
        logger.info('Close file restaurants.json')

    def process_item(self, item, spider):
        return self.handle_resto_item(item, spider)

    def handle_resto_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file_restaurants.write(line)
        return item