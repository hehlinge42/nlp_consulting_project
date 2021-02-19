# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
import pandas as pd

class RestoItem(scrapy.Item):
        
    restaurant_id = scrapy.Field()
    resto_TA_url = scrapy.Field()
    name = scrapy.Field()
    nb_reviews = scrapy.Field()
    min_price = scrapy.Field()
    max_price = scrapy.Field()
    cuisine = scrapy.Field()
    address = scrapy.Field()
    phone_number = scrapy.Field()
    website = scrapy.Field()
    menu = scrapy.Field()
    ranking = scrapy.Field()
    rating = scrapy.Field()