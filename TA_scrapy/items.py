# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

import pandas as pd

class ReviewRestoItem(scrapy.Item):
        
    review_id = scrapy.Field()
    restaurant_id = scrapy.Field()
    restaurant_name = scrapy.Field()
    rating = scrapy.Field()
    user = scrapy.Field()
    date_of_visit = scrapy.Field()
    title = scrapy.Field()
    comment = scrapy.Field()


class RestoItem(scrapy.Item):
        
    restaurant_id = scrapy.Field()
    name = scrapy.Field()
    nb_reviews = scrapy.Field()
    price_range = scrapy.Field()
    cuisine_type = scrapy.Field()
    city = scrapy.Field()
    address = scrapy.Field()
    ranking = scrapy.Field()
    rating = scrapy.Field()