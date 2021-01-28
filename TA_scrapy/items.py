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
    username = scrapy.Field()
    date_of_visit = scrapy.Field()
    rating = scrapy.Field()
    title = scrapy.Field()
    comment = scrapy.Field()


class RestoItem(scrapy.Item):
        
    restaurant_id = scrapy.Field()
    name = scrapy.Field()
    nb_reviews = scrapy.Field()
    min_price = scrapy.Field()
    max_price = scrapy.Field()
    cuisine = scrapy.Field()
    address = scrapy.Field()
    phone_number = scrapy.Field()
    website = scrapy.Field()
    ranking = scrapy.Field()
    rating = scrapy.Field()

class UserItem(scrapy.Item):

    username = scrapy.Field()
    fullname = scrapy.Field()
    date_joined = scrapy.Field()
    nb_contributions = scrapy.Field()
    nb_followers = scrapy.Field()
    nb_following = scrapy.Field()