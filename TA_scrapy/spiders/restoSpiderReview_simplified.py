# Logging packages
import logging
import logzero
from logzero import logger

# Scrapy packages
import scrapy
from TA_scrapy.items import ReviewRestoItem, RestoItem     # you can use it if you want but it is not mandatory
from TA_scrapy.spiders import get_info          # package where you can write your own functions


class RestoReviewSpider(scrapy.Spider):
    name = "RestoReviewSpider"

    def __init__(self, *args, **kwargs): 
        super(RestoReviewSpider, self).__init__(*args, **kwargs)

        # Set logging level
        logzero.loglevel(logging.WARNING)
        # logging.disable(logging.WARNING)

        # To track the evolution of scrapping
        self.main_nb = 0
        self.resto_nb = 0
        self.review_nb = 0
     #   self.old_id_resto = 0

    def start_requests(self):
        """ Give the urls to follow to scrapy
        - function automatically called when using "scrapy crawl my_spider"
        """

        # Basic restaurant page on TripAdvisor GreaterLondon
        url = 'https://www.tripadvisor.co.uk/Restaurants-g191259-Greater_London_England.html'
        yield scrapy.Request(url=url, callback=self.parse)

   
    def parse(self, response):
        """MAIN PARSING : Start from a classical reastaurant page
            - Usually there are 30 restaurants per page
            - 
        """
        # Display a message in the console
        logger.warn(' > PARSING NEW MAIN PAGE OF RESTO ({})'.format(self.main_nb))
        self.main_nb += 1

        # Get the list of the 30 restaurants of the page
        restaurant_urls = get_info.get_urls_resto_in_main_search_page(response)
        
        # For each url : follow restaurant url to get the reviews
        for restaurant_url in restaurant_urls:
            logger.warn('> New restaurant detected : {}'.format(restaurant_url))
            self.resto_nb += 1
            yield response.follow(url=restaurant_url, callback=self.parse_resto, cb_kwargs=dict(restaurant_id=self.resto_nb))
            yield response.follow(url=restaurant_url, callback=self.parse_review_page, cb_kwargs=dict(restaurant_id=self.resto_nb))
            

        # Get next page information
        next_page, next_page_number = get_info.get_urls_next_list_of_restos(response)
        
        # Follow the page if we decide to
        if get_info.go_to_next_page(next_page, next_page_number, max_page=1):
            yield response.follow(next_page, callback=self.parse)

    def parse_resto(self, response, restaurant_id):

        logger.warn('Entering parse_resto = ')

        xpath_name = '//h1[@class="_3a1XQ88S"]/text()'
        xpath_nb_reviews = '//div[@class="_1ud-0ITN"]/span/a/span/text()'
        xpath_price_cuisine = '//span[@class="_13OzAOXO _34GKdBMV"]//a/text()'
        xpath_phone_number = '//div[@class="_1ud-0ITN"]/span/span/span/a/text()'
        xpath_website = '//a[@class="_2wKz--mA _15QfMZ2L"]/@data-encoded-url'
        xpath_ranking = '//span[@class="_13OzAOXO _2VxaSjVD"]/a/span/text()'
        xpath_rating = '//div[@class="_1ud-0ITN"]/span/a/svg/@title'
        xpath_address = '//span[@class="_13OzAOXO _2VxaSjVD"]/span[1]/a/text()'
        
        resto_item = RestoItem()
        resto_item['restaurant_id'] = restaurant_id
        resto_item['name'] = response.xpath(xpath_name).get()
        resto_item['nb_reviews'] = response.xpath(xpath_nb_reviews).get()
        price_cuisine = response.xpath(xpath_price_cuisine).getall()
        resto_item['cuisine'] = price_cuisine[1:] # array
        
        resto_item['website'] = response.xpath(xpath_website).get()
        resto_item['ranking'] = response.xpath(xpath_ranking).get()
        resto_item['phone_number'] = response.xpath(xpath_phone_number).get()
        resto_item['rating'] = response.xpath(xpath_rating).get().split()[0]
        resto_item['address'] = response.xpath(xpath_address).get()

        raw_price = price_cuisine[0]
        min_price, max_price = raw_price.split(' - ')
        resto_item['min_price'] = len(min_price)
        resto_item['max_price'] = len(max_price)
        logger.warn('resto_item = ', resto_item)

        yield resto_item


    def parse_review_page(self, response, restaurant_id):
        """SECOND PARSING : Given a review page, gets each review url and get to parse it
            - Usually there are 10 reviews per page
        """
        logger.warn(' > PARSING NEW REVIEW PAGE ({})'.format(restaurant_id - 1))
            
        # Get the list of reviews on the page
        urls_review = get_info.get_urls_reviews_in_restaurant_page(response)

        # For each review open the link and parse it into the parse_review method
        for url_review in urls_review:
             yield response.follow(url=url_review, callback=self.parse_review, cb_kwargs=dict(restaurant_id=restaurant_id))

        # Get next page information
        next_page, next_page_number = get_info.get_urls_next_list_of_reviews(response)
        
        # Follow the page if we decide to
        if get_info.go_to_next_page(next_page, next_page_number, printing=False, max_page=2):
            yield response.follow(next_page, callback=self.parse_review_page, cb_kwargs=dict(restaurant_id=restaurant_id))


    def parse_review(self, response, restaurant_id):
        """FINAL PARSING : Open a specific page with review and client opinion
            - Read these data and store them
            - Get all the data you can find and that you believe interesting
        """
        logger.warn(' > PARSING NEW REVIEW ({})'.format(self.review_nb))

        # Count the number of review scrapped
        self.review_nb += 1

        xpath_username = '//div[@class="username mo"]/span/text()'
        xpath_date_of_visit = '//div[@class="prw_rup prw_reviews_stay_date_hsx"]/text()'
        xpath_rating = '//div[@class="rating reviewItemInline"]/span[1]/@class'
        xpath_title = '//div[@class="quote"]/a/span/text()'
        xpath_comment = '//div[@class="entry"]/p/text()'
        
        # You can store the scrapped data into a dictionnary or create an Item in items.py (cf XActuItem and scrapy documentation)
        review_item = ReviewRestoItem()
        review_item['review_id'] = self.review_nb
        review_item['restaurant_id'] = restaurant_id
        review_item['username'] = response.xpath(xpath_username).get()
        review_item['date_of_visit'] = response.xpath(xpath_date_of_visit).get()
        review_item['rating'] = response.xpath(xpath_rating).get()[-2]
        review_item['title'] = response.xpath(xpath_title).get()
        review_item['comment'] = response.xpath(xpath_comment).get()
        
        yield review_item
