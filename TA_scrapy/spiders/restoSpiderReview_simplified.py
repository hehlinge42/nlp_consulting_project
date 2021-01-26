# Logging packages
import logging
import logzero
from logzero import logger

# Scrapy packages
import scrapy
from TA_scrapy.items import ReviewRestoItem     # you can use it if you want but it is not mandatory
from TA_scrapy.spiders import get_info          # package where you can write your own functions


class RestoReviewSpider(scrapy.Spider):
    name = "RestoReviewSpider"

    def __init__(self, *args, **kwargs): 
        super(RestoReviewSpider, self).__init__(*args, **kwargs)

        # Set logging level
        logzero.loglevel(logging.WARNING)
        logging.disable(logging.WARNING)

        # To track the evolution of scrapping
        self.main_nb = 0
        self.resto_nb = 0
        self.review_nb = 0

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
        logger.warn(' > RESTAURANT URLS ({})'.format(len(restaurant_urls)))
        
        # For each url : follow restaurant url to get the reviews
        for restaurant_url in restaurant_urls:
            self.resto_nb += 1
            logger.warn('> New restaurant detected : {}'.format(restaurant_url))
            yield response.follow(url=restaurant_url, callback=self.parse_resto, cb_kwargs=dict(id_resto=self.resto_nb))

        # Get next page information
        next_page, next_page_number = get_info.get_urls_next_list_of_restos(response)
        logger.warn('> Next page of restaurants : {}'.format(next_page))
        
        # Follow the page if we decide to
        if get_info.go_to_next_page(next_page, next_page_number, printing=False, max_page=3):
            logger.warn('> Yielding new page')
            yield response.follow(next_page, callback=self.parse)


    def parse_resto(self, response, id_resto):
        """SECOND PARSING : Given a restaurant, get each review url and get to parse it
            - Usually there are 10 comments per page
        """
        logger.warn(' > PARSING NEW RESTO PAGE ({})'.format(id_resto))
        # if id_resto == 0:
        #     self.resto_nb += 1
        #     id_resto = self.resto_nb

        urls_review = get_info.get_urls_reviews_in_restaurant_page(response)
        logger.warn('> Reviews detected : {}'.format(len(urls_review)))

        # For each review open the link and parse it into the parse_review method
        for url_review in urls_review:
             yield response.follow(url=url_review, callback=self.parse_review, cb_kwargs=dict(restaurant_id=id_resto))

        # Get next page information
        next_page, next_page_number = get_info.get_urls_next_list_of_reviews(response)
        logger.warn('> Next page of reviews : {}'.format(next_page))
        
        # Follow the page if we decide to
        if get_info.go_to_next_page(next_page, next_page_number, printing=False, max_page=50):
            logger.warn('> Yielding new page of reviews')
            yield response.follow(next_page, callback=self.parse_resto, cb_kwargs=dict(id_resto=id_resto))


    def parse_review(self, response, restaurant_id):
        """FINAL PARSING : Open a specific page with review and client opinion
            - Read these data and store them
            - Get all the data you can find and that you believe interesting
        """

        # Count the number of review scrapped
        self.review_nb += 1
        
        logger.warn('> Parsing review : {}'.format(self.review_nb))

        x_path_username = '//div[@class="username mo"]/span/text()'
        x_path_date_visit = '//div[@class="prw_rup prw_reviews_stay_date_hsx"]/text()'
        x_path_rating = '//div[@class="rating reviewItemInline"]/span[1]/@class'
        x_path_title = '//div[@class="quote"]/a/span/text()'
        x_path_comment = '//div[@class="entry"]/p/text()'
        x_path_restaurant_name = '//div[@class="rating"]/span/a/text()'
        
        # You can store the scrapped data into a dictionnary or create an Item in items.py (cf XActuItem and scrapy documentation)
        review_item = ReviewRestoItem()
        review_item['review_id'] = self.review_nb
        review_item['restaurant_id'] = restaurant_id
        review_item['restaurant_name'] = response.xpath(x_path_restaurant_name).get()
        review_item['rating'] = response.xpath(x_path_rating).get()[-2]
        review_item['user'] = response.xpath(x_path_username).get()
        review_item['date_of_visit'] = response.xpath(x_path_date_visit).get()
        review_item['title'] = response.xpath(x_path_title).get()
        review_item['comment'] = response.xpath(x_path_comment).get()
        logger.warn('> Create Review : {}'.format(review_item))

        yield review_item
