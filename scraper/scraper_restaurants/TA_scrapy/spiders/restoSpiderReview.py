
from logzero import logger
import logzero
import logging
import glob
import pandas as pd

# Scrapy packages
import scrapy
import requests
from scrapy.selector import Selector
from TA_scrapy.items import RestoItem
from TA_scrapy.spiders import get_info

# Chromedriver package and options
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

class RestaurantSpider(scrapy.Spider):
    name = "RestaurantSpider"

    def __init__(self, directory='./scraped_data/', 
                root_url='https://www.tripadvisor.co.uk/Restaurants-g191259-Greater_London_England.html', 
                debug=0, max_resto=100, scrap_website_menu=0, *args, **kwargs):
        
        super(RestaurantSpider, self).__init__(*args, **kwargs)

        # Set logging level
        logzero.loglevel(int(debug))
        if int(debug) == 0 :
            logging.disable(logging.DEBUG)

        # Setting the list of already scraped restaurants
        existing_jsons = glob.glob("./scraped_data/restaurants/*.json")
        logger.warn(f' > FINDING EXISTING JSONS {existing_jsons}')
        self.already_scraped_restaurants = []
        self.next_file_id = len(existing_jsons) + 1
        for json in existing_jsons:
            json_df = pd.read_json(json, lines=True)
            restaurants = json_df['resto_TA_url'].to_list()
            restaurants = [resto.split("https://www.tripadvisor.co.uk")[1] for resto in restaurants]
            self.already_scraped_restaurants += restaurants

        # User defined parameters
        self.directory = directory
        self.root_url = root_url
        self.scrap_website_menu = int(scrap_website_menu)
        self.max_resto = int(max_resto)
        self.currency = '£'

        # To track the evolution of scrapping
        self.resto_offset = len(self.already_scraped_restaurants)
        self.main_nb = 0
        self.resto_nb = 0
        self.restaurants_ids = []

        logger.warn(f"FINDING {self.resto_offset} EXISTING RESTAURANTS")        

    def start_requests(self):
        """ Give the urls to follow to scrapy
        - function automatically called when using "scrapy crawl my_spider"
        """

        # Basic restaurant page on TripAdvisor GreaterLondon
        yield scrapy.Request(url=self.root_url, callback=self.parse)
   
    def parse(self, response):
        """ MAIN PARSING : Start from a classical reastaurant page
            - Usually there are 30 restaurants per page
        """

        logger.info(' > PARSING NEW MAIN PAGE OF RESTO ({})'.format(self.main_nb))
        self.main_nb += 1

        # Get the list of the 35 restaurants of the page
        restaurant_urls = get_info.get_urls_resto_in_main_search_page(response)
        
        restaurant_new_urls = set(restaurant_urls) - set(self.already_scraped_restaurants)
        logger.warn(f'> FINDING : {len(restaurant_urls) - len(restaurant_new_urls)} RESTAURANTS ALREADY SCRAPED IN THIS PAGE')

        # For each url : follow restaurant url to get the reviews
        for restaurant_url in restaurant_new_urls:
            logger.info('> New restaurant detected : {}'.format(restaurant_url))
            self.resto_nb += 1
            if self.resto_nb > self.max_resto:
                return None
            yield response.follow(url=restaurant_url, callback=self.parse_resto, 
                                  cb_kwargs=dict(restaurant_id=self.resto_nb))

        # Get next page information
        next_page, next_page_number = get_info.get_urls_next_list_of_restos(response)
        
        # Follow the page if we decide to
        if get_info.go_to_next_page(next_page, next_page_number, max_page=None):
            yield response.follow(next_page, callback=self.parse)


    # def parse_review_page(self, response, restaurant_id):
    #     """ SECOND PARSING : Given a review page, gets each review url and get to parse it
    #         - Usually there are 10 reviews per page
    #     """

    #     # Parse the restaurant if it has not been parsed yet
    #     if restaurant_id not in self.restaurants_ids:
    #         yield self.parse_resto(response, restaurant_id)
    #         self.restaurants_ids.append(restaurant_id)

    def parse_resto(self, response, restaurant_id):
        """ Create Restaurant Item saved in specific JSON file """

        logger.info(' > START PARSING NEW RESTO ({})'.format(restaurant_id - 1))
        
        xpath_name = '//h1[@class="_3a1XQ88S"]/text()'
        xpath_nb_reviews = '//div[@class="_1ud-0ITN"]/span/a/span/text()'
        xpath_price_cuisine = '//span[@class="_13OzAOXO _34GKdBMV"]//a/text()'
        xpath_phone_number = '//div[@class="_1ud-0ITN"]/span/span/span/a/text()'
        xpath_website = '//a[@class="_2wKz--mA _15QfMZ2L"]/@data-encoded-url'
        xpath_ranking = '//*[@id="component_44"]/div/div[2]/span[2]/a/span/b/span/text()'
        xpath_ranking_out_of = '//span[@class="_13OzAOXO _2VxaSjVD"]/a/span/text()'
        xpath_rating = '//div[@class="_1ud-0ITN"]/span/a/svg/@title'
        xpath_address = '//span[@class="_13OzAOXO _2VxaSjVD"]/span[1]/a/text()'
        
        resto_item = RestoItem()
        resto_item['restaurant_id'] = restaurant_id + self.resto_offset
        resto_item['name'] = response.xpath(xpath_name).get()
        resto_item['resto_TA_url'] = response.url
        resto_item['nb_reviews'] = response.xpath(xpath_nb_reviews).get()
        price_cuisine = response.xpath(xpath_price_cuisine).getall()

        # Retrieve price in the right format
        raw_price = -1
        try:
            raw_price = price_cuisine[0]
        except:
            resto_item['min_price'] = None
            resto_item['max_price'] = None
            resto_item['cuisine'] = []
        else:
            try:
                min_price, max_price = raw_price.split(' - ')
                resto_item['min_price'] = len(min_price)
                resto_item['max_price'] = len(max_price)
                resto_item['cuisine'] = price_cuisine[1:]
            except ValueError:
                if raw_price == len(raw_price) * self.currency:
                    resto_item['min_price'] = len(raw_price)
                    resto_item['max_price'] = len(raw_price)
                    resto_item['cuisine'] = price_cuisine[1:]
                else:
                    resto_item['min_price'] = None
                    resto_item['max_price'] = None
                    resto_item['cuisine'] = price_cuisine

        resto_item['address'] = response.xpath(xpath_address).get()
        resto_item['phone_number'] = response.xpath(xpath_phone_number).get()

        # Scrap websites and menus depending on user input
        if self.scrap_website_menu:
            driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
            driver.get(response.url)

            # Catch website (use Selenium as URL is generated by JS)
            website = driver.find_element_by_class_name('_2wKz--mA')
            website_url = website.get_attribute('href') 
            if website_url is None:
                resto_item['website'] = 'Website not found'
            else:
                resto_item['website'] = website_url

            # Catch menu
            menu = driver.find_elements_by_xpath('//span[@class="_13OzAOXO _2VxaSjVD ly1Ix1xT"]/a')
            try:
                resto_item['menu'] = menu[0].get_attribute('href')
            except IndexError:
                resto_item['menu'] = 'Menu not found'
        else:
            resto_item['website'] = 'Website not scraped'
            resto_item['menu'] = 'Menu not scraped'
            
        if response.xpath(xpath_ranking).get() is not None and response.xpath(xpath_ranking_out_of).get() is not None:
            resto_item['ranking'] = response.xpath(xpath_ranking).get() + response.xpath(xpath_ranking_out_of).get()
        else:
            resto_item['ranking'] = 'Ranking not found'

        resto_item['rating'] = response.xpath(xpath_rating).get().split()[0]

        return resto_item