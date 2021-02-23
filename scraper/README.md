# TripAdvisor scraper

The first step of the project is to scrap TripAdvisor to gather data regarding Londonese restaurants.

## Run from Command Line

```
scrapy crawl RestoReviewSpider -a directory='./scraped_data/' -a root_url='user_chosen_url' -a debug=0 -a maxpage_resto=2 -a maxpage_reviews=50 -a scrap_user=1 -a scrap_website_menu=0
```

Running the above command will overwrite the files ``` scraped_data/reviews.json ```, ``` scraped_data/restaurant.json ``` and ``` scraped_data/users.json ``` that are already provided.

-a option allows for command line input arguments with scrapy command
* directory (string, default='./scraped_data/'):
  user-designated folder where scraped reviews, restaurants and users information will be stored
* root_url (string, default='https://www.tripadvisor.co.uk/Restaurants-g191259-Greater_London_England.html'):
  root URL for list of restaurants (URL of the city chosen by user)
* debug (int, default=0):
  0 or 1 – for no debug information or with debug information respectively
* maxpage_resto (int, default=2):
  number of pages of restaurants to parse from base URL (1 page ≈ 35 restaurants)
* maxpage_reviews (int, default=50):
  number of pages of reviews to parse for given restaurant (1 page ≈ 10 reviews)
* scrap_user (int, default=1):
  0 or 1 – for not scraping user information (faster) or scraping them respectively
* scrap_website_menu (int, default=0):
  0 or 1 – for not scraping restaurants' website and menu or scraping them respectively

## Data Collected (JSON format)

* Review Information: ID (unique), restaurant ID, username, date of visit, rating, title, comment
* Restaurant Information: ID (unique), name, number of reviews, price, cuisine type, address, phone number, website, menu, ranking, rating
* User Information: username (unique), fullname, date joined, number of contributions, number of followers, number of followings

## Data Available on Git

Data scraped on 29/01/2021 can be found in the folder ``` scraped_data/* ``` and in the ```scraped_data.zip``` file.
It contains the data for the first 124 restaurants (maxpage_resto=2), their associated reviews capped at 500 per restaurant (maxpage_reviews=50) and data regarding the authors of these reviews.

## Other scrapers

We developed 2 other scapers available in the folders ``` scraper_restaurants ``` and ``` scraper_roftops ```.

* The ``` scraper_restaurants ``` enables to scrap restaurant information without collecting their associated reviews. We used it to create a quasi-exhaustive database of Londonese restaurants in a reasonable amount of time (more than 13,000 restaurants). This database is available in the file ``` scraper_restaurants/scraped_data/restaurants/restaurants_run_1.json ```.

Usage: inside the ``` scraper_restaurants ``` folder: ```scrapy crawl RestaurantSpider -a debug=1 -a max_resto=15000```

* The ``` scraper_rooftops ``` enables to scrap individual restaurants and their reviews from their TripAdvisor URL. We used it to add a dozen of Londonese rooftops that were missing in our restaurants database.

Usage: inside the ``` scraper_restaurants ``` folder, fill the ```rooftops.txt``` file with the URL to scrap and execute
```sh scrap_rooftops.sh```. Lines starting with ```#``` are ignored.