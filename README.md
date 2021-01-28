# Trip Advisor Scrapping Tool

Session 1: Scraping (Capgemini Data Camp)

Tool to scrap Trip Advisor UK website (https://www.tripadvisor.co.uk/) for restaurants and their associated reviews made by different users.

# Setup

```
pip install -r requirements.txt
```

# Run from Command Line

```
scrapy crawl RestoReviewSpider -a directory='./scrapped_data/' -a debug=0 -a maxpage_resto=3 -a maxpage_reviews=50
```

-a option allows for command line input arguments with scrapy command
* directory: directory of user defined data folder for scrapped reviews, restaurants and users information
* debug: 0 or 1, for no debug information or with debug information respectively
// url: root url for list of restaurants (URL of city)
* maxpage_resto: number of pages of restaurants to parse from base URL (1 page = ~35 restaurants)
* maxpage_reviews: number of pages of reviews to parse for given restaurant (1 page = 10 reviews)

# Data Collected (JSON format)

* Restaurant Information: Id (unique) name, number of reviews, price, cuisine type, address, phone number, website, ranking, rating
* Review Information: Id (unique), restaurant Id, Username, date of visit, rating, title, comment
* User Information: Username (unique), date joined, number of contributions, number of followers, number of following
