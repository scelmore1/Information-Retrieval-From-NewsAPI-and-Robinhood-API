from newsapi import NewsApiClient
import datetime
import json
import newspaper
from newspaper import Article

# use this api to request online articles relating to certain topics
# check https://newsapi.org/docs for more information about this API

# get an api key
news_api_key = '11c62d49756247ccadc727086529f7b1'

# initialize a news api client
news_api = NewsApiClient(api_key=news_api_key)

# get a list of sources and sort by category
sources = news_api.get_sources()['sources']
categories = set([source['category'] for source in sources])

# create a dictionary based off category
category_based_dict = {}
for cat in categories:
    category_based_dict.update({cat: []})
    for source in sources:
        if source['category'] == cat:
            category_based_dict[cat].append(source['id'])

# only get articles related to business and technology
business_tech = category_based_dict['business'] + category_based_dict['technology']

# get articles from past 3 weeks
past_month_dates = []
for x in range(21):
    past_month_dates.append(datetime.date.today() + datetime.timedelta(days=-x))

business_tech_articles = []
for date in past_month_dates:
    daily_request = news_api.get_everything(
        sources=','.join([str(x) for x in business_tech]), language='en',
        from_param=str(date), to=str(date), page_size=100, page=1)
    business_tech_articles.append(daily_request['articles'])

business_tech_articles = [item for sublist in business_tech_articles for item in sublist]

# use list of urls as document set
business_tech_articles_urls = list(set(article['url'] for article in business_tech_articles))

# article dictionary will have url as key, text of article as value
all_articles_dict = {}

# create json file from dictionary, append values if running from a new date
with open('newsapi_articles.json', 'r') as json_file:
    dict_data = json.load(json_file)
    all_articles_dict.update(dict_data)

# library 'newspaper' does web scrape of urls by calling .download() and .parse()
for article_url in business_tech_articles_urls:
    if article_url not in all_articles_dict:
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except newspaper.article.ArticleException:
            continue
        all_articles_dict[article_url] = article.title + ' ' + article.text

with open('newsapi_articles.json', 'w') as json_file:
    json.dump(all_articles_dict, json_file)

