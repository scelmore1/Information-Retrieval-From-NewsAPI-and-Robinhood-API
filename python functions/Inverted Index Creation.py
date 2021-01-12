from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import math

# use nltk library to get stopwords and import a Porter Stemmer
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# build an inverted index w/ the document terms contained in the following .json
articles_dict = {}
with open('newsapi_articles.json', 'r') as json_file:
    articles_dict = json.load(json_file)

# create a raw inverted index
inverted_index_raw = {}

# import terms into 'raw' inverted index
for url, text in articles_dict.items():
    for word in word_tokenize(text):
        if word.lower() not in stop_words and word.isalpha():
            stemmed_word = ps.stem(word.lower())
            if stemmed_word in inverted_index_raw:
                inverted_index_raw[stemmed_word].append(url)
            else:
                inverted_index_raw[stemmed_word] = [url]

# create a portfolio dictionary with company names and their stock tickers
portfolio_dict = {}

with open('stock_portfolio.json', 'r') as json_file:
    portfolio_dict = json.load(json_file)

all_stocks = list(x for v in portfolio_dict.values() for x in v)
query_terms = [(stock['Name'].lower(), stock['Symbol'].lower()) for stock in all_stocks]

# create new dictionary for holding the stemmed query terms
stemmed_query_terms = {}

# query terms must be tokenized and stemmed to match terms in inverted index
for term in query_terms:
    stemmed_term = []
    for word in word_tokenize(term[0]):
        if word not in stop_words and word.isalpha():
            stemmed_word = ps.stem(word)
            stemmed_term.append(stemmed_word)
    stemmed_term.append(term[1])
    stemmed_query_terms[term[0]] = stemmed_term

# create inverted index and pointer index to document id frequency
inverted_index = {}
pointer_index = {}

for term in inverted_index_raw:
    inverted_index[term] = {'freq': len(inverted_index_raw[term]), 'id': id(term)}
    pointer_index[id(term)] = {}
    for url in inverted_index_raw[term]:
        if url not in pointer_index[id(term)]:
            pointer_index[id(term)][url] = inverted_index_raw[term].count(url)

# n = number of documents
n = len(articles_dict)

# create dictionary of term idf's
term_idf_dict = {}

for term in inverted_index.keys():
    term_idf_dict[term] = math.log2(n / len(pointer_index[inverted_index[term]['id']]))

# create a tf_idf dictionary keyed on the url of the article
article_tf_idf_dict = {}
for url in articles_dict:
    article_tf_idf_dict[url] = {}

for term in inverted_index.keys():
    idf = term_idf_dict[term]
    for included_url in pointer_index[inverted_index[term]['id']].keys():
        count = pointer_index[inverted_index[term]['id']][included_url]
        article_tf_idf_dict[included_url][term] = (idf * count)

# write dictionaries to json files for easy reloading for future sessions
with open('inverted_index.json', 'w') as json_file:
    json.dump(inverted_index, json_file)

with open('pointer_index.json', 'w') as json_file:
    json.dump(pointer_index, json_file)

with open('article_tf_idf.json', 'w') as json_file:
    json.dump(article_tf_idf_dict, json_file)

with open('query_terms.json', 'w') as json_file:
    json.dump(stemmed_query_terms, json_file)