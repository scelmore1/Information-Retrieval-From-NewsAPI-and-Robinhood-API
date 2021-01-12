from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import os
import math

# use nltk library to get stopwords and import a Porter Stemmer
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

cranfield_docs = {}
doc_num = -1

# build an inverted index w/ the document terms from the cranfield file
with open('cran.all.1400', 'r') as txt_file:
    for line in txt_file:
        if line[:2] == '.I':
            if doc_num != -1:
                cranfield_docs[doc_num] = text
            doc_num += 1
            read_text = False
            text = ''
        elif line[:2] == '.W':
            read_text = True
        elif read_text:
            text += line
    cranfield_docs[doc_num] = text

# create a raw inverted index
inverted_index_raw = {}

for num, text in cranfield_docs.items():
    for word in word_tokenize(text):
        if word.lower() not in stop_words and word.isalpha():
            stemmed_word = ps.stem(word.lower())
            if stemmed_word in inverted_index_raw:
                inverted_index_raw[stemmed_word].append(num)
            else:
                inverted_index_raw[stemmed_word] = [num]

# create a query dictionary from cranfield query file
query_dict = {}
query_num = 0

with open('cran.qry', 'r') as txt_file:
    for line in txt_file:
        if line[:2] == '.I':
            if query_num != 0:
                query_dict[query_num] = text
            query_num += 1
            text = ''
        elif line[:2] == '.W':
            pass
        else:
            text += line
    query_dict[query_num] = text

# create new dictionary for holding the stemmed query terms
stemmed_query_terms = {}

# query terms must be tokenized and stemmed to match terms in inverted index
for num, text in query_dict.items():
    stemmed_term = []
    for word in word_tokenize(text):
        if word not in stop_words and word.isalpha():
            stemmed_word = ps.stem(word)
            stemmed_term.append(stemmed_word)
    stemmed_query_terms[num] = stemmed_term

# create inverted index and pointer index to document id frequency
inverted_index = {}
pointer_index = {}

for term in inverted_index_raw:
    inverted_index[term] = {'freq': len(inverted_index_raw[term]), 'id': id(term)}
    pointer_index[id(term)] = {}
    for num in inverted_index_raw[term]:
        if num not in pointer_index[id(term)]:
            pointer_index[id(term)][num] = inverted_index_raw[term].count(num)

# n = number of documents
n = len(cranfield_docs)

# create dictionary of term idf's
term_idf_dict = {}

for term in inverted_index.keys():
    term_idf_dict[term] = math.log2(n / len(pointer_index[inverted_index[term]['id']]))

# create a tf_idf dictionary keyed on the number of the document
document_tf_idf_dict = {}
for num in cranfield_docs:
    document_tf_idf_dict[num] = {}

for term in inverted_index.keys():
    idf = term_idf_dict[term]
    for included_url in pointer_index[inverted_index[term]['id']].keys():
        count = pointer_index[inverted_index[term]['id']][included_url]
        document_tf_idf_dict[included_url][term] = (idf * count)

# create a relevance dictionary from cranfield rel file
rel_dict = {k: [] for k in range(1, len(query_dict)+1)}

with open('cranqrel', 'r') as txt_file:
    for line in txt_file:
        tokenized = line.split()
        rel_dict[int(tokenized[0])].append(int(tokenized[1]))

# write dictionaries to json files for easy reloading for future sessions
with open('cranfield_inverted_index.json', 'w') as json_file:
    json.dump(inverted_index, json_file)

with open('cranfield_pointer_index.json', 'w') as json_file:
    json.dump(pointer_index, json_file)

with open('cranfield_document_tf_idf.json', 'w') as json_file:
    json.dump(document_tf_idf_dict, json_file)

with open('cranfield_query_terms.json', 'w') as json_file:
    json.dump(stemmed_query_terms, json_file)

with open('cranfield_query_rel.json', 'w') as json_file:
    json.dump(rel_dict, json_file)

