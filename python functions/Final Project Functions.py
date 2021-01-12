import json
import os
import copy
import pandas as pd
import numpy as np
from numpy.linalg import norm

# set to false if want to see document level specific results
hide_detail = True


def loadFiles(test=False):
    # load files depending on using test dataset or own dataset
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    if test:
        return ['cranfield_document_tf_idf.json', 'cranfield_query_terms.json',
                'cranfield_query_rel.json']
    else:
        return ['article_tf_idf.json', 'query_terms.json']


def getTermDocDF(document_tf_idf_dict, indices=None):
    # create a pandas dataframe of a document by term table
    if indices is None:
        indices = []
    rows_list = []

    for url, terms in document_tf_idf_dict.items():
        df_row = {'Url': url}
        for term, val in terms.items():
            df_row.update({term: val})
        rows_list.append(df_row)

    term_doc_df = pd.DataFrame(rows_list)
    term_doc_df.fillna(0, inplace=True)
    term_doc_df.set_index('Url', inplace=True)
    if indices:
        term_doc_df = term_doc_df.iloc[indices, :]
    return term_doc_df


def getTermCoOccurMatrix(relevant_doc_df):
    # build a term co-occurrence matrix to help with query expansion
    term_array = relevant_doc_df.to_numpy()
    term_array_trans = term_array.transpose()
    term_cooccur = np.dot(term_array_trans, term_array) / (norm(term_array) * norm(term_array_trans))
    return term_cooccur


def expandQuery(term_cooccur, relevant_doc_df, query_terms, n=4):
    # get expanded queries for all query terms, use n = 3 to start
    # set n = n + 1 to account for own term matching
    expanded_query_term_dict = copy.deepcopy(query_terms)
    for name, term_list in query_terms.items():
        for term in term_list:
            if term in relevant_doc_df.columns:
                term_loc = relevant_doc_df.columns.get_loc(term)
                cooccur_vals = term_cooccur[term_loc]
                indices = np.argpartition(cooccur_vals, -n)[-n:]
                extra_terms = list(relevant_doc_df.columns[indices])
                expanded_query_term_dict[name].extend(extra_terms)

    # get rid of duplicate terms
    for name, term_list in expanded_query_term_dict.items():
        expanded_query_term_dict[name] = list(set(term_list))
    return expanded_query_term_dict


def queryDocuments(query_term_dict, relevant_doc_df):
    # querying documents
    rows_list = []

    for name, query_terms in query_term_dict.items():
        df_row = {'Name': name}
        for term in query_terms:
            if term in relevant_doc_df.columns:
                df_row.update({term: 1})
        rows_list.append(df_row)

    query_df = pd.DataFrame(rows_list)
    query_df.fillna(0, inplace=True)
    query_df.set_index('Name', inplace=True)

    term_doc_queries_df = relevant_doc_df.append(query_df)
    term_doc_queries_df.fillna(0, inplace=True)
    term_doc_queries_array = term_doc_queries_df.to_numpy()
    return term_doc_queries_array


def queryResults(query_term_dict, term_doc_queries_array):
    # make a dictionary of the results
    query_results = {}

    # get cosine similarity between query and documents
    term_doc_queries_array = np.nan_to_num(term_doc_queries_array)
    for i, name in enumerate(query_term_dict):
        query_results[name] = []
        query_loc = len(term_doc_queries_array) - len(query_term_dict) + i
        for row in term_doc_queries_array[:-len(query_term_dict), :]:
            query_results[name].append(
                np.dot(term_doc_queries_array[query_loc], row) / (norm(term_doc_queries_array[query_loc]) * norm(row)))
    return query_results


def expandToFullArray(local_query_results, term_doc_df, indices):
    # use to expand the local query results to the entire term array
    full_array_query_results = []
    c = 0
    for i in range(len(term_doc_df)):
        if i in indices:
            full_array_query_results.append(local_query_results[c])
            c += 1
        else:
            full_array_query_results.append(0)
    return full_array_query_results


def getRelevantDocuments(local_query_results):
    # get indices of the relevant documents to our query
    query_relevant_docs = {}
    for name, values in local_query_results.items():
        non_zero_indices = [i for i in range(len(values)) if values[i] > 0]
        query_relevant_docs[name] = non_zero_indices
    return query_relevant_docs


def outputQueryResultsForTest(expanded_query_results, relevant_doc_df, threshold=.2):
    # give results as all documents that exceed a certain cosine similarity
    query_results = {}
    print('Cosine Similarity Threshold = {}'.format(threshold))

    for name, result in expanded_query_results.items():
        arr = np.array(result)
        arr = np.nan_to_num(arr)
        indices = np.where(arr > threshold)
        new_arr = arr[indices]
        rel_docs = relevant_doc_df.iloc[indices]
        if not hide_detail:
            print('\nDocument {} matching documents are: '.format(name))
        for i, num in enumerate(rel_docs.index.values):
            if not hide_detail:
                print('\t{}: cosine sim = {}'.format(num, new_arr[i]))
        query_results[name] = rel_docs.index.values.tolist()
    return query_results


def outputPrecisionResultsForTest(query_top_docs, query_relevance, global_or_local):
    # get precision results for test using query relevance file
    sum_tot_tp = 0
    sum_tot_fn = 0
    sum_tot_fp = 0
    for query_num in query_top_docs:
        sum_query_tp = 0
        sum_query_fn = 0
        sum_query_fp = 0
        for doc_num in query_top_docs[query_num]:
            # Cranfield docs are not 0 based, add 1 to doc_nums
            if int(doc_num) + 1 in query_relevance[query_num]:
                sum_query_tp += 1
            else:
                sum_query_fp += 1
        sum_query_fn += (len(query_relevance[query_num]) - sum_query_tp)
        query_precision = 0
        query_recall = 0
        if sum_query_tp + sum_query_fp:
            query_precision = sum_query_tp / (sum_query_tp + sum_query_fp)
        if sum_query_tp + sum_query_fn:
            query_recall = sum_query_tp / (sum_query_tp + sum_query_fn)
        if not hide_detail:
            print(
                '\nPrecision score for query number {} is {} in {}'.format(query_num, query_precision, global_or_local))
            print('\nRecall score for query number {} is {} in {}'.format(query_num, query_recall, global_or_local))
        sum_tot_tp += sum_query_tp
        sum_tot_fp += sum_query_fp
        sum_tot_fn += sum_query_fn

    overall_precision = 0
    overall_recall = 0
    if sum_tot_tp + sum_tot_fp:
        overall_precision = sum_tot_tp / (sum_tot_tp + sum_tot_fp)
    if sum_tot_tp + sum_tot_fn:
        overall_recall = sum_tot_tp / (sum_tot_tp + sum_tot_fn)
    print('\nPrecision score overall for {} is {}'.format(global_or_local, overall_precision))
    print('\nRecall score overall for {} is {}'.format(global_or_local, overall_recall))


def outputQueryResults(expanded_query_results, relevant_doc_df):
    # give results as top n articles for each company
    # start with n = 3
    n = 3
    expanded_query_top_n = {}

    for name, result in expanded_query_results.items():
        arr = np.array(result)
        if np.isnan(arr).all():
            expanded_query_top_n[name] = 'No Results'
            continue
        indices = np.argpartition(arr, -n)[-n:]
        top_articles = relevant_doc_df.iloc[indices]
        expanded_query_top_n[name] = top_articles.index.values.tolist()

    for name in expanded_query_top_n:
        print('\nCompany {} top {} articles are: '.format(name, n))
        if expanded_query_top_n[name] == 'No Results':
            print('\tNo articles relating to {}'.format(name))
            continue
        for i, article in enumerate(expanded_query_top_n[name]):
            print('\t{}: {}'.format(i + 1, article))
    return expanded_query_top_n


def outputPrecisionResults(expanded_query_top_n, global_or_local):
    # output precision scores for comparison between global / local methods
    relevance_results = {}
    # give relevance scores for each url returned:
    for name, articles in expanded_query_top_n.items():
        if expanded_query_top_n[name] == 'No Results':
            continue
        relevance_results[name] = {}
        for article in articles:
            relevance = input('Is {} relevant to company {}? 1 for yes 0 for no'.format(article, name))
            relevance_results[name][article] = relevance

    # get precision results
    sum_tot = 0
    for name in relevance_results:
        sum_comp = 0
        for result in relevance_results[name].values():
            sum_comp += int(result)
            sum_tot += int(result)
        company_score = sum_comp / len(relevance_results[name])
        if not hide_detail:
            print('\nPrecision score for company {} is {} in {}'.format(name, company_score, global_or_local))

    overall_score = sum_tot / sum(len(v) for v in relevance_results.values())
    print('\nPrecision score overall for {} is {}'.format(global_or_local, overall_score))
