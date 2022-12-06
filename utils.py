import pandas as pd
def bm25okapi_search(tokenized_query, bm25, corpus, n_results = 1):
    """
    Function that takes a tokenized query and prints the first 100 words of the 
    n_results most relevant results found in the corpus, based on the BM25
    method.
    
    Parameters
    ----------
    @param tokenized_query: list, array-like
        A valid list containing the tokenized query.
    @param bm25: BM25 object,
        A valid object of type BM25 (BM25Okapi or BM25Plus) from the library
        `rank-bm25`, initialized with a valid corpus.
    @param corpus: list, array-like
        A valid list containing the corpus from which the BM25 object has been 
        initialized. As returned from function read_corpus().
    @param n_results: int, default = 1
        The number of top results to print.
    """
    # Get top results for the query
    top_results = bm25.get_top_n(tokenized_query, corpus, n = n_results)
    # Take words from each result
    top_results_id = [int(' '.join(top_result).split("-->")[-1]) 
                             for top_result in top_results]
    
    # Return results
    return top_results_id


def search(query_df, ir_model, tokenize_lyric, n_results= 5):
    #return result dataframe
    search_results= pd.DataFrame()
    for i, query in query_df.iterrows():
        #return relevance lyric_id rank array
        lyrics_id=bm25okapi_search(query['query_token'], ir_model, tokenize_lyric, n_results)
        for rank,lyric_id in enumerate(lyrics_id):
            search_results= pd.concat([search_results,
                                    pd.DataFrame([{'rank': rank+1,
                                                    'id': lyric_id,
                                                    'qid': query['qid']}])
                                    ]).reset_index(drop=True)
    return search_results