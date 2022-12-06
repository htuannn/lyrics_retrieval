import streamlit as st
import lemma
import preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus


st.title("SONG'S INFORMATION RETRIEVAL FROM LYRICS")
df = pd.read_csv('music_dataset.csv', encoding='utf-8')

pre = []
st.header("Choose data preprocessing methods:")
pre1 = st.selectbox("Choose restoring word root form methods:", ("stemming", "nltk_lemmatizer", "DIY_lemmatizer", "none"))
st.markdown("Select if you want to keep the negative meaning of sentence:")
pre2 = st.checkbox("handle_negation")
st.markdown("Select if you want to remove the stopwords:")
pre3 = st.checkbox("remove_stopwords")

pre.append(pre1)
if pre2:
    pre.append("handle_negation")
if pre3:
    pre.append("remove_stopwords")

st.write('You select:', pre)

run_pre = st.button('Start Preprocess')
if run_pre:
    clean = preprocess.Preprocessing(pre)
    lyrics = np.array([])
    for lyric in df.lyrics:
        lyrics=np.append(lyrics,clean.Preprocess(lyric))
    tokenize_lyric=[]
    for lyric in lyrics:
        tokenize_lyric.append(lyric.split())
    for i in range(len(df.id)):
        tokenize_lyric[i].append(f"-->{df.id[i]}")
    st.markdown(tokenize_lyric)

    tab1, tab2 = st.tabs(["Evaluation", "Search Engine"])

    with tab1:
        st.header('Model Evaluation')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            test_df = pd.read_csv(uploaded_file)
            st.write(test_df)
        
        bm25 = BM25Plus(tokenize_lyric)

        def bm25okapi_search(tokenized_query, bm25, corpus, n_results = 1):
            # Get top results for the query
            top_results = bm25.get_top_n(tokenized_query, corpus, n = n_results)
            # Take words from each result
            top_results_id = [int(' '.join(top_result).split("-->")[-1]) 
                                    for top_result in top_results]       
            # Return results
            return top_results_id

        test_query=pd.read_csv('test.csv')
        test_query['qid']=test_query.index +1
            

        query_token=[]
        for _, row in test_query.iterrows(): 
            query=row['corpus'].replace('...',' ')
            query=clean.Preprocess(query)
            query=query.split(" ")
            query_token.append(query)
            

        test_query['query_token']= query_token

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
                                            'qid': query['qid']}])],
                                            ).reset_index(drop=True)
            return search_results

        search_results=search(test_query, bm25, tokenize_lyric, 10)

        judgments= test_query[['qid','id']]
        judgments['relevancy grade']= 1
        labeled_search_results = search_results.merge(judgments, how='left', on=['qid', 'id']).fillna(0)
        relevances_rank = labeled_search_results.groupby(['qid', 'relevancy grade'])['rank'].min()
        #Calculate evaluate Score as Mean Reciprocal Rank (MRR) metric
        ranks = relevances_rank.loc[:, 1]
        reciprocal_ranks = 1 / (ranks)
        MRR = reciprocal_ranks.sum()/len(judgments)
        st.metric(label="Mean Reciprocal Rank", value=str(MRR))
        st.write(ranks)

    with tab2:
        pass
        