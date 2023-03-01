import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
import time

from utils import bm25okapi_search, search
import lemma
import preprocess


st.title("SONG'S INFORMATION RETRIEVAL FROM LYRICS")
uploaded_data_file = st.file_uploader("Choose a music data file")
if uploaded_data_file is not None :
    df = pd.read_csv(uploaded_data_file, encoding='utf-8')


pipeline = []
st.sidebar.header("Choose data preprocessing methods:")
normalise_text = st.sidebar.selectbox("Choose restoring word root form methods:", ("PorterStemmer", "WordNetLemmatizer", "DIY_lemmatizer", "none"))
#st.sidebar.markdown("Select if you want to keep the negative meaning of sentence:")
#handle_negation = st.sidebar.checkbox("handle_negation")
st.sidebar.markdown("Select if you want to remove the stopwords:")
remove_stopwords = st.sidebar.checkbox("remove_stopword")

pipeline.append(normalise_text)

#if handle_negation:
    #pipeline.append("handle_negation")
    
if remove_stopwords:
    pipeline.append("remove_stopword")

st.sidebar.write('Your selection:', pipeline)

pipeline.append("nltk_word_tokenizer")


if 'tokenize_lyric' not in st.session_state:
    st.session_state.tokenize_lyric = None
if 'preprocesser' not in st.session_state:
    st.session_state.preprocesser = None

if st.sidebar.button('Start preprocessing', disabled= (uploaded_data_file == None)):
    lyrics = np.array([])
    preprocesser = preprocess.Preprocessing(Pipeline=pipeline)
    process_bar = st.sidebar.progress(0)
    with st.spinner(text="Please wait..."):
        start=time.time()
        for i, lyric in enumerate(df.lyrics):
            lyrics=np.append(lyrics,preprocesser.Preprocess(lyric))
            process_bar.progress((i + 1)/len(df.lyrics))
        end=time.time()
        tokenize_lyric=[]
        for lyric in lyrics:
            tokenize_lyric.append(lyric.split())
        for i in range(len(df.id)):
            tokenize_lyric[i].append(f"-->{df.id[i]}")

    st.session_state.tokenize_lyric = tokenize_lyric
    st.session_state.preprocesser= preprocesser
    st.session_state.time=end-start
if st.session_state.tokenize_lyric is not None:
    st.sidebar.success('Done!', icon="✅")
    st.sidebar.write(f'Executing time: {st.session_state.time:.4f}s')

tab1, tab2 = st.tabs(["Evaluation", "Search Engine"])

with tab1:
    st.header('Model Evaluation')
    uploaded_test_file = st.file_uploader("Choose a test set file")
    if uploaded_test_file is not None:
        test_df = pd.read_csv(uploaded_test_file)
        st.write(test_df)
        top_n_rank= st.number_input('Input number of top rank to get', min_value=1, max_value=100)
        if st.button('Evaluate'):
            if st.session_state.tokenize_lyric is None:
                st.error("You haven't preprocessed the data yet!! Please run it first!!", icon="🚨")
            else:
                preprocesser=st.session_state.preprocesser
                with st.spinner(text="Searching..."):
                    tokenize_lyric = st.session_state.tokenize_lyric
                    test_df['qid'] = test_df.index +1
                    query_token=[]
                    for _, row in test_df.iterrows(): 
                        query=row['corpus'].replace('...',' ')
                        query=preprocesser.Preprocess(query)
                        query=query.split(" ")
                        query_token.append(query)
                    
                    test_df['query_token']= query_token

                    start=time.time()
                    bm25 = BM25Okapi(tokenize_lyric)
                    search_results=search(test_df, bm25, tokenize_lyric, top_n_rank)
                    end=time.time()

                    judgments= test_df[['qid','id']]
                    judgments['relevancy grade']= 1
                    labeled_search_results = search_results.merge(judgments, how='left', on=['qid', 'id']).fillna(0)
                    relevances_rank = labeled_search_results.groupby(['qid', 'relevancy grade'])['rank'].min()
                    #Calculate evaluate Score by Mean Reciprocal Rank (MRR) metric
                    #RR = 1/position of first relevant result
                    ranks = relevances_rank.loc[:, 1]
                    reciprocal_ranks = 1 / (ranks)
                    MRR = reciprocal_ranks.sum()/len(judgments)

                    rank, count= np.unique(ranks, return_counts= True)
                counts_rank= np.zeros(top_n_rank+1)
                for (i, num) in zip(rank, count):
                    counts_rank[i]= num

                counts_rank[0]= len(test_df)- count.sum()

                fig =plt.figure(figsize=(8,4))
                barWidth=0.25
                plt.title(f"Frequency of ratings with top_n_rank={top_n_rank}")
                plt.xlabel("Rank")

                plt.bar(np.arange(0,top_n_rank+1),counts_rank, width=barWidth)
                st.pyplot(fig)
                st.metric(label="Mean Reciprocal Rank", value=str(MRR))

                st.write(f"Search execution time: {(end-start):.4f}s")

with tab2:

    query=st.text_input('Input query:')
    top_relevant= st.number_input('Number of most relevant result', min_value=1, max_value=100)

    if st.button("Search"):
        if st.session_state.tokenize_lyric is None:
            st.error("You haven't preprocessed the data yet!! Please run it first!!", icon="🚨")
        else:
            preprocesser=st.session_state.preprocesser
            query=query.replace('...',' ')
            query=preprocesser.Preprocess(query)
            token=query.split(" ")
            with st.spinner(text="Searching..."):
                tokenize_lyric = st.session_state.tokenize_lyric
                bm25 = BM25Okapi(tokenize_lyric)
                rank=bm25okapi_search(token, bm25, tokenize_lyric, n_results= top_relevant)
            st.write(df[df.id.isin(rank)].reset_index(drop=True))
