# SONG'S INFORMATION RETRIEVAL BASED ON LYRICS

![alt text](https://github.com/htuannn/Information-Retrieval-Music-Searching-Machine/blob/09f46816b8a27aeec9b7db766272c5830af30141/sample.jpeg "Samples")

## Introduction
>Information Retrieval is the process through which a computer system can respond to a user's query for text-based information on a specific topic. IR was one of the first and remains one of the most important problems in the domain of natural laguague processing (NLP) - [stanford cs276](https://web.stanford.edu/class/cs276/)

This Search Engine gives the result information about the song based on the relevance of the query about the lyrics provided by the user.
## Motivation
The system supports users to search for songs based on a query from the lyrics.

We build an appilcation with similar idea with Shazam, MusixMatch
## Data 
The database we use for this retrieval model is from [Song Lyrics Dataset](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset) on Kaggle.

This dataset contains lyric of songs by various artists. Thanks to The Author for creating this dataset, and for inspiring us to make this project. 

## Requirements
- numpy
- pandas
- re 
- pickle
- json
- nltk 
- rank_bm25

Install all packages with the line: ``` pip install -r requirements.txt ```

After installing the NLTK package, please do install NLTK Data for specific functions to work.
Following this command in your terminal: 
```
python
import nltk
nltk.download('popular')
```
## Usage
We deployed our application to Streamlit framework for demo purposes of our project. 

To run it, firstly, install the environment according to the requirements section above. 

Then, run with the line: ``` streamlit run music-retrieval.py ```

Or without using Streamlit framework, you can run with jupyter notebook file: 
``` jupyter notebook Information_Retrieval.ipynb ```

Remember to load the data file **music_data.csv** to be able to perform the next operations. 

You can use your own custom music database by creating a file with the same structure as our data file **music_data.csv**. 

