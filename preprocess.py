import os
import nltk
import json
import pickle
import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from lemma import LemmatizationWithPOSTagger
class Preprocessing:
	def __init__(self, Pipeline=['']):
		self.Pipeline=Pipeline

		self.tokenizer='No_tokenize'
		if "nltk_word_tokenizer" in self.Pipeline:
			self.tokenizer='nltk'
		elif "word_space_tokenize" in self.Pipeline:
			self.tokenizer='split'

		if "PorterStemmer" in self.Pipeline:
			self.stemmer = PorterStemmer()
		if "WordNetLemmatizer" in self.Pipeline:
			self.lemmatizer=LemmatizationWithPOSTagger(use_nltk_lemma=True)
		if "DIY_lemmatizer" in self.Pipeline:
			self.lemmatizer=LemmatizationWithPOSTagger()

		self.CONTRACTION_MAP= self._load('dictionary')
		self.stop_words= set(nltk.corpus.stopwords.words('english'))
		self.stop_words.remove("not")
	
	def _load(self, path):
		#load dictionary file from 'dictionary' folder
		if 'contraction_word_dictionary.txt' not in os.listdir(path):
			print("Not find 'contraction_word_dictionary.txt' file.")
			return {}
		else:
			with open(os.path.join(path,'contraction_word_dictionary.txt')) as f:
				return json.loads(f.read())

	def text_lowercase(self, text):
		#convert to lowercase
		return text.lower()

	def convert_unicode(self,text):
		return text.encode('ascii', 'ignore').decode()

	def delete_tag(self, text):
		#remove tag ['[Verse 1]','[Chorus]','[Intro]',...] in lyric
		return re.sub('\[(.*?)\]','', text)

	def remove_whitespace(self, text):
		#remove extra space
		return  " ".join(text.split())

	def remove_stopwords(self, text):
		#remove word exist in stopword dictionary
		word_tokens = nltk.tokenize.word_tokenize(text)
		filtered_text = [word for word in word_tokens if word not in self.stop_words]
		return " ".join(filtered_text)

	def remove_punctuation(self, text):
		# remove punctuation ex:,.?!;'"+
		translator = str.maketrans('', '', string.punctuation)
		return text.translate(translator)

	def expand_contraction(self, text):
		"""
		return string-type
		"""
		return ' '.join([self.CONTRACTION_MAP.get(item, item) for item in text.split()])
	

	def lemmatize(self, text):
		return ". ".join([str(sent) for sent in self.lemmatizer.lemmatize(text.split('. '), self.tokenizer)])

	def stem(self, text):
		return ". ".join([" ".join([self.stemmer.stem(word) for word in sent.split(' ')]) for sent in text.split('. ')])

	"""
	def handle_negation(self, text):
		match = re.search(r'\b(?:not)\b (\S+)', text)
		if match:
			text= text.replace(match.group(0), 'NEG_' + match.group(1))
		return text
	"""

	def Preprocess(self, str):
		str = self.text_lowercase(str)
		str= self.convert_unicode(str)
		str = self.delete_tag(str)
		str = str.replace('\n\n', '')
		str = str.replace('\n', '. ')
		str= self.remove_whitespace(str)
		str = self.expand_contraction(str)

		if any(opt in self.Pipeline for opt in ['WordNetLemmatizer', 'DIY_lemmatizer']):
			str= self.lemmatize(str)

		if 'PorterStemmer' in self.Pipeline:
			str= self.stem(str)

		str= self.remove_punctuation(str)
		
		"""
		if 'handle_negation' in self.Pipeline:
			str = self.handle_negation(str)
		"""

		if 'remove_stopword' in self.Pipeline:
			str= self.remove_stopwords(str)
		return str