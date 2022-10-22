import os
import nltk
import json
import pickle
import re
import string
import pandas as pd
import numpy as np

class Preprocessing:
	def __init__(self):
		self.CONTRACTION_MAP= self._load('dictionary')
		self.stop_words= set(nltk.corpus.stopwords.words('english'))

	def _load(self, path):
		#load dictionary file from 'dictionary' folder
		if 'contraction_word_dictionary.txt' not in os.listdir(path):
			print("Not find 'contraction_word_dictionary.txt' file.")
			return {}
		else:
			with open(os.path.join(path,'contraction_word_dictionary.txt')) as f:
				return json.loads(f.read())

	def text_lowercase(self, text):
		return text.lower()

	def delete_tag(self, text):
		return re.sub('\[(.*?)\]','', text)

	def remove_whitespace(self, text):
		return  " ".join(text.split())

	def remove_stopwords(self, text):
		word_tokens = nltk.tokenize.word_tokenize(text)
		filtered_text = [word for word in word_tokens if word not in self.stop_words]
		return filtered_text

	def remove_punctuation(self, text):
		translator = str.maketrans('', '', string.punctuation)
		return text.translate(translator)

	def replace_cw(self, text):
		"""
		return string-type
		"""
		return ' '.join([self.CONTRACTION_MAP.get(item, item) for item in text.split()])


	def Preprocess(self, str):
		str = self.text_lowercase(str)
		str = delete_tag(str)
		str = str.replace('\n\n', '')
		str = str.replace('\n', ' ')
		str = self.replace_cw(str)
		str= self.remove_stopwords(str)
		str= self.remove_punctuation(" ".join(str))
		str= self.remove_whitespace(str)
		return str