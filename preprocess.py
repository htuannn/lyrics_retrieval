import os
import nltk
import json
import pickle
import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import wordnet

from lemma import LemmatizationWithPOSTagger
class Preprocessing:
	def __init__(self, Pipeline=['']):
		self.Pipeline=Pipeline

		if "stemming" in self.Pipeline:
			self.stemming = PorterStemmer()
		if "nltk_lemmaziter" in self.Pipeline:
			self.lemmatizer=LemmatizationWithPOSTagger(use_nltk_lemma=True)
		if "lemmatizer" in self.Pipeline:
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
		return text.lower()

	def convert_unicode(self,text):
		return text.encode('ascii', 'ignore').decode()

	def delete_tag(self, text):
		return re.sub('\[(.*?)\]','', text)

	def remove_whitespace(self, text):
		return  " ".join(text.split())

	def remove_stopwords(self, text):
		word_tokens = nltk.tokenize.word_tokenize(text)
		filtered_text = [word for word in word_tokens if word not in self.stop_words]
		return " ".join(filtered_text)

	def remove_punctuation(self, text):
		translator = str.maketrans('', '', string.punctuation)
		return text.translate(translator)

	def replace_cw(self, text):
		"""
		return string-type
		"""
		return ' '.join([self.CONTRACTION_MAP.get(item, item) for item in text.split()])
	

	def lemmatize(self, text):
		return ". ".join(self.lemmatizer.lemmatize(text.split('. ')))

	def stem(self, text):
		#return " ".join([self.stemming.stem(word) for word in text.split('. ')])
		pass

	def handle_negation(self, text):
		match = re.search(r'\b(?:not)\b (\S+)', text)
		if match:
			text= text.replace(match.group(0), 'NEG_' + match.group(1))
		return text

	def Preprocess(self, str):
		str = self.text_lowercase(str)
		str= self.convert_unicode(str)
		str = self.delete_tag(str)
		str = str.replace('\n\n', '\n')
		str = str.replace('\n', '. ')
		str = self.replace_cw(str)
		if any(opt in self.Pipeline for opt in ['nltk_lemmaziter', 'lemmatizer']):
			str= self.lemmatize(str)

		if 'stemming' in self.Pipeline:
			str= self.stem(str)

		str= self.remove_punctuation(str)
		str= self.remove_whitespace(str)

		if 'handle_negation' in self.Pipeline:
			str = self.handle_negation(str)

		str= self.remove_stopwords(str)
		return str