import os
import nltk
import json
import pickle
import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import wordnet

class Preprocessing:
	def __init__(self):
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

	def negation_handler(self, sentence):
		temp = int(0)
		sentence=nltk.word_tokenize(sentence)
		for i in range(len(sentence)):
			if sentence[i-1] in ['not',"n't"]:
				antonyms = []
				for syn in wordnet.synsets(sentence[i]):
					syns = wordnet.synsets(sentence[i])
					w1 = syns[0].name()
					temp = 0
					for l in syn.lemmas():
						if l.antonyms():
							antonyms.append(l.antonyms()[0].name())
					max_dissimilarity = 0
					for ant in antonyms:
						syns = wordnet.synsets(ant)
						w2 = syns[0].name()
						syns = wordnet.synsets(sentence[i])
						w1 = syns[0].name()
						word1 = wordnet.synset(w1)
						word2 = wordnet.synset(w2)
						if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
							temp = 1 - word1.wup_similarity(word2)
						if temp>max_dissimilarity:
							max_dissimilarity = temp
							antonym_max = ant
							sentence[i] = antonym_max
							sentence[i-1] = ''
		while '' in sentence:
			sentence.remove('')
		return sentence

	def handle_negation(self, text):
		match = re.search(r'\b(?:not)\b (\S+)', text)
		if match:
			text= text.replace(match.group(0), 'NEG_' + match.group(1))
		return text

	def Preprocess(self, str, handle_negation= False):
		str = self.text_lowercase(str)
		str= self.convert_unicode(str)
		str = self.delete_tag(str)
		str = str.replace('\n\n', '')
		str = str.replace('\n', ' ')
		str = self.replace_cw(str)
		str= self.remove_punctuation(str)
		str= self.remove_whitespace(str)
		str= self.remove_stopwords(str)
		if handle_negation:
			str = self.negation_handler(str)
		

		return str