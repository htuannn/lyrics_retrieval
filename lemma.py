import nltk
import os
import pickle
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd

class Lemmatization(object):
    def __init__(self, dict_path='dictionary'):
        self.dict_path = dict_path
        self.vowels = "aeiou"
        self.word_lemma_dict={}
        with open(os.path.join(self.dict_path, "BNClemma10_3_with_c5.txt"), 'r', encoding='utf-8') as file:
            lines=file.read()
        lines = lines.replace('\ufeff', '').split('\n')
        for line in lines:
            parts = line.split("->")
            if len(parts) <2:
                continue
            lemma= parts[0].strip().lower()
            forms= parts[1].split(",")

            for form in forms:
                data= form.split(">")
                tag= self._bnc_to_ud(data[0].replace('<','').strip())
                word = data[1].strip().lower()
                self._add_lemma_to_dict(word, tag, lemma)

        irregular_df = pd.read_csv(os.path.join(self.dict_path, 'noun_exceptions.csv'), index_col=0).squeeze("columns")
        irregular_dict= irregular_df.to_dict()
        self.irregular_dict = dict((v,k) for k,v in irregular_dict.items())

    def _bnc_to_ud(self, tag):
        #Convert tag
        if "AJ" in tag:
            return "ADJ"
        if tag == "AT0":
            return "DET"
        if "AV" in tag:
            return "ADV"
        if tag == "CJC":
            return "CCONJ"
        if tag in ["CJS", "CJT"]:
            return "SCONJ"
        if tag in ["CRD", "ORD"]:
            return "NUM"
        if tag == "DPS":
            return "PRON"
        if tag in ["DT0", "DTQ"]:
            return "DET"
        if tag == "EX0":
            return "PRON"
        if tag == "ITJ":
            return "INTJ"
        if tag in ["NN0","NN1","NN2"]:
            return "NOUN"
        if tag == "NPO":
            return "PROPN"
        if "PN" in tag:
            return "PRON"
        if tag in ["POS","TO0","XX0","ZZ0"]:
            return "PART"
        if "PR" in tag:
            return "ADP"
        if "PU" in tag:
            return "PUNCT"
        if tag == "UNC":
            return "NOUN"
        if tag.startswith("V"):
                return "VERB"

    def _convert_pos(self,treebank_tag):
        """
        return POS compliance to lemmatization dictionary POS 
        """
        #JJ, JJR, JJS -> ADJ
        if treebank_tag.startswith('J'):
            return "ADJ"
        #VB, VBD, VBG, VBN, VBP, VBZ -> VERB
        if treebank_tag.startswith('V') :
            return "VERB"
        #NN, NNS, NNP, NNP, NNPS -> NOUN
        if treebank_tag.startswith('N'):
            return "NOUN"
        #RB, RBR, RBR, RBS, RP -> ADV
        if treebank_tag.startswith('R'):
            return "ADV"
        return treebank_tag


    def _add_lemma_to_dict(self, word, tag, lemma):
        if word in self.word_lemma_dict:
            self.word_lemma_dict[word][tag]=lemma
        else:
            self.word_lemma_dict[word]={tag: lemma}
    
    def _inflect_noun_singular(self, word):
        if word in self.irregular_dict:
            return self.irregular_dict[word]
        
        #rule_based Lemmatizer
        if len(word) <3:
            return word

        if word.endswith('s'):
            if len(word) > 3:
                #leaves, thieves, wives
                if word.endswith('ves'):
                    if len(word[:-3]) >2:
                        #leaves, thieves -> leaf, thief
                        return word.replace('ves', 'f')
                    else:
                        # wives -> wife
                        return word.replace('ves', 'fe')
                
                #stories, parties
                if word.endswith('ies'):
                    #->story, party
                    return word.replace('ies', 'y')

                #tomatoes, echoes
                if word.endswith('es'):
                    if word.endswith('ese') and word[-4] in self.vowels:
                        return word[:-1]
                    if word.endswith('zzes'):
                        return word.replace('zzes', 'z')
                    return word[:-2]
                if word.endswith('ys'):
                    return word.replace('ys','y')
                return word[:-1]
        return word

    def lemmatize(self, word, pos):
        if word is None:
            return ''
        if pos == None:
            pos = ''
        word = word.lower()
        pos = pos.upper()
        if pos == 'NOUN':
            return self._inflect_noun_singular(word)

        if word in self.word_lemma_dict:
            if pos in self.word_lemma_dict[word]:
                return self.word_lemma_dict[word][pos]
        return word


class LemmatizationWithPOSTagger(object):
    def __init__(self, use_nltk_lemma = False):
        self.use_nltk_lemma = use_nltk_lemma
        if self.use_nltk_lemma is not True:
            self.lemmatizer= Lemmatization()
        else:
            self.lemmatizer = WordNetLemmatizer()
        
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,token):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = nltk.pos_tag(token.split())
        return pos_tokens
    
    def lemmatize(self, tokens):
        # lemmatization using pos tagg   
        pos_tokens= [self.pos_tag(token) for token in tokens]
        
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        #pos_tokens = [[(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        
        if self.use_nltk_lemma is not True:
            pos_tokens = [" ".join([self.lemmatizer.lemmatize(word, self.lemmatizer._convert_pos(pos_tag)) for (word, pos_tag) in pos]) for pos in pos_tokens]
        else:
            pos_tokens = [" ".join([self.lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos]) for pos in pos_tokens]
        return pos_tokens