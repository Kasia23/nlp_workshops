import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle 


from collections import defaultdict
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Cleaner:
    def __init__(self):
        self.morfeusz = defaultdict(str, pickle.load(open('./data/polish_dict.pickle', 'rb')))
        
    def lemmatize(self, text):
        text = str(text).split()
        for i, word in enumerate(text):
            if len(word) > 2:            
                if word.endswith('.'):
                    lemma = self.morfeusz[word.replace('.','')]+" ."
                else:
                    lemma = self.morfeusz[word]
                if len(lemma) == 0:
                    lemma = word
                text[i] = lemma
            else:
                text[i] = ''
        text = ' '.join(x for x in text if len(x) > 0)
        return text

class Network:
    def __init__(self, args):
        self.__dict__.update(args)

class Data:
    def __init__(self, data_focus):
        self.data_focus = data_focus
        self.training_set = pd.read_pickle('./data/train_offers.pickle')
        self.test_set = pd.read_pickle('./data/test_offers.pickle')
        self.raw_train_X, self.raw_val_X, self.raw_train_y, self.raw_val_y = train_test_split(self.training_set[data_focus], self.training_set['label'], test_size=0.15, random_state=400)
        self.raw_test_X = self.test_set[data_focus]
        self.raw_test_y = self.test_set['label']
        self.tokenizer = None
        self.cleaner = Cleaner()
        self.unique_class_names = sorted(list(pd.unique(self.raw_train_y)) + ["\t", "\n"])
                
    def fit_tokenizer(self, data, max_number_words):
        tokenizer = Tokenizer(max_number_words)
        tokenizer.fit_on_texts(data)
        self.tokenizer = tokenizer
        return
    
    def add_lemma_eos(self, data):
        return data.apply(lambda x: self.lemmatize(x) + " <eos>")
    
    def preprocess_data(self, max_number_words, max_seq_len):
        self.train_X = self.add_lemma_eos(self.raw_train_X)
        self.val_X = self.add_lemma_eos(self.raw_val_X)
        self.test_X = self.add_lemma_eos(self.raw_test_X)
        self.fit_tokenizer(self.train_X, max_number_words)
        self.train_X = self.text_to_vector(self.train_X, max_seq_len)
        self.val_X = self.text_to_vector(self.val_X, max_seq_len)
        self.test_X = self.text_to_vector(self.test_X, max_seq_len)
        self.train_y, self.train_y_input, self.train_y_output = self.full_transform_labels(self.raw_train_y)
        self.val_y, self.val_y_input, self.val_y_output = self.full_transform_labels(self.raw_val_y)
        self.test_y, self.test_y_input, self.test_y_output = self.full_transform_labels(self.raw_test_y)
        return 
    
    def lemmatize(self, text):
        return self.cleaner.lemmatize(text)
    
    def idx_to_word(self, idx):
        word = self.tokenizer.word_index[idx]
        return word
    
    def text_to_vector(self, data, max_seq_len):
        x = self.tokenizer.texts_to_sequences(data)
        x = pad_sequences(x, maxlen=max_seq_len, padding='post')
        return x
    
    def full_transform_labels(self, labels, max_decoder_seq_length=3, num_decoder_tokens=37):
        labels = labels.apply(lambda x: ("\t " + x.replace(" ", "_") + " \n").split(" "))
        inp, out = self.transform_labels(labels, max_decoder_seq_length, num_decoder_tokens)
        return labels, inp, out 
    
    def transform_labels(self, labels, max_decoder_seq_length=3, num_decoder_tokens=37):
        labels_names = sorted(list(set([y for x in labels for y in x])))
        labels_transformed_input = np.zeros((len(labels), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
        labels_transformed_output = np.zeros((len(labels), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
        for i, categories in enumerate(labels):
            if not type(categories)==list: 
                categories = [categories] #here ensure that we are dealing with a list 
            for t, category in enumerate(categories):
                labels_transformed_input[i, t, labels_names.index(category)] = 1.
                if category=='\n':
                    labels_transformed_input[i, t+1:, labels_names.index(category)] = 1.
                if t > 0:
                    labels_transformed_output[i, t - 1, labels_names.index(category)] = 1.
                    if category=='\n':
                        labels_transformed_output[i, t:, labels_names.index(category)] = 1.
        return labels_transformed_input, labels_transformed_output
        

"""load word embeddings from a given file, returns a dict, where a key is a word and a value is an embedding"""
def load_word_embeddings(vecs):
    f = open(vecs, 'r')
    line = f.readline()
    split_line = line.split()
    length = int(split_line[1])
    model = {}
    for line in f:
        try:
            split_line = line.split()
            word = ' '.join(split_line[:-length])
            
            embedding = np.array([float(val) for val in split_line[-length:]])
            model[word] = embedding
        except:
            print('Couldnt read {}'.format(word))
            break
    return model

def generate_embedding_matrix(embeddings, word_index, shape=300,  max_num_words=20000):
    embedding_matrix = np.zeros((len(word_index)+1, shape))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.rand(shape)
    embedding_matrix=embedding_matrix[:max_num_words+1]
    return embedding_matrix



