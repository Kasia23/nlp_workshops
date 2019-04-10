import pickle
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from string import punctuation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if os.getcwd().endswith("doc2vec"):
    # goes one folder "up". Can't be run multiple times or your work directory will get rekt
    os.chdir(os.path.dirname(os.getcwd()))

from doc2vec.constants import GENSIM_MODEL_PATH, TRAIN_PATH, TEST_PATH, PUNC_CHARS, STOPWORDS_SET, SGJP_PATH, \
    TOKENIZER_PATH
from doc2vec.preprocess_utils import prepare_sgjp_dict, clear_offers, tokenize_texts
from doc2vec.model_utils import model_cnn


def load_preprocess_data(pickle_path, stopwords_set=None, lemmatize_dict=None, remove_punct=None):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    data['text'] = data['job_name'].str.cat(data['job_content'], sep=' ')
    data.dropna(inplace=True, subset=['text'])  # drop data where we don't have text
    return data['label'], clear_offers(data=data, text_col='text', stopwords_list=stopwords_set,
                                       lemmatize_dict=lemmatize_dict, remove_punct=remove_punct)


lemmatize_dict = prepare_sgjp_dict(SGJP_PATH)

train_labels, train_data_prep = load_preprocess_data(TRAIN_PATH, stopwords_set=STOPWORDS_SET,
                                                     lemmatize_dict=lemmatize_dict,
                                                     remove_punct=PUNC_CHARS + punctuation
                                                     )

test_labels, test_data_prep = load_preprocess_data(TEST_PATH, stopwords_set=STOPWORDS_SET,
                                                   lemmatize_dict=lemmatize_dict,
                                                   remove_punct=PUNC_CHARS + punctuation
                                                   )

print(train_labels.value_counts())

all_labels = np.array(train_labels.append(test_labels))
ohe = OneHotEncoder()
ohe.fit(all_labels.reshape(-1, 1))

train_labels_bin = ohe.transform(train_labels.values.reshape(-1, 1))
test_labels_bin = ohe.transform(test_labels.values.reshape(-1, 1))

tokenizer, train_tokens = tokenize_texts(train_data_prep, TOKENIZER_PATH, oov_token='unk')
test_tokens = tokenizer.texts_to_sequences(test_data_prep.str.join(' '))

MODEL_PARAMS = {
    'maxlen': 300,
    'word_emb_size': 100,
    'filter_sizes': [5, 1],
    'num_filters': 300,
    'batch_size': 256,
    'n_epochs': 10,
    'doc_emb_size': 100,
    'is_trainable': True,
    'emb_weights': 'uniform',
    'nb_tokens': len(tokenizer.word_index) + 1,
    'output_len': len(ohe.categories_[0])
}

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data_prep)]
model = Doc2Vec(documents, vector_size=MODEL_PARAMS['doc_emb_size'], window=2, min_count=1, workers=4)

model.save(GENSIM_MODEL_PATH)

# If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

# Infer vectors for docs:
train_doc2vec = np.matrix(train_data_prep.apply(model.infer_vector).to_list())
test_doc2vec = np.matrix(test_data_prep.apply(model.infer_vector).to_list())

# Tokenize and pad offers
train_pad_tokens = pad_sequences(train_tokens, maxlen=MODEL_PARAMS['maxlen'])
test_pad_tokens = pad_sequences(test_tokens, maxlen=MODEL_PARAMS['maxlen'])

# Define the model
cnn_model = model_cnn(params=MODEL_PARAMS)
print(cnn_model.summary())

# Train the model
cnn_model.fit(train_pad_tokens, train_labels_bin, batch_size=MODEL_PARAMS['batch_size'],
              epochs=MODEL_PARAMS['n_epochs'], validation_split=0.1,
              class_weight=MODEL_PARAMS.get('class_weights'))

print(cnn_model.evaluate(test_pad_tokens, test_labels_bin))

cnn_model_doc2vec = model_cnn(params=MODEL_PARAMS, doc2vec=True)
print(cnn_model_doc2vec.summary())
cnn_model_doc2vec.fit([train_pad_tokens, train_doc2vec], train_labels_bin, batch_size=MODEL_PARAMS['batch_size'],
                      epochs=MODEL_PARAMS['n_epochs'], validation_split=0.1,
                      class_weight=MODEL_PARAMS.get('class_weights'))

print(cnn_model_doc2vec.evaluate([test_pad_tokens, test_doc2vec], test_labels_bin))
