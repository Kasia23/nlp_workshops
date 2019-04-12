import pickle
import os

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from string import punctuation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if os.getcwd().endswith("doc2vec"):
    # goes one folder "up". Can't be run multiple times or your work directory will get rekt
    os.chdir(os.path.dirname(os.getcwd()))

from doc2vec.constants import GENSIM_MODEL_PATH, TRAIN_PATH, TEST_PATH, PUNC_CHARS, STOPWORDS_SET, SGJP_PATH, \
    TOKENIZER_PATH
from doc2vec.preprocess_utils import prepare_sgjp_dict, clear_offers, tokenize_texts, tokenize, \
    prepare_gensim_word_index_dict
from doc2vec.model_utils import model_cnn, model_doc2vec, gensim_model


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
# Let's see what our labels are
print(train_labels.value_counts())

# One-hot encode our labels
all_labels = np.array(train_labels.append(test_labels))
ohe = OneHotEncoder()
ohe.fit(all_labels.reshape(-1, 1))

train_labels_bin = ohe.transform(train_labels.values.reshape(-1, 1))
test_labels_bin = ohe.transform(test_labels.values.reshape(-1, 1))

# Tokenize the texts for neural network
tokenizer, train_tokens = tokenize_texts(train_data_prep, TOKENIZER_PATH, oov_token='unk')
test_tokens = tokenizer.texts_to_sequences(test_data_prep.str.join(' '))

MODEL_PARAMS = {
    'input_len': 300,
    'word_emb_size': 100,
    'doc_emb_size': 100,
    'filter_sizes': [5, 1],
    'num_filters': 300,
    'batch_size': 256,
    'n_epochs': 10,
    'nb_tokens': len(tokenizer.word_index) + 1,
    'output_len': len(ohe.categories_[0])
}

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data_prep)]
model = Doc2Vec(documents, vector_size=MODEL_PARAMS['doc_emb_size'], window=2, min_count=1, workers=4)

model.save(GENSIM_MODEL_PATH)

# If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

# Infer Doc2Vec representations for docs:
train_doc2vec = np.matrix(train_data_prep.apply(model.infer_vector).to_list())
test_doc2vec = np.matrix(test_data_prep.apply(model.infer_vector).to_list())

# Pad offers
train_pad_tokens = pad_sequences(train_tokens, maxlen=MODEL_PARAMS['input_len'])
test_pad_tokens = pad_sequences(test_tokens, maxlen=MODEL_PARAMS['input_len'])

# 1. Pure Word2Vec model
word2vec_model = model_cnn(params=MODEL_PARAMS)
print(word2vec_model.summary())
word2vec_model.fit(train_pad_tokens, train_labels_bin, batch_size=MODEL_PARAMS['batch_size'],
                   epochs=MODEL_PARAMS['n_epochs'], validation_split=0.1)
print(word2vec_model.evaluate(test_pad_tokens, test_labels_bin))

# 2. Doc2Vec + Word2Vec model
doc2vec_word2vec_model = model_cnn(params=MODEL_PARAMS, doc2vec=True)
print(doc2vec_word2vec_model.summary())
doc2vec_word2vec_model.fit([train_pad_tokens, train_doc2vec], train_labels_bin, batch_size=MODEL_PARAMS['batch_size'],
                           epochs=MODEL_PARAMS['n_epochs'], validation_split=0.1)
doc2vec_word2vec_model.save('doc2vec_word2vec_model.h5')
print(doc2vec_word2vec_model.evaluate([test_pad_tokens, test_doc2vec], test_labels_bin))

# 3. Pure Doc2Vec model
doc2vec_model = model_doc2vec(params=MODEL_PARAMS)
print(doc2vec_model.summary())
doc2vec_model.fit(train_doc2vec, train_labels_bin, batch_size=MODEL_PARAMS['batch_size'],
                  epochs=MODEL_PARAMS['n_epochs'], validation_split=0.1)
doc2vec_model.save('doc2vec_model.h5')
print(doc2vec_model.evaluate(test_doc2vec, test_labels_bin))

# TODO: EXERCISES

# EXERCISE 1 - model
# In this exercise you will use word vectors trained by gensim.
# You need to tokenize the documents using gensim's vocabulary,
# and pass the embedding matrix to the Embedding layer in model definition. Make sure to freeze the layer weights!

gensim_emb_matrix = model.wv.vectors
gensim_word_index_dict = prepare_gensim_word_index_dict(model.wv.vocab)
gensim_train_tokens = train_data_prep.apply(tokenize, args=(gensim_word_index_dict,))
gensim_test_tokens = test_data_prep.apply(tokenize, args=(gensim_word_index_dict,))

GENSIM_MODEL_PARAMS = {
    'input_len': 300,
    'word_emb_size': 100,
    'doc_emb_size': 100,
    'filter_sizes': [5, 1],
    'num_filters': 300,
    'batch_size': 256,
    'n_epochs': 10,
    # TODO: WRITE YOUR CODE BELOW
    'nb_tokens': len(gensim_word_index_dict) + 1,
    'embedding_matrix': gensim_emb_matrix,
    # TODO: END OF YOUR CHANGES
    'output_len': len(ohe.categories_[0])
}

gensim_padded_train_tokens = pad_sequences(gensim_train_tokens, maxlen=MODEL_PARAMS['input_len'])
gensim_padded_test_tokens = pad_sequences(gensim_test_tokens, maxlen=MODEL_PARAMS['input_len'])

pure_gensim_model = gensim_model(params=GENSIM_MODEL_PARAMS)
print(pure_gensim_model.summary())
pure_gensim_model.fit([gensim_padded_train_tokens, train_doc2vec], train_labels_bin,
                      batch_size=GENSIM_MODEL_PARAMS['batch_size'],
                      epochs=GENSIM_MODEL_PARAMS['n_epochs'], validation_split=0.1)
pure_gensim_model.save('pure_gensim_model.h5')
print(pure_gensim_model.evaluate([gensim_padded_test_tokens, test_doc2vec], test_labels_bin))

# EXERCISE 2 - working with doc2vec representations
# Choose 3 different offers.
# For each of those, find 10 that are most similar using the gensim model and print their content to console
for index, offer in train_data_prep.iloc[:3].iteritems():
    # TODO: WRITE YOUR CODE BELOW
    similarities = [(ind, model.wv.n_similarity(offer, train_data_prep[ind])) for ind, off in
                    train_data_prep.drop(index).iteritems()]
    three_best_offers = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
    print('\n Wyjściowa oferta')
    print(' '.join(offer))

    print('\n Najlepsze rekomendacje')
    for ind, similarity in three_best_offers:
        print(' '.join(train_data_prep[ind]))
    # TODO: END OF YOUR CHANGES
