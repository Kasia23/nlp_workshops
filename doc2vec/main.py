import pickle
import os

import numpy as np

from string import punctuation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if os.getcwd().endswith("doc2vec"):
    # goes one folder "up". Can't be run multiple times or your work directory will get rekt
    os.chdir(os.path.dirname(os.getcwd()))

from doc2vec.constants import GENSIM_MODEL_PATH, TRAIN_PATH, TEST_PATH, PUNC_CHARS, STOPWORDS_SET, SGJP_PATH
from doc2vec.preprocess_utils import prepare_sgjp_dict, clear_offers
from doc2vec.model_utils import model_cnn


def load_preprocess_data(pickle_path, stopwords_set=None, lemmatize_dict=None, remove_punct=None):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    data['text'] = data['job_name'].str.cat(data['job_content'], sep=' ')
    return clear_offers(data=data, text_col='text', stopwords_list=stopwords_set,
                        lemmatize_dict=lemmatize_dict, remove_punct=remove_punct
                        )


MODEL_PARAMS = {
    'maxlen': 300,
    'word_emb_size': 100,
    'filter_sizes': [5, 1],
    'num_filters': 300,
    'batch_size': 512,
    'n_epochs': 2,
    'doc_emb_size': 100,
    'nb_tokens': 300,
    'is_trainable': True,
    'emb_weights': 'uniform'
}

lemmatize_dict = prepare_sgjp_dict(SGJP_PATH)

train_data_prep = load_preprocess_data(TRAIN_PATH, stopwords_set=STOPWORDS_SET,
                                       lemmatize_dict=lemmatize_dict,
                                       remove_punct=PUNC_CHARS + punctuation
                                       )

test_data_prep = load_preprocess_data(TEST_PATH, stopwords_set=STOPWORDS_SET,
                                      lemmatize_dict=lemmatize_dict,
                                      remove_punct=PUNC_CHARS + punctuation
                                      )

MODEL_PARAMS['output_len'] = 200

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data_prep)]
model = Doc2Vec(documents, vector_size=MODEL_PARAMS['doc_emb_size'], window=2, min_count=1, workers=4)

model.save(GENSIM_MODEL_PATH)

# If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

# Infer vectors for docs:
doc2vec_train = np.matrix(train_data_prep.apply(model.infer_vector).to_list())

cnn_model = model_cnn(params=MODEL_PARAMS)
print(cnn_model.summary())

cnn_model_doc2vec = model_cnn(params=MODEL_PARAMS, doc2vec=True)
print(cnn_model_doc2vec.summary())
