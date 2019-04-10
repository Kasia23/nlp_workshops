import pickle
import codecs
import pandas as pd
from tqdm import tqdm
from keras.callbacks import *
from keras.layers import *
from keras.optimizers import Adam, Adadelta
from keras.models import Model
from tensorflow.contrib.keras.api.keras.initializers import Constant
from sklearn.metrics import classification_report, confusion_matrix


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def conf_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, columns=["False", "True"], index=["False", "True"])
    cm.index.name, cm.columns.name = 'Actual', 'Predicted'
    return cm


def calculate_preds_binary(preds):
    preds_binary = []
    for x in list(preds):
        if x < 0.5:
            preds_binary.append(0)
        else:
            preds_binary.append(1)
    return preds_binary


def one_or_zero(number, k):
    return 1 if number >= k else 0


def load_embeddings(emb_path, nrows=None):
    # load embeddings
    embeddings_index = {}
    f = codecs.open(emb_path, encoding='utf-8')
    for i, line in enumerate(tqdm(f)):
        if (nrows is not None) and i > nrows:
            break
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index


def prepare_embedding_matrix(params, word_index, emb_path, emb, nrows):
    if emb == "fb_emb":
        embeddings_index = load_embeddings(emb_path, nrows)
        words_not_found = []
        embedding_matrix = np.zeros((params['nb_tokens'], params['emb_len']))
        for word, i in word_index.items():
            if i - 1 >= params['nb_tokens']:
                continue
            embedding_vector = embeddings_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i - 1] = embedding_vector[0:params['emb_len']]
            else:
                words_not_found.append(word)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        return Constant(embedding_matrix), embeddings_index, False
    else:
        return 'uniform', None, True


def build_model_blstm(params, emb_weights):
    lstm_units = params['lstm_units']
    nb_tokens = params['nb_tokens']
    maxlen = params['maxlen']
    offer_rep_dim = params['offer_rep_dim']
    emb_len = params['emb_len']
    distance = params['distance']
    is_trainable = params['is_trainable']

    input_1 = Input(shape=(maxlen,), dtype='int32')
    input_2 = Input(shape=(maxlen,), dtype='int32')

    emb_layer = Embedding(nb_tokens, output_dim=emb_len, input_length=maxlen, mask_zero=False,
                          embeddings_initializer=emb_weights, trainable=is_trainable)
    blstm_layer = Bidirectional(LSTM(units=lstm_units, return_sequences=True), merge_mode='concat', weights=None)
    dense = Dense(offer_rep_dim, activation='sigmoid')

    blstm_encoders = []
    for char_array in [input_1, input_2]:
        embs = emb_layer(char_array)
        blstm = blstm_layer(embs)
        dropout = Dropout(0.15)(blstm)
        dense_ = dense(dropout)
        flatten = Flatten()(dense_)
        blstm_encoders.append(flatten)
    if distance == 'cosine':
        distance = Dot([1, 1], normalize=True)(blstm_encoders)
    else:  # manhattan
        distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))(blstm_encoders)
    model = Model([input_1, input_2], [distance])

    if params['optimizer'] == 'adam':
        optimizer = Adam()
    elif params['optimizer'] == 'adadelta':
        optimizer = Adadelta(clipnorm=1.)
    else:
        raise ValueError

    if params['loss'] == 'contrast':
        loss = contrastive_loss
    elif params['loss'] == 'mse':
        loss = 'mean_squared_error'
    elif params['loss'] == 'bin':
        loss = 'binary_crossentropy'
    else:
        raise ValueError

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', 'mae', 'cosine'])
    return model


def model_statistics(preds_binary, y_true):
    print("Liczba poprawnie przewidzianych ogłoszeń: ",
          str(sum(preds_binary == y_true.reset_index(drop=True))))
    print("Liczba wszystkich ogłoszeń w zbiorze testowym: ", str(len(preds_binary)))
    cm = conf_matrix(y_true, preds_binary)
    print('Confusion matrix: \n{}'.format(cm))
    metrics = classification_report(y_true, preds_binary)
    print('Metryki: \n{}'.format(metrics))

    return cm, metrics
