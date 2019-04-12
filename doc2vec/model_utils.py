from keras.layers import *
from keras.models import Model
from keras.initializers import Constant


def model_cnn(params, doc2vec=False):
    text_input = Input(shape=(params['input_len'],))
    x = Embedding(params['nb_tokens'], params['word_emb_size'])(text_input)
    maxpool_pool = []
    for i in range(len(params["filter_sizes"])):
        conv = Conv1D(params['num_filters'], kernel_size=params["filter_sizes"][i],
                      kernel_initializer='he_normal', activation='relu', padding='same')(x)
        maxpool_pool.append(MaxPooling1D(pool_size=params['word_emb_size'], strides=None, padding="valid")(conv))
    x = Concatenate(axis=1)(maxpool_pool)
    x = Flatten()(x)

    if doc2vec:
        doc2vec_input = Input(shape=(params['doc_emb_size'],))
        x = Concatenate(axis=1)([x, doc2vec_input])

    x = Dropout(0.1)(x)
    x = Dense(params['output_len'] * 2)(x)
    x = Dropout(0.1)(x)

    outp = Dense(params['output_len'], activation="sigmoid")(x)
    if doc2vec:
        model = Model(inputs=[text_input, doc2vec_input], outputs=outp)
    else:
        model = Model(inputs=text_input, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_doc2vec(params):
    doc2vec_input = Input(shape=(params['doc_emb_size'],))
    x = Dense(params['output_len'] * 4)(doc2vec_input)
    x = Dropout(0.1)(x)
    x = Dense(params['output_len'] * 4)(x)
    x = Dropout(0.1)(x)
    x = Dense(params['output_len'] * 2)(x)
    x = Dropout(0.1)(x)

    outp = Dense(params['output_len'], activation="sigmoid")(x)
    model = Model(inputs=doc2vec_input, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def gensim_model(params):
    text_input = Input(shape=(params['input_len'],))
    # TODO: WRITE YOUR CODE BELOW
    x = Embedding(params['nb_tokens'], params['word_emb_size'],
                  embeddings_initializer=Constant(params['embedding_matrix']), trainable=False)(text_input)
    # TODO: END OF YOUR CHANGES
    maxpool_pool = []
    for i in range(len(params["filter_sizes"])):
        conv = Conv1D(params['num_filters'], kernel_size=params["filter_sizes"][i],
                      kernel_initializer='he_normal', activation='relu', padding='same')(x)
        maxpool_pool.append(MaxPooling1D(pool_size=params['word_emb_size'], strides=None, padding="valid")(conv))
    x = Concatenate(axis=1)(maxpool_pool)
    x = Flatten()(x)

    doc2vec_input = Input(shape=(params['doc_emb_size'],))
    x = Concatenate(axis=1)([x, doc2vec_input])

    x = Dropout(0.1)(x)
    x = Dense(params['output_len'] * 2)(x)
    x = Dropout(0.1)(x)

    outp = Dense(params['output_len'], activation="sigmoid")(x)
    model = Model(inputs=[text_input, doc2vec_input], outputs=outp)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
