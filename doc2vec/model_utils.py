from keras.layers import *
from keras.models import Model


def model_cnn(params, doc2vec=False):
    inp = Input(shape=(params['maxlen'],))
    x = Embedding(params['nb_tokens'], params['word_emb_size'], embeddings_initializer=params['emb_weights'],
                  trainable=params["is_trainable"])(inp)
    x = Reshape((params['maxlen'], params['word_emb_size'],))(x)  # TODO: wyjebać jak to niepotrzebne

    maxpool_pool = []
    for i in range(len(params["filter_sizes"])):
        conv = Conv1D(params['num_filters'], kernel_size=params["filter_sizes"][i],
                      kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPooling1D(pool_size=params['word_emb_size'], strides=None, padding="valid")(conv))
    # TODO: gdzieś dodać doc2vecowe rzeczy
    x = Concatenate(axis=1)(maxpool_pool)
    x = Flatten()(x)
    x = Dropout(0.1)(x)

    outp = Dense(params['output_len'], activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
