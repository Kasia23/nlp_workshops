import pickle
import datetime
import pandas as pd
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def prepare_sgjp_dict(path):
    sgjp_df = pd.read_csv(path, skiprows=29, nrows=None, sep='\t', names=['forma', 'lemat', 'int', 'inne1', 'inne2'])
    sgjp_df.drop_duplicates('forma', inplace=True)
    sgjp_df['lemat'] = sgjp_df['lemat'].apply(lambda x: x.split(':')[0])
    sgjp_df.set_index('forma', inplace=True)
    sgjp_df['lemat'] = sgjp_df['lemat'].str.lower()
    return sgjp_df['lemat'].to_dict()


def remove_stopwords(content, stopwords_set):
    content = [item for item in content if item not in stopwords_set]
    return content


def lemmatize_word_list(word_list, form_lemma_dict, unk_out):
    if unk_out:  # if we can't find the word, what shall we do? Option 1: throw out the word
        return [form_lemma_dict.get(x).lower() for x in word_list if form_lemma_dict.get(x).lower()]
    else:  # option 2: leave the word as it is
        return [form_lemma_dict.get(x, x).lower() for x in word_list]


def tokenize_texts(content_frame, tokenizer_path=None, oov_token=None):
    tokenizer = Tokenizer(oov_token=oov_token)
    texts = content_frame.str.join(' ')
    tokenizer.fit_on_texts(texts)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer, tokenizer.texts_to_sequences(texts)


def clear_offers(data, text_col, stopwords_list=None, lemmatize_dict=None, remove_punct=''):
    # Clearing data
    seq = data[text_col].apply(text_to_word_sequence)
    seq = seq.apply(lambda x: [item.strip(remove_punct) if isinstance(item, str) else item for item in x])
    print(datetime.datetime.now(), ' Clearing data - DONE')

    # Removing stopwords
    if stopwords_list:
        seq = seq.apply(remove_stopwords, args=(stopwords_list,))
        print(datetime.datetime.now(), ' Removing stopwords - DONE')

    # lemmatyzacja
    if lemmatize_dict:
        seq = seq.apply(lemmatize_word_list, args=(lemmatize_dict, False))
        print(datetime.datetime.now(), ' Lemmatizing - DONE')

    return seq


def tokenize(sequence, word_index_dict):
    """
    :param sequence: sequence that we want tokenized
    :param word_index_dict:
           dictionary with keys - words from texts and values - indices that correspond to embedding matrix rows
    :return: sequence (a list) of tokens (indeces from embedding matrix)
    """
    return [word_index_dict[x] for x in sequence if word_index_dict.get(x)]


def prepare_gensim_word_index_dict(gensim_wv_vocab):
    return {k: v.index for (k, v) in gensim_wv_vocab.items()}
