import random
import datetime
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from langdetect import detect
from bs4 import BeautifulSoup, NavigableString
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from time import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def remove_stopwords(content):
    content = [item for item in content if item not in list(stopwords.words('english'))
]
    return content

def lemmatize_word_list(word_list, form_lemma_dict, unk_out):
    if unk_out:  # if we can't find the word, what shall we do? Option 1: throw out the word
        return [form_lemma_dict.get(x).lower() for x in word_list if form_lemma_dict.get(x).lower()]
    else:  # option 2: leave the word as it is
        return [form_lemma_dict.get(x, x).lower() for x in word_list]


def prepare_representation(content_frame, oov_token=None):
    tokenizer = Tokenizer(oov_token=oov_token)
    texts = content_frame.str.join(' ')
    tokenizer.fit_on_texts(texts)
    return tokenizer, pd.DataFrame(pd.Series(tokenizer.texts_to_sequences(texts), index=content_frame.index))


def clear_text(text, strip_chars, is_remove_stopwords=True, is_lemmatize=True):
    # czyszczenie danych
    seq = text.apply(text_to_word_sequence)
    seq = seq.apply(
        lambda x: [item.lstrip(strip_chars).rstrip(strip_chars) if isinstance(item, str) else item for item in x])
    print(datetime.datetime.now(), 'Oczyszczenie danych - SUKCES')

    # usuwanie stopwordsow
    if is_remove_stopwords:
        seq = seq.apply(remove_stopwords)
        print(datetime.datetime.now(), 'Usuniecie stopwordsow - SUKCES')

    # lemmatyzacja
    if is_lemmatize:
        lemmatizer = WordNetLemmatizer()
        seq = seq.apply(lambda x: [lemmatizer.lemmatize(each) for each in x])
        print(datetime.datetime.now(), 'Lemmatyzacja - SUKCES')

    return seq


