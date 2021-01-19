import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9а-яё #+_]')

def preprocess_text(text):
    text = text.lower()
    
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    
#     tokens = mystem.lemmatize(text)
    tokens = [word for word in text.split()]
    
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation
             ]
    
    text = " ".join(tokens)
    
    return text


def get_metrics(y_true, y_pred, group_val):
    for group in set(group_val):
        mask = np.where(group_val == group)
        
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask])
        print("{}, accuracy: {:.3f}, f1: {:.3f}".format(group, acc, f1))
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("accuracy: {:.3f}, f1: {:.3f}".format(acc, f1))
