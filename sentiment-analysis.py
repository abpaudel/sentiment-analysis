import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# train_path = "../resource/lib/publicdata/aclImdb/train/"
# test_path = "../resource/asnlib/public/imdb_te.csv"

train_path = './data/aclImdb/train/'
test_path = './data/imdb_te.csv'

stopwords = open('./stopwords.en.txt').read().split()

def make_csv(inpath, outpath='./data/', name='imdb_tr.csv'):
    train_data = load_files(inpath, encoding='utf-8', categories=['neg', 'pos'])
    tr_data  = {'text': train_data.data, 'polarity': train_data.target}
    pd.DataFrame.from_dict(data=tr_data).to_csv(outpath+name, columns=['text', 'polarity'])
    return outpath+name  
    
def unigram_tdm():
    pipeline = Pipeline([('vectorizer',  CountVectorizer(stop_words=stopwords)),
                        ('classifier',  SGDClassifier(loss='hinge', penalty='l1')) ])
    return pipeline

def bigram_tdm():
    pipeline = Pipeline([('vectorizer',  CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))),
                        ('classifier',  SGDClassifier(loss='hinge', penalty='l1'))])
    return pipeline

def unigram_tfidf():
    pipeline = Pipeline([('vectorizer',  CountVectorizer(stop_words=stopwords)),
                        ('tfidf_transformer',  TfidfTransformer()),
                        ('classifier',  SGDClassifier(loss='hinge', penalty='l1')) ])
    return pipeline

def bigram_tfidf():
    pipeline = Pipeline([('vectorizer',  CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))),
                        ('tfidf_transformer',  TfidfTransformer()),
                        ('classifier',  SGDClassifier(loss='hinge', penalty='l1'))])
    return pipeline

  
if __name__ == "__main__":
    train_df = pd.read_csv(make_csv(train_path), encoding = 'ISO-8859-1')
    test_df = pd.read_csv(test_path, encoding = 'ISO-8859-1')

    models = [(unigram_tdm(), 'unigram.output.txt'),
            (bigram_tdm(), 'bigram.output.txt'),
            (unigram_tfidf(), 'unigramtfidf.output.txt'),
            (bigram_tfidf(), 'bigramtfidf.output.txt')]

    for model in models:
        model[0].fit(train_df['text'], train_df['polarity'])
        output = model[0].predict(test_df['text'])
        file = open('./data/'+model[1], 'w')
        file.write('\n'.join(output.astype(str)))
        file.close()
