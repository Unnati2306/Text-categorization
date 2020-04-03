# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:19:37 2019

"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print(twenty_train.target)
print(twenty_train.target.size)
print(twenty_train.target_names)

count_vect=CountVectorizer(stop_words='english',lowercase=True)
print(count_vect)

X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts)
print(X_train_counts.shape) 
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) 
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english',lowercase=True)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)

sample=['God is one','OpenGL on the GPU is fast']
new_counts=count_vect.transform(sample)
new_tfidf=tfidf_transformer.transform(new_counts)
sample_predict=clf.predict(new_tfidf)
print("\nPredicted categories:")
for x in sample_predict:
    print(x)
    
print("Accuracy:",accuracy_score(twenty_test.target,predicted))
print(metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
