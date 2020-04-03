# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.  
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print(twenty_train.target)
print(twenty_train.target.size)
print(twenty_train.target_names)

print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts)
#print(count_vect.get_feature_names())
print(X_train_counts.shape)
#print(X_train_counts.toarray())
 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) 
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
nb = np.mean(predicted == twenty_test.target)

text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
alpha=1e-3, max_iter=5, random_state=30)),])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
svm = np.mean(predicted_svm == twenty_test.target)

print("Naive Bayes:",nb)
print("SVM:svm",svm)
 