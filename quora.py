import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
import collections
import sklearn as sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction import text 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import gensim
from collections import defaultdict
from itertools import islice
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from keras import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D,Dense, Input, Embedding, Dropout, LSTM, CuDNNGRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import time
%matplotlib inline
warnings.filterwarnings('ignore')

train_df = pd.DataFrame.from_csv("../input/train.csv")
test_df =  pd.DataFrame.from_csv("../input/test.csv")

train_df.head(5)


text =" ".join(train_df.question_text)
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Target Status")
explode=(0,0.1)
labels ='0','1'
ax.pie(list(dict(collections.Counter(list(train_df.target))).values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


X_train, X_test, y_train, y_test = train_test_split(train_df.question_text, train_df.target, test_size=0.33, random_state=42)

count_vectorizer = CountVectorizer(min_df=5, stop_words='english')
vect = count_vectorizer.fit(train_df.question_text)

X_vect_train = vect.transform(X_train) # documents-terms matrix of training set
X_vect_test = vect.transform(X_test) # documents-terms matrix of testing set

tf_train_transformer = TfidfTransformer(use_idf=False).fit(X_vect_train)
tf_test_transformer =  TfidfTransformer(use_idf=False).fit(X_vect_test)

xtrain_tf = tf_train_transformer.transform(X_vect_train)
xtest_tf = tf_test_transformer.transform(X_vect_test)
type(xtrain_tf),xtrain_tf.shape, train_df.shape

count_vectorizer.get_feature_names()[-5:]



results_df = pd.DataFrame()

# MULTINOMINA_NAIVE_BAYES
nb_ = MultinomialNB()
nb_clf = nb_.fit(X=xtrain_tf, y=y_train)
results_df.set_value("NB" , "countVectorizer" , accuracy_score(y_test,nb_clf.predict(xtest_tf)))

# RANDOM_FORES_CLASSFIER
rf_clf = RandomForestClassifier(n_estimators=25, max_depth=15,random_state=42)
rf_clf.fit(X=xtrain_tf,y=y_train)
results_df.set_value("RF" , "countVectorizer" , accuracy_score(y_test,rf_clf.predict(xtest_tf)))

#LoggicRegression
lreg_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=42)
lreg_clf.fit(X=xtrain_tf, y=y_train)                         
results_df.set_value("LREG" , "countVectorizer" , accuracy_score(y_test,lreg_clf.predict(xtest_tf)))

results_df


fig,axes=plt.subplots(1,1,figsize=(8,8))
axes.set_ylabel("Accuracy")
plt.ylim((.92,.97))
results_df.plot(kind="bar",ax=axes)

text_ = train_df.question_text
targets_ = train_df.target
class GetSentences(object):
    def __iter__(self):
        counter = 0
        for sentence_iter in text_:
            tmp_sentence = sentence_iter
            counter += 1
            yield tmp_sentence.split()
len(text_)


num_features = 200  # Word vector dimensionality
min_word_count = 5  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words
get_sentence = GetSentences()
model = gensim.models.Word2Vec(sentences=get_sentence, min_count=min_word_count, size=num_features, workers=4)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

# Most Similar Word
model.most_similar(positive=["LNMIIT"])

model.save('word2vec.model')

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(next(iter(self.word2vec.items()))[1])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(next(iter((word2vec.items()))))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


       # 1- MEAN VECTORIZER
etree_w2v = Pipeline([("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "w2v_mean", accuracy_score(y_test, etree_w2v.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_w2v_tfidf = Pipeline([("w2v tfidf vectorizer", TfidfEmbeddingVectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_w2v_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "w2v_tfidf", accuracy_score(y_test, etree_w2v_tfidf.predict(X_test)))

####SVM####

# 1- MAIN VECTORIZER
svm_w2v = Pipeline([("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)), ("SVM", LinearSVC(random_state=0, tol=1e-4))])
# svm_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "w2v_mean", accuracy_score(y_test, etree_w2v_tfidf.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), ("SVM", LinearSVC(random_state=0, tol=1e-4))])
# svm_w2v_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "w2v_tfidf", accuracy_score(y_test, svm_w2v_tfidf.predict(X_test)))

####MLP####

# 1- MAIN VECTORIZER
mlp_w2v = Pipeline(
    [("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10, 2), random_state=1))])
# mlp_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "w2v_mean", accuracy_score(y_test, mlp_w2v.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_w2v_tfidf = Pipeline(
    [("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10,2), random_state=1))])
results_df.set_value("NB","w2v_mean",0.476575)
results_df.set_value("ExtraTree","w2v_mean",0.939119)
results_df.set_value("SVM","w2v_mean",0.939144)
results_df.set_value("MLP","w2v_mean",0.939035)
results_df.set_value("LREG","w2v_mean",0.938836)

results_df.set_value("NB","w2v_tfidf",0.260479)
results_df.set_value("ExtraTree","w2v_tfidf",0.939144)
results_df.set_value("SVM","w2v_tfidf",0.939033)
results_df.set_value("MLP","w2v_tfidf",0.939035)
results_df.set_value("LREG","w2v_tfidf",0.953311)
fig,axes=plt.subplots(1,1,figsize=(8,8))
axes.set_ylabel("Accuracy")
axes.set_title("word2vec results for 67% training and 23% testing")
# plt.ylim((.93,.95))
results_df[["w2v_mean","w2v_tfidf"]].dropna().plot(kind="bar",ax=axes)


del w2v
del model
import gc; gc.collect()
time.sleep(10)

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
glov = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

nb_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("Guassian NB", GaussianNB())])
# nb_glov.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "glov_mean", accuracy_score(y_test, nb_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
nb_glov_tfidf = Pipeline([("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("Guassian NB", MultinomialNB())])
# nb_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "glov_tfidf", accuracy_score(y_test, nb_glov_tfidf.predict(X_test)))

###EXTRA TREE####

# 1- MEAN VECTORIZER
etree_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_glov.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "glov_mean", accuracy_score(y_test, etree_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "glov_tfidf", accuracy_score(y_test, etree_glov_tfidf.predict(X_test)))

####SVM####

# 1- MAIN VECTORIZER
svm_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("SVM", LinearSVC(random_state=42, tol=1e-5))])
# svm_glov.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "glov_mean", accuracy_score(y_test, svm_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "glov_tfidf", accuracy_score(y_test, svm_glov_tfidf.predict(X_test)))

####MLP####

# 1- MAIN VECTORIZER
mlp_glov = Pipeline(
    [("glov mean vectorizer", MeanEmbeddingVectorizer(glov)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 2), random_state=42))])
# mlp_glov.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "glov_mean", accuracy_score(y_test, mlp_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 2), random_state=42))])
# mlp_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "glov_tfidf", accuracy_score(y_test, mlp_glov_tfidf.predict(X_test)))


####LREG####

# 1- MAIN VECTORIZER
lreg_glov = Pipeline(
    [("glov mean vectorizer", MeanEmbeddingVectorizer(glov)),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_glov.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "glov_mean", accuracy_score(y_test, lreg_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
lreg_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("LEREG", "glov_tfidf", accuracy_score(y_test, lreg_glov_tfidf.predict(X_test)))
results_df.set_value("NB","glov_mean",0.533364)
results_df.set_value("ExtraTree","glov_mean",0.939131)
results_df.set_value("SVM","glov_mean",0.939140)
results_df.set_value("MLP","glov_mean",0.938973)
results_df.set_value("LREG","glov_mean",0.938836)

results_df.set_value("NB","glov_tfidf",0.941720)
results_df.set_value("ExtraTree","glov_tfidf",0.945798)
results_df.set_value("SVM","glov_tfidf",0.953854)
results_df.set_value("MLP","glov_tfidf",0.939035)
results_df.set_value("LREG","glov_tfidf",0.953311)

results_df


fig,axes=plt.subplots(1,1,figsize=(15,8))
plt.ylim((.5,1))
axes.set_ylabel("Accuracy")
axes.set_title("Traditional Classfieris Results for 67% Training and 23% Testing with Two Types of Embedding")
results_df[results_df.index != "RF"].plot(kind="bar",ax=axes)


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Done")