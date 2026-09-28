import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.cv = CountVectorizer(token_pattern=r"\S+", dtype=np.float32)
        tf = self.cv.fit_transform(docs).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean() if dl.mean() > 0 else 1.0
        denom = self.k1 * (1 - self.b + self.b * dl / avg)
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + denom[rows]) * idf[tf.indices]
        self.mat_t = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).T.tocsr()
        return self

    def query_matrix(self, queries):
        q = self.cv.transform(queries)
        q.data[:] = 1
        return q.astype(np.float32).tocsr()

    def score(self, qmat):
        return (qmat @ self.mat_t).tocsr()


class CharTfidf:
    # символьные n-граммы спасают от опечаток и сокращений типа "англ"
    def __init__(self, ngram=(3, 5), max_features=600_000):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2,
                                   max_features=max_features, sublinear_tf=True, dtype=np.float32)

    def fit(self, docs):
        self.mat_t = self.vec.fit_transform(docs).T.tocsr()
        return self

    def score(self, queries):
        return (self.vec.transform(queries) @ self.mat_t).tocsr()
