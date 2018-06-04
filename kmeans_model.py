from __future__ import print_function
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
import sys
from time import time

import numpy as np

class KMeansModel(object):

    def __init__(self,dataset,modeling_list,
                 use_hashing,use_idf,
                 n_clusters,
                 n_features,
                 verbose
                 ):
        self.dataset = dataset
        self.data = dataset[modeling_list]
        self.n_features = n_features
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.verbose = verbose
        self.n_clusters = n_clusters
        self.labels = dataset.category.astype("category").cat.codes



    def vectorize(self):
        print("Extracting features from the training dataset using a sparse vectorizer")

        t0 = time()
        if self.use_hashing:
            if self.use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=self.n_features,
                                           stop_words='english', alternate_sign=False,
                                           norm=None, binary=False)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                               stop_words='english',
                                               alternate_sign=False, norm='l2',
                                               binary=False)
        else:
            self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=self.n_features,
                                         min_df=2, stop_words='english',
                                         use_idf=self.use_idf)
        self.X = self.vectorizer.fit_transform(self.data)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % self.X.shape)
        print()


    def apply_lsa(self,n_components):
        self.n_components = n_components
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        self.svd = TruncatedSVD(self.n_components)
        normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(self.svd, normalizer)

        self.X = self.lsa.fit_transform(self.X)

        with open('sv.pkl', 'wb') as f:
            pickle.dump(self.svd.singular_values_,f)

        self.dataset['component_1'] = self.X[:,0].copy()
        self.dataset['component_2'] = self.X[:, 1].copy()
        self.dataset['component_3'] = self.X[:, 2].copy()


        print("done in %fs" % (time() - t0))


        explained_variance = self.svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))


    def run(self,miniBatch = False):
        self.miniBatch = miniBatch

        if self.miniBatch:
            self.km = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=self.verbose)
        else:
            self.km = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=100, n_init=1,
                        verbose=self.verbose)

        print("Clustering sparse data with %s" % self.km)
        t0 = time()
        self.km.fit(self.X)
        print("done in %0.3fs" % (time() - t0))
        print()

        self.dataset['cluster'] = self.km.labels_.copy()


    def get_metrics(self):
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.labels, self.km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(self.labels, self.km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.labels, self.km.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(self.labels, self.km.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, self.km.labels_, sample_size=1000))

        print()

        if not self.use_hashing:
            print("Top terms per cluster:")

            if self.n_components:
                original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]

            terms = self.vectorizer.get_feature_names()
            for i in range(self.n_clusters):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()