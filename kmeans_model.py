from __future__ import print_function
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

class KMeansModel(object):
    def __init__(self,dataset,
                 modeling_list,
                 n_clusters,
                 n_features,
                 ):
        '''
        :param dataset          : pandas dataframe for data I/O
        :param modeling_list    : column of text to be used by clustering
        :param n_clusters       : number of cluster for the k-means algorithm
        :param n_features       : number of features for the count matrix
        '''
        self.dataset = dataset
        self.data = dataset[modeling_list]
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.labels = dataset.category.astype("category").cat.codes

    def vectorize(self):
        '''
        Create tfidf matrix
        '''
        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()
        self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=self.n_features,
                                         min_df=2, stop_words='english')
        self.X = self.vectorizer.fit_transform(self.data)
        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % self.X.shape)
        print()


    def apply_lsa(self,n_components):
        '''
        Perform LSA
        :param n_components: number of components for LSA
        :return:
        '''
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
        '''
        Perform k-means clustering
        :param miniBatch: Boolean to select between regular k-means and mini batch
        :return:
        '''
        self.miniBatch = miniBatch
        if self.miniBatch:
            self.km = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000)
        else:
            self.km = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=100, n_init=1)

        print("Clustering sparse data with %s" % self.km)
        t0 = time()
        self.km.fit(self.X)
        print("done in %0.3fs" % (time() - t0))
        print()

        # update clustering
        self.dataset['cluster'] = self.km.labels_.copy()


    def get_metrics(self):
        '''
        Print metrics for the clustering
        Print top term of each cluster
        '''
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.labels, self.km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(self.labels, self.km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.labels, self.km.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(self.labels, self.km.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, self.km.labels_, sample_size=1000))

        print()

        print("Top terms per cluster:")

        original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = self.vectorizer.get_feature_names()
        for i in range(self.n_clusters):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()