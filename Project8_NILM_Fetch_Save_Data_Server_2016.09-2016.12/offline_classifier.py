# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:03:42 2016


"""



import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import newaxis

import pandas as pd
import pickle
import numpy as np
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

from sklearn.cluster import AffinityPropagation
from sklearn import mixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.api as sm
from sklearn import manifold
from sklearn import decomposition
from sklearn.decomposition import NMF, LatentDirichletAllocation
#from pyclustering.cluster.optics import optics
#from pyclustering.utils import draw_clusters


import OpticsClusterArea as OP
import AutomaticClustering as AutoC
from itertools import *

from detect_peak import *
exec(open('kmeans.py').read())



#np.set_printoptions(threshold=np.nan)

def data_conv(data):
    n = len(data)
    k = len(data[0])
    new = np.empty([n,k])
    for i, d in enumerate(data):
        new[i,:] = d[:k]
    return new

feature = pickle.load(open('features_test.pkl', 'rb'))
p_tr = data_conv(feature.dp_tr.values)
q_tr = data_conv(feature.dq_tr.values)
pq_tr = np.hstack([p_tr, q_tr])
p_s = feature.dp_s.reshape(-1,1)
q_s = feature.dq_s.reshape(-1,1)
dp_dq = feature.dp_dq.reshape(-1,1)
dh_s_t = data_conv(feature.dh_s.values)
dh_s = np.vstack([dh_s_t[:,0],dh_s_t[:,2],dh_s_t[:,4]]).transpose()
p_n = feature.p_n.values.reshape(-1,1)
t_s = feature.time_stamp.values
p_diff = np.diff(p_tr)
q_diff = np.diff(q_tr)
pq_diff = np.c_[p_diff,q_diff]
np.place(p_n, p_n==0, -1)


# #manual features
# def normalize(X):
#     X = pd.DataFrame(X)
#     return ((X-X.mean())/(X.max()-X.min())).values
# p_t = pd.DataFrame(p_tr)
# q_t = pd.DataFrame(q_tr)
# assert 0
# delta_p = pd.DataFrame(p_tr[:,1:] - p_tr[:,:-1])
# delta_q = pd.DataFrame(q_tr[:,1:] - q_tr[:,:-1])
# delta_pq = pd.concat((delta_p,delta_q))
# delta_p_max = delta_p.max(axis=1).values.reshape(-1,1)
# delta_p_min = delta_p.min(axis=1).values.reshape(-1,1)
# delta_q_max = delta_q.max(axis=1).values.reshape(-1,1)
# delta_q_min = delta_q.min(axis=1).values.reshape(-1,1)

# delta_p_abs = np.abs(p_tr[:,1:] - p_tr[:,:-1]).sum(axis=1).reshape(-1,1)
# delta_q_abs = np.abs(q_tr[:,1:] - q_tr[:,:-1]).sum(axis=1).reshape(-1,1)

# X_manual = normalize(np.hstack([p_s, delta_p_max,delta_p_min,dh_s]))

# #def log_data(X, base=2):
# #    X[np.abs(X)<=base] = X[np.abs(X)<=base]/base
# #    X[X>base] = np.log(X[X>base])/np.log(base)
# #    X[X<-base] = -np.log(-X[X<-base])/np.log(base)
# #    return X
# #
# #pq_log = log_data(pq_tr)
# #pq_diff_log = log_data(pq_diff)
# #PCA
# def do_pca(data, info_re = 0.99):
#     info = 0
#     k=1
#     while info<=info_re and k<=100:
#         pca = PCA(n_components=k)
#         pca.fit(data)
#         info = pca.explained_variance_ratio_.sum()
#         k += 1
#         print(info)
#     return k-1,pca

# #preprocessing
# k,pca_ = do_pca(pq_tr)
# #plot_red(pca_.components_)
# X_pca = pd.DataFrame(pca_.transform(pq_tr)).values[:5000,:]
# X_train_ = pd.DataFrame(np.hstack([dp_dq,p_tr.max(axis=1).reshape(-1,1),p_tr.min(axis=1).reshape(-1,1),\
# q_tr.max(axis=1).reshape(-1,1),q_tr.min(axis=1).reshape(-1,1),dh_s]))
# X_train_norm = normalize(np.hstack([ _X_train, X_train_]))
# print(k)

# k_grad, pca_grad = do_pca(np.c_[np.diff(p_tr),np.diff(q_tr)])
# X_train_grad = pca_grad.transform(np.c_[np.diff(p_tr),np.diff(q_tr)])
# print(k_grad)

# tsne = manifold.TSNE(n_components=6, init='pca', random_state=0)
# _X_tsne = tsne.fit_transform(pq_tr)
# X_tsne = normalize(np.hstack([ _X_tsne, X_train_]))


# pq_s = np.c_[p_s,q_s]
# dpgmm_p = mixture.DPGMM(n_components=50, covariance_type='spherical', alpha=1e2, n_iter=100)
# dpgmm_p.fit(pq_s)
# print(len(np.unique(dpgmm_p.predict(pq_s))))
# dpgmm_p_labels = dpgmm_p.predict(pq_s)

# def plot_pq(labels, p_s, q_s):
#     colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
#     for l,c in zip(np.unique(labels), colors):
#         plt.scatter(p_s[labels==l], q_s[labels==l],color=c,alpha=0.5)
# plot_pq(dpgmm_p_labels, p_s, q_s)

# def compute_scores(X):
#     pca = PCA()
#     fa = FactorAnalysis()

#     pca_scores, fa_scores = [], []
#     for n in n_components:
#         pca.n_components = n
#         fa.n_components = n
#         pca_scores.append(np.mean(cross_val_score(pca, X)))
#         fa_scores.append(np.mean(cross_val_score(fa, X)))

#     return pca_scores, fa_scores


# def shrunk_cov_score(X):
#     shrinkages = np.logspace(-2, 0, 30)
#     cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
#     return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


# def lw_score(X):
#     return np.mean(cross_val_score(LedoitWolf(), X))

# n_components = np.r_[10:51]
# pca_scores, fa_scores = compute_scores(pq_tr)
# k_pca = n_components[np.argmax(pca_scores)]
# k_fa = n_components[np.argmax(fa_scores)]

# #fa
# fa = decomposition.FactorAnalysis(n_components=5)
# fa.fit(pq_tr)
# plot_red(fa.components_)

# #NMF
# nmf = decomposition.NMF(n_components=10)
# nmf.fit(global_norm(pq_tr))
# plot_red(nmf.components_)

# #ica
# def global_norm(X):
#     return (X-X.min())/(X.max()-X.min())
# ica = decomposition.FastICA(n_components=5)
# ica.fit(global_norm(pq_tr))
# plot_red(ica.components_)
# X_ica = ica.transform(global_norm(pq_tr))*(pq_tr.max()-pq_tr.min())

# #lda
# lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
#                                 learning_method='online', learning_offset=50.,
#                                 random_state=0)
# lda.fit(global_norm(pq_tr))
# lda.score(global_norm(pq_tr))
# plot_red(lda.components_)

# #####################################################################################################################
# #classifier

# #Agglomerative Clustering
# from sklearn.cluster import AgglomerativeClustering
# ac = AgglomerativeClustering(linkage='complete', n_clusters=k_pca)
# ac_labels = ac.fit_predict(_X_train)
# #dpgmm
# dpgmm = mixture.DPGMM(n_components=60, covariance_type='diag', alpha=1e5, n_iter=100)
# dpgmm.fit(X_ica)
# print(len(np.unique(dpgmm.predict(X_ica))))
# dpgmm_labels = dpgmm.predict(X_ica)

# #spectral
# spectral = cluster.SpectralClustering(n_clusters=k_pca, eigen_solver='arpack', affinity="nearest_neighbors")
# s_labels = spectral.fit_predict(X_red)
# #k-mean+gmm
# #ks, Wks, Wkbs, sk = gap_statistic(X_train_norm)
# #plt.plot(Wks)
# #indicator_k = np.where(Wks[:-1]-Wkbs[1:]+sk[1:]>=0)[0]
# #best_k = ks[indicator_k[0]]
# #print(best_k)
# #gmm = mixture.GMM(n_components=best_k, covariance_type='full', n_iter=100)
# #gmm_labels = gmm.predict(X_train_norm)
# #gmm_prob = gmm.predict_proba(X_train_norm)

# #mean shift
# #bandwidth = estimate_bandwidth(X_train_norm, quantile=0.1, n_samples=20)
# #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# #ms.fit(X_train_norm)
# #ms_labels = ms.labels_
# #print(ms_labels.max())
# #ms_cluster_centers = ms.cluster_centers_

# #ap
# af_up = AffinityPropagation(damping=0.9, preference=None).fit(X_red)
# print(af_up.labels_.max())
# afl = af_up.labels_

# #GMM
# def gmm_bic(X):
#     lowest_bic = np.infty
#     bic = []
#     n_components_range = range(1, 20)
#     cv_types = ['spherical', 'tied', 'diag', 'full']
#     for cv_type in cv_types:
#         for n_components in n_components_range:
#             print(n_components)
#             # Fit a mixture of Gaussians with EM
#             gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
#             gmm.fit(X)
#             bic.append(gmm.bic(X))
#             if bic[-1] < lowest_bic:
#                 lowest_bic = bic[-1]
#                 best_gmm = gmm
#     return best_gmm

# #gmm = gmm_bic(pq_s)
# gmm = mixture.GMM(n_components=24, covariance_type='diag')
# gmm.fit(X_ica)
# gmm_labels = gmm.predict(X_ica)
# print(np.unique(gmm_labels))

# plot_pq(gmm_labels, p_s, q_s)
# #dbscan
# def distance(X):
#     euclidean_d = euclidean_distances(X, squared=True)
#     np.fill_diagonal(euclidean_d,euclidean_d.max())
#     return euclidean_d.min(axis=1)

# def find_eps(X):
#     #plt.scatter(distance(X),np.zeros_like(distance(X)))
#     kde = sm.nonparametric.KDEUnivariate(distance(X))
#     kde.fit()
#     ind_k=detect_peaks(kde.density)
#     #print(kde.support.max())
#     return kde.support[ind_k[-1]]
# def dbscan(X):
#     db = DBSCAN(eps=2*find_eps(X), min_samples=2).fit(X)
#     return db
    
# db = dbscan(X_red)

# #optics
# def optics(X, eps=10):
#     RD, CD, order = OP.optics(X,eps)
#     RPlot = RD[order]
#     RPoints = X[order]
#     #hierarchically cluster the data
#     rootNode = AutoC.automaticCluster(RPlot, RPoints)    
#     leaves = AutoC.getLeaves(rootNode, [])
#     labels_ = np.zeros_like(CD, dtype=int)-1
#     for idx, item in enumerate(leaves):
#         labels_[order[item.start:item.end+1]] = idx
# #    outlier_idx=0
# #    print()
# #    for idx, item in enumerate(leaves):
# #        if idx>0:
# #            if item.start-leaves[idx-1].end >= eps:
# #                labels_[order[leaves[idx-1].end+1:item.start]] = len(leaves)+outlier_idx
# #                outlier_idx += 1
#     return labels_
# #def process_optics(X, eps, minpts):
# #    optics_instance = optics(X, eps, minpts)
# #    optics_instance.process()
# #    clusters = optics_instance.get_clusters()
# #    print(len(clusters))
# #    noise = optics_instance.get_noise()
# #    draw_clusters(X, clusters, [], '.')
    
    
# l = optics(X_pca,10)
# print(np.unique(l))

# #debacl
# import debacl as dcl
# def dbcl(X,k=500, min_p=None):
#     if min_p is None:
#         min_p = int(len(X)/200)
#     tree = dcl.construct_tree(X, k)
#     pruned_tree = tree.prune(min_p)
#     cluster_labels = pruned_tree.get_clusters()
#     upper_level_idx = cluster_labels[:, 0]
#     #upper_level_set = X[upper_level_idx, :]
#     dbl = np.ones_like(X[:,0])*-1
#     l_ = cluster_labels[:, 1]
    
#     dbl[upper_level_idx] = l_
#     print(len(np.unique(dbl[dbl!=-1])))
#     return dbl
# dbl = dbcl(X_red,min_p=10)



#####################################################################################################################
#plot the feature and events
def plot_pq(t_s, pq_tr, label):
    for idx, (t,data) in enumerate(zip(t_s,pq_tr)):
        plt.subplot(2, 1, 1)
        #p_tr = data.dp_tr.values
        tt = time.strftime('%Y-%m-%d@%H_%M_%S', time.gmtime(t-6*3600))
        plt.plot(data[:1000], label = 'p', color = 'k', alpha=.8 )
        plt.title('Event happened at time:'+str(tt))
        plt.ylabel('real power')
        plt.legend(['p'])
        #plt.show()
        plt.subplot(2, 1, 2)
        plt.plot(data[1000:], label = 'q', color = 'r', alpha=.8 )
        plt.xlabel('time (1/60s)')
        plt.ylabel('image power')
        plt.legend(['q'])
        plt.savefig('classes/%d_%s.png' %(label, tt))
        plt.close()



def plot_events(clf_labels, pq_tr):
    labels = np.unique(clf_labels)
    for label in labels[labels>-1]:
        pq_byclass = pq_tr[clf_labels==label,:]
        t_byclass = t_s[clf_labels==label]
        plot_pq(t_byclass, pq_byclass, label)

plot_events(l, pq_tr)
    
#check pca plane
def plot_pca_2d(pca_data, label_list, alllabels, d=[0,1]):
    colors = cm.rainbow(np.linspace(0, 1, len(label_list)))
    for l,c in zip(np.unique(alllabels)[label_list], colors):
        p_data = pca_data[alllabels==l,:]
        plt.scatter(p_data[:,d[0]], p_data[:,d[1]], color=c, alpha=0.5,s=2)
    plt.show()
   
label_list=np.r_[10:15]
plot_pca_2d(X_pca, label_list, hdbl, d=[0,1])

label_list=np.r_[0:af_up.labels_.max()]
plot_pca_2d(X_train_norm, label_list, af_up.labels_, d=[0,1])
#pd.Series(af_up.labels_, index=range(len(af_up.labels_))).to_csv('label.csv')

#check original plane
def plot_pq_original(p_t, q_t, label_list):
    for idx in np.unique(label_list):
        plt.subplot(2, 1, 1)
        plt.plot(p_t.iloc[label_list==idx, :].transpose(),alpha=0.5,color='k',label = 'p')#.plot(label='real power', legend=False,alpha=0.3,title=str(idx)+'p')
        #plt.legend(['p'])
        plt.title('Event number:'+str(idx))
        plt.ylabel('real power')
        plt.subplot(2, 1, 2)
        plt.plot(q_t.iloc[label_list==idx, :].transpose(),alpha=0.5,color='r',label = 'q')#.plot(label='image power', legend=False,alpha=0.3,title=str(idx)+'q')
        plt.xlabel('time (1/60s)')
        plt.ylabel('reactive power')
        #plt.legend(['q'])
        plt.savefig('result_fig/%d_%d.png' %(idx,len(p_t.iloc[label_list==idx, :])))
        plt.show()
        plt.close()
        
plot_pq_original(p_t,q_t,hdbl)
    
#check components
def plot_red(X_red):
    for X_idx in X_red:
        pd.DataFrame(X_idx).plot(legend=False,alpha=0.3,title='component')
plot_red(lda.components_)
# <codecell> deep learning
from keras.layers import containers, AutoEncoder, Dense, Dropout
from keras import models
import keras
from keras import backend as K



def normalize(X):
    #X=pd.DataFrame(X)
    return (X-X.mean())/(X.max()-X.min())
#cnn_ae

#ae
X_train_ = np.c_[normalize(p_tr),normalize(q_tr)]

start = time.time()
encoder = containers.Sequential([Dense(500, input_dim=X_train_.shape[1], 
                                       activity_regularizer = keras.regularizers.ActivityRegularizer(l1=10)),
                                       Dense(16, activity_regularizer = keras.regularizers.ActivityRegularizer(l1=10))])
decoder = containers.Sequential([Dense(500, input_dim=16), Dense(X_train_.shape[1])])

autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False)
model = models.Sequential()
model.add(autoencoder)

model.compile(optimizer='sgd', loss='mse')
X_red = model.predict(X_train_, verbose=1)
print(time.time()-start)
#model.fit(X_train_, X_train_, nb_epoch=100)

#get_l3 = K.function([model.layers[0].encoder.input], [model.layers[0].encoder.get_output(train=True)])
#X_red = get_l3([X_train_])[0]
#X_red = X_red / np.abs(X_red).min()

#hdbscan
import hdbscan
start = time.time()
hdbl = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X_red)
print(time.time()-start)
print(np.unique(hdbl))