# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:31:54 2015


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import svm
import brewer2mpl
from pandas.tools.plotting import parallel_coordinates
from matplotlib import rcParams
from matplotlib.pyplot import savefig
from sklearn.lda import LDA
from sklearn.qda import QDA

dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

#import scipy.stats as stats

#load excel data
xls = 'trainingData.xlsx'
#xls_test = 'Heat Scan data_IIT_30415.xlsx'
names = ['models','locations',
         'tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6',
         'load_1','load_2','load_3','load_4','load_5','load_6',
         'months','hours','label']
tprtlist = ['tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6']
loadlist = ['load_1','load_2','load_3','load_4','load_5','load_6']
data_0=pd.io.excel.read_excel(xls,sheetname='Sheet1',
                            names=names).dropna()
print 'data_0', data_0

#data cleaning
model = set(data_0.models)
model = sorted(model)
for m in model:
    data_0[m] = [m in meter for meter in data_0.models]
data_0[model].sum()
print 'data_0[model]', data_0[model]


location = set(data_0.locations)
location = sorted(location)
for m in location:
    data_0[m] = [m in meter for meter in data_0.locations]
data_0[location].sum()
print data_0[location]
month = set(data_0.months)
month = sorted(month)
for m in month:
    data_0[m] = [m in meter for meter in data_0.months]
data_0[month].sum()
print data_0[month]
feature_index = tprtlist 
X = data_0[feature_index].values
y = data_0['label'].values

list_0 = ['tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6','label']
list_1 = ['load_1','load_2','load_3','load_4','load_5','load_6','label']
mkeys=[1,2,3]
mvals=model
rmap={e[0]:e[1] for e in zip(mkeys,mvals)}
#d2=data_0.groupby(['models','locations'])
d2=data_0.groupby('May')
#fig, axes=plt.subplots(figsize=(20,20), nrows=len(location)*len(model), ncols=1)
colors=[dark2_cmap.mpl_colormap(col) for col in np.linspace(1,0,2)]
#
#for m,subset in d2:
#    a='###'
#    print(m)
#    parallel_coordinates(subset[list_0],'label',colors=colors, alpha=0.12)
#    plt.show()
#    parallel_coordinates(subset[list_1],'label',colors=colors, alpha=0.12)
#    plt.show()
    
  
#analyz data
#smaller_frame=data_0[loadlist]#tprtlist+loadlist
#axeslist=scatter_matrix(smaller_frame, alpha=0.8, figsize=(12,12), diagonal="kde")
#for ax in axeslist.flatten():
#    ax.grid(False)
    
#PCA
pca = PCA(n_components=5)
X_E = pca.fit_transform(X)
#print(pca.explained_variance_ratio_)
#plt.scatter(X_E[:, 0], X_E[:, 1])
def do_pca(d,n):
    pca = PCA(n_components=n)
    X = pca.fit_transform(d)
    print(pca.explained_variance_ratio_)
    return X, pca
X2, pca2=do_pca(X,2)
df = pd.DataFrame({"x":X2[:,0], "y":X2[:,1],"label":np.where(y==1,
                   "bad","good")})
colors = ["red", "yellow"]
#for label, color in zip(df['label'].unique(), colors):
#    mask = df['label']==label
#    plt.scatter(df[mask]['x'], df[mask]['y'],c=color,label=label,alpha=0.6)
#plt.legend()


##kernel PCA
#def do_kpca(d,n):
#    kpca = KernelPCA(n_components=2,kernel='rbf',fit_inverse_transform=True,gamma=0.1)
#    X_kpca=kpca.fit_transform(d)
#    return X_kpca,kpca
#X2,kpca2=do_kpca(X,2)
#df = pd.DataFrame({"x":X2[:,0], "y":X2[:,1],"label":np.where(y==1,
#                   "bad","good")})   
#colors = ["red", "blue"]
#for label, color in zip(df['label'].unique(), colors):
#    mask = df['label']==label
#    plt.scatter(df[mask]['x'], df[mask]['y'],c=color,label=label,alpha=0.8)
#plt.legend()

#LDA vs QDA





X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_E,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

gbdt = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1,
                                  max_depth=5, random_state=1234).fit(X_train,y_train)
gbdt.score(X_test, y_test)
pd.crosstab(y_test,gbdt.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
indices = np.argsort(gbdt.feature_importances_)
plt.barh(np.arange(len(feature_index)),gbdt.feature_importances_[indices])
plt.yticks(np.arange(len(feature_index))+0.5, np.array(feature_index)[indices])
_=plt.xlabel('Relative importance')
np.array(feature_index)[indices][::-1]

##logistic regressian
#def fit_logistic(X_train, y_train, reg=0.0001, penalty='l2'):
#    clf = LogisticRegression(C=reg, penalty=penalty)
#    clf.fit(X_train, y_train)
#    return clf
#def cv_optimize(X_train, y_train, paramslist, penalty='l2', n_folds=10):
#    clf = LogisticRegression(penalty=penalty)
#    parameters = {"C":paramslist}
#    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
#    gs.fit(X_train, y_train)
#    return gs.best_params_, gs.best_score_
#def cv_and_fit(X_train, y_train, paramslist, penalty='l2', n_folds=10):
#    bp, bs = cv_optimize(X_train, y_train, paramslist, penalty=penalty, n_folds=n_folds)
#    print("BP,BS",bp,bs)
#    clf = fit_logistic(X_train, y_train, penalty=penalty, reg=bp['C'])
#    return clf
#
#clf=cv_and_fit(X_train_pca,y_train_pca,np.logspace(-4,3, num=100))
#pd.crosstab(y_test_pca,clf.predict(X_test_pca), rownames=['Actual'], colnames=['Predicted'])

#adaboost decision tree
#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200)#algorithm="SAMME",
#bdt.fit(X_train,y_train)
#pd.crosstab(y_test,bdt.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
#indices = np.argsort(bdt.feature_importances_)
#plt.barh(np.arange(len(feature_index)),bdt.feature_importances_[indices])
#plt.yticks(np.arange(len(feature_index))+0.5, np.array(feature_index)[indices])
#_=plt.xlabel('Relative importance')
#np.array(feature_index)[indices][::-1]
#GBDT


#svm
#def fit_svc(X_train, y_train, gamma=1e-3,C=1e8):
#    svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X_train, y_train)
#    return svc
#def cv_optimize_svc(X_train, y_train, gammalist, Clist, n_folds=10):
#    clf = svm.SVC()
#    parameters = {"C":Clist,"gamma":gammalist,"kernel":['linear','rbf']}
#    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds).fit(X_train, y_train)
#    return gs.best_params_, gs.best_score_
#def cv_and_fit_svc(X_train, y_train, gamma, C, n_folds=10):
#    bp, bs = cv_optimize_svc(X_train, y_train, gamma, C, n_folds=n_folds)
#    #print("BP,BS",bp,bs)
#    clf = fit_svc(X_train, y_train, gamma=bp['gamma'], C=bp['C'],kernel=bp['kernel'])
#    return clf
##clf_svc=cv_and_fit_svc(X_train,y_train,np.logspace(-4,3, num=2),np.logspace(-4,3, num=2))
#clf_svc=fit_svc(X_train,y_train)
##pd.crosstab(y_test_pca,clf_svc.predict(X_test_pca), rownames=['Actual'], colnames=['Predicted'])
#pd.crosstab(y_test,clf_svc.predict(X_test), rownames=['Actual'], colnames=['Predicted'])


#X_train, X_test, y_train, y_test = train_test_split(X, y)
#clf1 = LinearRegression()
#clf1.fit(X_train, y_train)
#predicted_train = clf1.predict(X_train)
#predicted_test = clf1.predict(X_test)
#trains = X_train.reshape(1,-1).flatten()
#tests = X_test.reshape(1,-1).flatten()
#print(clf1.coef_, clf1.intercept_)
#X_reconstructed = pca.inverse_transform(X_E)
#plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], c='b', s=35, alpha=0.7)

#data=pd.io.excel.read_excel(xls,sheetname='Sheet1',
#                            parse_cols=[2,3,4,9,16,17,18,19,20,21,22,23,24,25,26,27,66],names=names).dropna()

##clean data
##analyz data
#data.set_index("meters", inplace=1)
#smaller_frame=data[tprtlist]
#axeslist=scatter_matrix(smaller_frame, alpha=0.8, figsize=(12,12), diagonal="kde")
#for ax in axeslist.flatten():
#    ax.grid(False)
#    
#smaller_frame.corr()
#t1_2 = smaller_frame[['tprt_1','tprt_2']].values
#t1_2n = (t1_2-t1_2.mean(axis=0))/t1_2.std(axis=0)
#t1_std_vec = t1_2n[:,0]
#t1_std = t1_std_vec.reshape(-1,1)
#t2_std_vec = t1_2n[:,1]
#t2_std = t2_std_vec.reshape(-1,1)
#
#X_train, X_test, y_train, y_test = train_test_split(t1_std, t2_std_vec)
#clf1 = LinearRegression()
#clf1.fit(X_train, y_train)
#predicted_train = clf1.predict(X_train)
#predicted_test = clf1.predict(X_test)
#trains = X_train.reshape(1,-1).flatten()
#tests = X_test.reshape(1,-1).flatten()
#print(clf1.coef_, clf1.intercept_)
#
#plt.scatter(t1_std_vec, t2_std_vec, c='r')
#plt.plot(trains, predicted_train, c='g', alpha=0.5)
#plt.plot(tests, predicted_test, c='g', alpha=0.3)
#
#plt.scatter(predicted_test, predicted_test-y_test, c='g', s=40)
#
#clf1.score(X_train, y_train), clf1.score(X_test, y_test)
#
#pca = PCA(n_components=1)
#X_E = pca.fit_transform(t1_2)
#X_reconstructed = pca.inverse_transform(X_E)
#plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], c='b', s=35, alpha=0.7)
#
#
#
#
#databymeters=data.groupby('meters')
#pd.crosstab(data.locations, data.models)
#
#model = set(data.models)
#model = sorted(model)
#for m in model:
#    data[m] = [m in meter for meter in data.models]
#data[model].sum()
#
#location = set(data.locations)
#location = sorted(location)
#for l in location:
#    data[l] = [l in meter for meter in data.locations]
#data[location].sum()
#
##temperature
##fig, axes = plt.subplots(figsize=(10,20), nrows=6, ncols=1)
##i=0
##modelkeys=[1,2,3,4]
##model_dict={e[0]:e[1] for e in zip(modelkeys,model)}
##for ax in axes.flatten():
##    tprt=tprtlist[i]
##    tprt_value=data[tprt]
##    minmax=[tprt_value.min(), tprt_value.max()]
##    counts=[]
##    nbins=100
##    histbinslist = np.linspace(minmax[0], minmax[1], nbins)
##    counts=-np.diff([tprt_value[tprt_value>x].count() for x in histbinslist]).min()
##    for k,g in data.groupby('models'):
##        #style = {'histtype':'step', 'alpha':1.0, 'bins':histbinslist, 'label':model_dict[k]}
##        ax.hist(g[tprt])
##        ax.set_xlim(minmax)
##        ax.set_title(tprt)
##        ax.grid(False)
##    ax.set_ylim([0,counts])
##    ax.legend()
##    i=i+1
##fig.tight_layout()
#
##load vs temperature
#
#
##form training data
#x = data.iloc[:,4:]