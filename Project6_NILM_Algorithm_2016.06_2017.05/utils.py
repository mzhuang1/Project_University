"""User Profile

import datetime
import pandas as pd
import scipy as sp
import numpy as np
import pywt
import pickle
from statsmodels.robust import mad
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from detect_peak import *
import statsmodels.api as sm
from pandas.compat import range, lrange, lmap, map, zip, string_types

from cvxopt.solvers import *
from l1tf.pandas_wrapper import *
from statistic import *

from keras.layers import Dense, Dropout, Input
from keras.models import Model
import keras
from keras import backend as K

import hdbscan

from socket import *

def wt_denoising(power): 

    noisy_coefs = pywt.wavedec(power, 'db8', mode='constant')
    sigma = mad(noisy_coefs[-1])
    uthresh = sigma*np.sqrt(2*np.log(len(power)))
    denoised = noisy_coefs[:]
    denoised[1:] = (pywt.threshold(i, value=uthresh, mode = 'soft') for i in denoised[1:])
    return pywt.waverec(denoised, 'db8', mode='constant')

def kde(real_power):

    kde = sm.nonparametric.KDEUnivariate(real_power)
    kde.fit()
    return detect_peaks(kde.density, mpd = 8*len(kde.support)/(kde.support[-1]-kde.support[0]), mph = 1/10/len(real_power))


def acf(denoised_power, **kwds):

    n = len(denoised_power)
    data = np.asarray(denoised_power)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)
    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    
    y = lmap(r, x)
    z99 = 2.5758293035489004
    return y, z99 / np.sqrt(n)


def acf_peak_index(denoised_power):

	power_var = pd.Series(denoised_power).rolling(base_window).var().dropna(how='any')
	power_acf, z99=acf(denoised_power[power_var.idxmin()-base_window+1:power_var.idxmin()])
	return detect_peaks(power_acf, mpd=min_event_interval, mph=z99)


def lambda0(peak_idx):

    if peak_idx.size==0:
        return 1
    di = peak_idx - np.r_[0, peak_idx[:-1]]
    mu = di.mean()
    if mu<=30 and peak_idx.size>5:
        return 50
    else:
        return 1


def find_trend(power, denoised_power):

	power = pd.Series(power).values

	power_trend = l1tf(power,lambda0(acf_peak_index(denoised_power))*max(0.05,(200/(power.max()-power.min()))**\
            max(1, min(5,np.log(power.max()-power.min())-4))))

	return power_trend, power-power_trend

def split_groups(outlier_idx, stepsize=6):

    return np.split(outlier_idx, np.where(np.diff(outlier_idx) > stepsize)[0]+1)

def clean_labels(real_power_trend, labels):

    outlier_idx = np.where(labels==-1)[0]
    if outlier_idx.size==0:
        return labels
    outlier_group = split_groups(outlier_idx)
    for loc in outlier_group:
        labels[loc[0]:loc[-1]+1] = -1
    for ix, loc in enumerate(outlier_group):
        #ignore the start and end signal
        if loc[0]==0 or loc[-1]==len(labels)-1:
            continue
        else:
            if (labels[loc[0]-1] == labels[loc[-1]+1] or np.abs(real_power_trend[loc[0]-1]-real_power_trend[loc[-1]+1])<5) \
            and real_power_trend[np.r_[loc[0]-1:loc[-1]+1]].max()-real_power_trend[np.r_[loc[0]-1:loc[-1]+1]].min()<=20:
                labels[loc] = labels[loc[-1]+1]
    
        if ix+1<len(outlier_group):
            len_normal = 2*(outlier_group[ix+1][0] - loc[-1] + 1)
            if (len_normal<len(loc) or len_normal<len(outlier_group[ix+1])) and len_normal<=60:
                #print(np.r_[loc[-1]+1:outlier_group[ix+1][0]])
                labels[np.r_[loc[-1]+1:outlier_group[ix+1][0]]] = -1
    return labels

def iter_clean_labels(trend_real_power):

    labels = DBSCAN(eps=1, min_samples=10).fit_predict(trend_real_power.reshape(-1,1))
    while not np.all(labels == clean_labels(trend_real_power, labels)):
        labels = clean_labels(trend_real_power, labels)
    return labels

def find_event_loc(trend_real_power, labels, acf_peak):

    event_loc = np.where((np.abs(labels[1:]-labels[:-1])>0)&(np.logical_or(labels[1:]==-1, labels[:-1]==-1)))[0]

    if acf_peak.size>0:
        if acf_peak[0]<=30:
            event_loc = event_loc[(event_loc>=acf_peak[0]) & (event_loc<=len(trend_real_power)-acf_peak[0])]
        else:
            event_loc = event_loc[(event_loc>=5) & (event_loc<=len(trend_real_power)-5)]
    else:
        event_loc = event_loc[(event_loc>=5) & (event_loc<=len(trend_real_power)-5)]
    return event_loc

def extend_event_loc(trend_real_power, event_loc, labels):

    trend_real_power_grad = np.diff(trend_real_power)
    event_start = event_loc[0]
    #print('define start point first time', event_start)
    while event_start - steady_before_window >=0:
        event_start = event_start - steady_before_window
        if np.abs(trend_real_power_grad[event_start:event_start+steady_before_window].sum())<1:
            break
    #print('define start point second time', event_start)

    for idx, loc in enumerate(event_loc):
        if idx+1 == len(event_loc):
            event_end = loc
            while 1:
                if event_end + 2*steady_after_window >= len(trend_real_power):
                    #print('data is not enough to extract feature')
                    return []
                else:
                    event_end = event_end + 2*steady_after_window
                    if np.abs(trend_real_power_grad[event_end-2*steady_after_window:event_end].sum())<1:
                        break
            if len(trend_real_power)-event_end <= 30:
                #print('data is not enough to extract feature')
                return []
        else:
            if np.all(labels[loc+1:event_loc[idx+1]] == -1):
                continue
            #flat_pct = flat_idx[(flat_idx>=loc) & (flat_idx<=event_loc[idx+1])].size/(event_loc[idx+1]-loc)
            #change = np.abs(p_trend[event_loc[idx+1]]-p_trend[loc])
            else:
                event_end = int((loc+event_loc[idx+1])/2)
                #print('steady state found')
                break
    if event_start <= steady_before_window:
        event_start = 0
    else:
        event_start = int(event_start - steady_before_window/2)
    #print('the event start and end point', event_start, event_end)
    return np.r_[event_start:event_end]

def extract_power(trend_real_power, trend_reactive_power, event_loc, align=True):

    if event_loc[0] == 0:
        start_adjust = 1
    else:
        start_adjust = int(steady_before_window/2)
    #end_adjust = int(steady_after_window/2)

    trend_real_power = np.array(trend_real_power)
    trend_reactive_power = np.array(trend_reactive_power)

    feature_trend_real_power = trend_real_power[event_loc]-trend_real_power[event_loc[0]:event_loc[0]+start_adjust].mean()
    feature_trend_reactive_power = trend_reactive_power[event_loc]-trend_reactive_power[event_loc[0]:event_loc[0]+start_adjust].mean()

    if np.isnan(feature_trend_real_power).any():
        print('there is something error:', event_loc)
        assert 0
    
    if align:
        if len(trend_real_power) < fixed_window:
            max_grad_loc = np.argmax(np.abs(np.diff(feature_trend_real_power)))
            num_before = np.abs(fixed_window/4 - max_grad_loc)
            num_after = np.abs(fixed_window*3/4 - (len(feature_trend_real_power) - max_grad_loc))

            feature_trend_real_power = np.concatenate((np.zeros(num_before), feature_trend_real_power,
                                                       np.ones(num_after)*feature_trend_real_power[-1]))
            feature_trend_reactive_power = np.concatenate((np.zeros(num_before), feature_trend_reactive_power, 
                                                           np.ones(num_after)*feature_trend_reactive_power[-1]))

    return feature_trend_real_power, feature_trend_reactive_power

def normalize(feature):

    return (feature-feature.values.mean())/(feature.values.max()-feature.values.min())


def dimension_reduce(feature_real_power, feature_reactive_power):
    

    normalization_coef = np.array([feature_real_power.values.mean(), feature_real_power.values.max(), feature_real_power.values.min(), 
                                   feature_reactive_power.values.mean(), feature_reactive_power.values.max(), feature_reactive_power.values.min()])

    feature_train = np.r_[normalize(feature_real_power), normalize(feature_reactive_power)]
 
 
    input_signal = Input(shape=(feature_train.shape[1],))
    encoded = Dense(encoder_layer_1_num, activity_regularizer = keras.regularizers.ActivityRegularizer(l1=10))(input_signal)
    encoded = Dense(encoder_layer_2_num, activity_regularizer = keras.regularizers.ActivityRegularizer(l1=10))(encoded)
    
    decoded = Dense(encoder_layer_1_num)(encoded)
    decoded = Dense(feature_train.shape[1])(decoded)
    
    autoencoder = Model(input = input_signal, output=decoded)
    
    reduce_dimension_model = Model(input=input_signal, output=encoded)
    
    autoencoder.compile(optimizer='adadelta', loss='mse')
    
    autoencoder.fit(feature_train, feature_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True)
    
    feature_reduced = reduce_dimension_model.predict(feature_train, verbose=1)

    return reduce_dimension_model, feature_reduced, normalization_coef

def h_clustering(feature_reduced):

    clustering_model = hdbscan.HDBSCAN(min_cluster_size=15).fit(feature_reduced)

    return clustering_model, clustering_model.labels_

def self_learning(feature, sample_labels):

    return deep_learning_model, feature_reduced


def normalize_online(feature_real_power, feature_reactive_power, normalization_coef):

    feature_real_power_reduced = (feature_real_power - normalization_coef[0])/(normalization_coef[1] - normalization_coef[2])
    feature_reactive_power_reduced = (feature_real_power - normalization_coef[3])/(normalization_coef[4] - normalization_coef[5])
    return np.r_[feature_real_power_reduced,feature_reactive_power_reduced]

def online_receiving_save2mongodb(socket_data):
    voltage = socket_data[:voltage_index]
    current = socket_data[voltage_index:current_index]
    real_power = socket_data[current_index:real_power_index]
    reactive_power = socket_data[real_power_index:reactive_power_index]
    time_stamp = socket_data[-2]
    device_id = int(socket_data[-1])
    
    db.raw_data.insert_one(
    {
        "device_id": device_id,
        "voltage":pickle.dumps(voltage),
        "current":pickle.dumps(current),
        "real_power":pickle.dumps(real_power),
        "reactive_power":pickle.dumps(reactive_power),
        "time_stamp":float(time_stamp)

    }
    )
    
    db.power_history.insert_one(
    {
        "device_id": device_id,
        "real_power": real_power[-1],
        "energy": np.mean(real_power),
        "time_stamp": float(time_stamp)

    }
    )
    
    db.appliance_state.update_one(
    {"device_id":device_id},
    {
        "$set":{
        "real_power_real_time":float(real_power[-1]),
        },
    },
    upsert=True
    )
    
    return real_power, reactive_power, int(device_id), time_stamp
    