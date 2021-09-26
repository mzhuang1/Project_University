# -*- coding: utf-8 -*-


import pandas as pd
import scipy as sp
import numpy as np
import pywt
from statsmodels.robust import mad
import numpy.ma as ma
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
import numpy.linalg.linalg as LA
from scipy import sparse
import math
import segment
import fit
from detect_peak import *
import statsmodels.api as sm
from pandas.compat import range, lrange, lmap, map, zip, string_types

from cvxopt.solvers import *
from l1tf.pandas_wrapper import *

def wt_denoising(data):  
    noisy_coefs = pywt.wavedec(data, 'db1', level=3, mode='constant')
    sigma = mad(noisy_coefs[-1])
    uthresh = sigma*np.sqrt(2*np.log(len(data)))
    denoised = noisy_coefs[:]
    denoised[1:] = (pywt.threshold(i, value=uthresh, mode = 'soft') for i in denoised[1:])
    signal = pywt.waverec(denoised, 'db1', mode='constant')
    return signal

def lambda0(peak):
    if peak.size==0:
        return 1
    di = peak - np.r_[0, peak[:-1]]
    mu = di.mean()
    if mu<=30 and peak.size>5:
        return 50
    else:
        return 1

def acf(series, **kwds):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)
    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    z99 = 2.5758293035489004
    return y, z99 / np.sqrt(n)

# def l1trend(data, acf_idx):
#     lambda1 = max(1, min(5,np.log(p[idx].max()-p[idx].min())-5))
#     l1trend = l1tf(p[idx],lambda0(ind)*max(0.05,(200/(p[idx].max()-p[idx].min()))**lambda1))
#     if acf_idx.size>0:
#         if acf_idx[0]<=30 and acf_idx.size>10:
#             coef=8
#         else:
#             coef=.1
#         if data.max()-data.min()<=120:
#             coef=8
#     else:
#         if data.max()-data.min()<=120:
#             coef=8
#         else:
#             coef = .1
#     return l1tf(data, coef)

def consecutive(data, stepsize=6):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def clean_label(data, label):
    outlier_idx = np.where(label==-1)[0]
    if outlier_idx.size==0:
        return label
    outlier_group = consecutive(outlier_idx)
    for loc in outlier_group:
        label[loc[0]:loc[-1]+1] = -1
    for ix, loc in enumerate(outlier_group):
        #ignore the start and end signal
        if loc[0]==0 or loc[-1]==len(label)-1:
            continue
        else:
            if (label[loc[0]-1] == label[loc[-1]+1] or np.abs(data[loc[0]-1]-data[loc[-1]+1])<5) \
            and data[np.r_[loc[0]-1:loc[-1]+1]].max()-data[np.r_[loc[0]-1:loc[-1]+1]].min()<=20:
                label[loc] = label[loc[-1]+1]

        if ix+1<len(outlier_group):
            len_normal = 2*(outlier_group[ix+1][0] - loc[-1] + 1)
            if (len_normal<len(loc) or len_normal<len(outlier_group[ix+1])) and len_normal<=60:
                #print(np.r_[loc[-1]+1:outlier_group[ix+1][0]])
                label[np.r_[loc[-1]+1:outlier_group[ix+1][0]]] = -1
    return label

outputfre=60
mean_window_bf = 10 #the size of mean rolling mean before events
mean_window_af = 10 #the size of mean rolling mean after events
data_window_size = int(outputfre*5) #if length of data is longer than this, start to do detection
fix_window_size = 1000
min_hold_loc_interval = 10
delta_p = 9 #threshold of delta p

def ev_detect(data,databuffer,q1):
    #print(len(data))
    index1=data[6]-np.linspace(63/60,0,64)
    data=pd.DataFrame({'u' : pd.Series(data[0],index=index1),
                         'i' : pd.Series(data[1],index=index1),
                         'p' : pd.Series(data[2],index=index1),
                         'q' : pd.Series(data[3],index=index1),
                         'harmony' : pd.Series(data[4],index=data[6]-np.linspace(1,12/60,4))})
    databuffer=pd.concat([databuffer,data]) 
    #denoising power signal
    #p_value = wt_denoising(databuffer['p'].values) #good for kde detection, segmentation, event detection
    #q_value = wt_denoising(databuffer['q'].values) #good for kde detection, segmentation
    while len(databuffer['p']) >= data_window_size:
        #print(len(p_value))
        #denoising power signal
        p_value = wt_denoising(databuffer['p'].values) #good for kde detection, segmentation, event detection
        q_value = wt_denoising(databuffer['q'].values) #good for kde detection, segmentation
        
        #kde detect the event
        kde = sm.nonparametric.KDEUnivariate(p_value)
        kde.fit()
        ind_kde=detect_peaks(kde.density, mpd = 8*len(kde.support)/(kde.support[-1]-kde.support[0]), mph = 1/10/len(p_value))
        
        #if only one peak in kde then there is no event in time series data, return dataframe
        if ind_kde.size<2:
            return databuffer.drop(databuffer.index[:data_window_size*0.5])

        #caclulate the autocorrelation coeff
        var_p = pd.rolling_var(pd.Series(p_value),data_window_size).dropna(how='any')
        p_value_acf, z99=acf(p_value[var_p.idxmin()-data_window_size+1:var_p.idxmin()])
        ind_acf = detect_peaks(p_value_acf, mpd=min_hold_loc_interval, mph=z99)

        var_q = pd.rolling_var(pd.Series(q_value),data_window_size).dropna(how='any')
        q_value_acf, z99_q=acf(q_value[var_q.idxmin()-data_window_size+1:var_q.idxmin()])
        ind_acf_q = detect_peaks(q_value_acf, mpd=min_hold_loc_interval, mph=z99_q)


        p_trend = l1tf(databuffer['p'].values,lambda0(ind_acf)*max(0.05,(200/(databuffer['p'].values.max()-databuffer['p'].values.min()))**\
            max(1, min(5,np.log(databuffer['p'].values.max()-databuffer['p'].values.min())-4))))
        q_trend = l1tf(databuffer['q'].values,lambda0(ind_acf_q)*max(0.05,(200/(databuffer['q'].values.max()-databuffer['q'].values.min()))**\
            max(1, min(5,np.log(databuffer['q'].values.max()-databuffer['q'].values.min())-4))))

        label = clean_label(p_trend, DBSCAN(eps=1, min_samples=10).fit_predict(p_trend.reshape(-1,1)))
        while not np.all(label == clean_label(p_trend, label)):
            label = clean_label(p_trend, label)

        if np.any(label[-2*mean_window_af:] == -1):
            return databuffer

        if np.unique(label[label!=-1]).size <= 1:
            return databuffer.drop(databuffer.index[:data_window_size*0.5])

        
        event_loc = np.where((np.abs(label[1:]-label[:-1])>0)&(np.logical_or(label[1:]==-1, label[:-1]==-1)))[0]

        

        if ind_acf.size>0:
            if ind_acf[0]<=30:
                event_loc = event_loc[(event_loc>=ind_acf[0]) & (event_loc<=len(p_trend)-ind_acf[0])]
            else:
                event_loc = event_loc[(event_loc>=5) & (event_loc<=len(p_trend)-5)]
        else:
            event_loc = event_loc[(event_loc>=5) & (event_loc<=len(p_trend)-5)]
        debug_list = []

        # print('label',label)
        # print('event_loc',event_loc)
        # plt.plot(p_trend)
        # plt.plot(databuffer['p'].values, alpha=0.3)
        # plt.show()

        if event_loc.size==0:

            # print('quit','+++++++++++++++++++++++++++++++++++++++++++')
            # plt.plot(p_trend)
            # plt.plot(databuffer['p'].values, alpha=0.3)
            # plt.show()
            
            return databuffer.drop(databuffer.index[:data_window_size*0.5])

        p_trend_grad = p_trend[1:] - p_trend[:-1]
        p_value_grad = p_value[1:] - p_value[:-1]
        #locate start and end points
        event_start = event_loc[0]
        while event_start - mean_window_bf >=0:
                event_start = event_start - mean_window_bf
                if np.abs(p_trend_grad[event_start:event_start+mean_window_bf].sum())<1:
                    break
        
        # flat_idx = np.where(np.abs(p_trend_grad)<=0.3)[0]

        for idx, loc in enumerate(event_loc):
            if idx+1 == len(event_loc):
                event_end = loc
                while 1:
                    if event_end + 2*mean_window_af >= len(p_value):
                        print('data is not enough to extract feature')
                        return databuffer
                    else:
                        event_end = event_end + 2*mean_window_af
                        if np.abs(p_trend_grad[event_end-2*mean_window_af:event_end].sum())<1:
                            break
                if len(p_trend)-event_end <= 30:
                    return databuffer
            else:
                if np.all(label[loc+1:event_loc[idx+1]] == -1):
                    continue
                #flat_pct = flat_idx[(flat_idx>=loc) & (flat_idx<=event_loc[idx+1])].size/(event_loc[idx+1]-loc)
                #change = np.abs(p_trend[event_loc[idx+1]]-p_trend[loc])
                else:
                    event_end = int((loc+event_loc[idx+1])/2)
                    print('stable status found')
                    break

        ################################################################################## 
        #define new start and end points
        if event_start <= mean_window_bf:
            event_start_mean = event_start + 1
            event_start = 0
        else:
            event_start = int(event_start - mean_window_bf/2)
            event_start_mean = int(event_start + mean_window_bf/2)
        event_end_mean = int(event_end - mean_window_af/2)

        print('start and end points:', event_start, event_end)
        print(event_loc, 'event_loc')
        #print(label)
        # plt.plot(p_value, alpha=0.6)
        # plt.plot(p_trend)
        # plt.plot(databuffer['p'].values, alpha=0.3)
        # plt.show()

        
        #plt.plot(p_value_est)
        debug_list.append([event_start, event_start_mean, event_end_mean, event_end])
        ##################################################################################
        # if ind_acf.size>0:
        #     if ind_acf[0]<=30 or ind_acf.size>20:
        #         p_signal = p_trend
        #         q_signal = q_trend
        #     else:
        #         p_signal = databuffer['p'].values
        #         q_signal = databuffer['q'].values
        # else:
        p_signal = p_trend#databuffer['p'].values
        q_signal = q_trend#databuffer['q'].values

        #stable status
        dp_s=p_trend[event_end_mean:event_end].mean()\
        -p_trend[event_start:event_start_mean].mean()# 9.dp_s: stable delta p
        dq_s=q_trend[event_end_mean:event_end].mean()\
        -q_trend[event_start:event_start_mean].mean()# 10.dq_s: stable delta q
        dp_dq=(5+np.abs(dp_s))/(5+np.abs(dq_s))# 13.dp_dq: delta p over delta q

        #transient status
        dp_tr = p_signal[event_start:event_end]\
         - p_trend[event_start:event_start_mean].mean()
        dq_tr = q_signal[event_start:event_end]\
         - q_trend[event_start:event_start_mean].mean()
        #print(dp_tr)
        if np.isnan(dp_tr).any():
            print('there is something error:', event_start, event_start_mean, event_end_mean, event_end, event_loc,p_mean_grad)
            print('dp_tr is:', databuffer['p'].iloc[event_start:event_end].values,'\n',databuffer['p'].iloc[event_start:event_start_mean].mean())
            assert 0
        #harmonic
        if event_end - event_start <= 48:
            delta_t_h = int((48-(event_end-event_start))/2+2)
        else:
            delta_t_h = 0
        
        #print('delta_t_h is:', delta_t_h)
        
        event_end_h = event_end + delta_t_h
        event_start_h = max(0,event_start - delta_t_h)
        dh = databuffer['harmony'].iloc[event_start_h:event_end_h].dropna(how='any').values
        #print(event_start, event_end, dh)
        dh_s = dh[-1] - dh[0]
        # #print(np.array([data_h for data_h in dh]))
        # dh = np.array([data_h for data_h in dh]) - dh[0]

        # #print('stable harmonic is', dh_s)

        #time stamp of event
        time_stamp=databuffer.index[event_start-1]

        ##################################################################################
        #fix windows size of transient status
        #mean_make_window_size = 6

        if len(dp_tr) < fix_window_size:
            #print(event_start, event_end,np.abs(dp_tr[1:] - dp_tr[:-1]))
            max_loc_p_tr_grad = np.argmax(np.abs(dp_tr[1:] - dp_tr[:-1])) 
            p_tr_mean_bf = 0
            q_tr_mean_bf = 0

            p_tr_mean_af = p_trend[event_end_mean:event_end].mean() \
            - p_trend[event_start:event_start_mean].mean()
            q_tr_mean_af = q_trend[event_end_mean:event_end].mean() \
            - q_trend[event_start:event_start_mean].mean()


            num_bf = np.abs(fix_window_size/4-max_loc_p_tr_grad)
            num_af = np.abs(fix_window_size*3/4-(len(dp_tr) - max_loc_p_tr_grad))
            #print(max_loc_p_tr_grad, len(dp_tr), num_bf, num_af)
            dp_tr = np.concatenate((np.ones(num_bf)*p_tr_mean_bf, dp_tr, np.ones(num_af)*p_tr_mean_af))
            dq_tr = np.concatenate((np.ones(num_bf)*q_tr_mean_bf, dq_tr, np.ones(num_af)*q_tr_mean_af))

        # if len(dp_tr) > fix_window_size:
        #     max_loc_p_tr_grad = np.argmax(np.abs(dp_tr[1:] - dp_tr[:-1]))
            
        #     if max_loc_p_tr_grad <= fix_window_size/4:
        #         num_bf = fix_window_size/4-max_loc_p_tr_grad
        #         num_af = (len(dp_tr) - max_loc_p_tr_grad) - fix_window_size*3/4

        #         p_tr_mean_bf = 0
        #         q_tr_mean_bf = 0

        #         dp_tr = np.concatenate((np.ones(num_bf)*p_tr_mean_bf, dp_tr[:-num_af]))
        #         dq_tr = np.concatenate((np.ones(num_bf)*q_tr_mean_bf, dq_tr[:-num_af]))
                
        #     else:
        #         num_bf = max_loc_p_tr_grad - fix_window_size/4
        #         num_af = max(0,fix_window_size*3/4-(len(dp_tr) - max_loc_p_tr_grad))
        #         p_tr_mean_af = databuffer['p'].iloc[event_end_mean:event_end].mean() \
        #         - databuffer['p'].iloc[event_start:event_start_mean].mean()
        #         q_tr_mean_af = databuffer['q'].iloc[event_end_mean:event_end].mean() \
        #         - databuffer['q'].iloc[event_start:event_start_mean].mean()
        #         print(max_loc_p_tr_grad, len(dp_tr), num_bf, num_af)
        #         print('p values is :', p_value[event_start:event_end], p_mean_grad[event_start:event_end])
        #         dp_tr = np.concatenate((dp_tr[num_bf:], np.ones(num_af)*p_tr_mean_af))
        #         dq_tr = np.concatenate((dq_tr[num_bf:], np.ones(num_af)*q_tr_mean_af))

        # h_fix_size = fix_window_size/16
        # dh_first = dh[:,0]
        # dh_third = dh[:,2]
        # dh_fifth = dh[:,4]

        # if len(dh) <= h_fix_size:
            
        #     dh_grad_max_loc = np.argmax(np.abs(dh_first[1:] - dh_first[:-1]))

        #     dh_num_bf = np.abs(h_fix_size/4 - dh_grad_max_loc)
        #     dh_num_af = np.abs(h_fix_size*3/4 - (len(dh_first) - dh_grad_max_loc))

        #     dh_first = np.concatenate((np.ones(dh_num_bf)*dh_first[0], dh_first, np.ones(dh_num_af)*dh_first[-1]))
        #     dh_third = np.concatenate((np.ones(dh_num_bf)*dh_third[0], dh_third, np.ones(dh_num_af)*dh_third[-1]))
        #     dh_fifth = np.concatenate((np.ones(dh_num_bf)*dh_fifth[0], dh_fifth, np.ones(dh_num_af)*dh_fifth[-1]))
        # if len(dh) > h_fix_size:
        #     dh_grad_max_loc = np.argmax(np.abs(dh_first[1:] - dh_first[:-1]))

        #     if dh_grad_max_loc <= h_fix_size/4:
        #         dh_num_bf = h_fix_size/4 - dh_grad_max_loc
        #         dh_num_af = (len(dh_first) - dh_grad_max_loc) - h_fix_size*3/4

        #         dh_first = np.concatenate((np.ones(dh_num_bf)*dh_first[0], dh_first[:-dh_num_af]))
        #         dh_third = np.concatenate((np.ones(dh_num_bf)*dh_third[0], dh_third[:-dh_num_af]))
        #         dh_fifth = np.concatenate((np.ones(dh_num_bf)*dh_fifth[0], dh_fifth[:-dh_num_af]))
        #     else:
        #         dh_num_bf = dh_grad_max_loc - h_fix_size/4
        #         dh_num_af = max(0, h_fix_size*3/4 - (len(dh_first) - dh_grad_max_loc))
        #         print('dh_num is:', dh_num_bf, dh_num_af, dh_grad_max_loc)
        #         dh_first = np.concatenate((dh_first[dh_num_bf:], np.ones(dh_num_af)*dh_first[-1]))
        #         dh_third = np.concatenate((dh_third[dh_num_bf:], np.ones(dh_num_af)*dh_third[-1]))
        #         dh_fifth = np.concatenate((dh_fifth[dh_num_bf:], np.ones(dh_num_af)*dh_fifth[-1]))

        # dh = np.array([dh_first, dh_third, dh_fifth])

        # dp_tr = dp_tr - dp_tr.min()
        # dq_tr = dq_tr - dq_tr.min()
        # du_tr = du_tr - du_tr.min()
        # di_tr = di_tr - di_tr.min()
        ##################################################################################

        #others
        if dp_s > delta_p*0.5:
            p_n = 1
        elif dp_s < -delta_p*0.5:
            p_n = 0
        else:
            p_n = -1
        #print('p_n is ', p_n)
        ##################################################################################
        #save feature to datafream and send it to queue
        feature=pd.DataFrame({'dp_tr' : [dp_tr], 'dq_tr' : [dq_tr],'dp_s' : dp_s, \
         'dq_s' : dq_s, 'dp_dq' : dp_dq, 'dh_s':[dh_s],\
         'time_stamp' : time_stamp, 'p_n': p_n, 'event_loc':[event_loc], 'p_trend':[p_trend],\
         'p_value': [p_value], 'p':[databuffer['p'].values], 'label':[label], 'debug':[debug_list]})
        if p_n > -1:
            q1.put(feature)
        databuffer = databuffer.drop(databuffer.index[:event_end])

    return databuffer