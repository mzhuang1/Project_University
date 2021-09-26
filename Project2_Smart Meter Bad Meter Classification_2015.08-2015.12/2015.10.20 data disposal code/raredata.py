__author__ = 'ray+zhuang'
import datetime
import numpy as np
import csv_io
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt, savetxt
import time
import matplotlib as plt
from pylab import *

def filehandler(pathtrain,pathtest,pathout):
    dataset = genfromtxt(open(pathtrain, 'r'), delimiter=',',dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open(pathtest, 'r'), delimiter=',',dtype='f8')[1:]
    #rf=RandomForestClassifier(n_estimators=100, n_jobs=4)
    #rf.fit(train,target)
    #savetxt(pathout, rf.predict(test), delimiter=',', fmt='%f')

if __name__=='__main__':
    pathloaddata = 'D:/data/meter/smart meter materials/loaddata.csv'
    pathtem = 'D:/data/meter/smart meter materials/temperatures.csv'
    pathwea = 'D:/data/meter/smart meter materials/weatherchicago.csv'

    loaddata = csv_io.read_data(pathloaddata)
    metertem = csv_io.read_data(pathtem)
    wea = csv_io.read_data(pathwea)
    # load meter loaddata in each hour
    meterloaddic = {}
    for i in range(0, len(loaddata), 1):
        strtime = time.strptime(loaddata[i][3], "%Y-%m-%d %H:%M:%S")
        ftime = time.strftime("%Y%m%d%H%M", strtime)
        meterid = loaddata[i][1][1:-1]
        meterload = float(loaddata[i][4])
        if meterloaddic.has_key(meterid):
            meterloaddic[meterid][ftime]=meterload
        else:
            meterloaddic[meterid]={}
            meterloaddic[meterid][ftime]=meterload

    #load weather data
    weadic={}
    for i in range(0, len(wea), 1):
        date = wea[i][0]
        weathertem = wea[i][1]
        if weathertem.isdigit():
            weadic[date]=weathertem
            dateh = long(date)+30
            weadic[str(dateh)] = weathertem
        else:
            weadic[date]=60
            dateh = long(date)+30
            weadic[str(dateh)] = 60
    #load meter temperatrure

    metertemdic = {}# 二维map，第一层主键是电表id，第二维主键是电表实际温度对应的时间

    for i in range(0,len(metertem),1):
        strtime = time.strptime(metertem[i][5], "%Y/%m/%d %H:%M:%S")
        ftime = time.strftime("%Y%m%d%H%M", strtime)
        meterid = metertem[i][1][1:-1]
        mtem = float(metertem[i][6])
        if metertemdic.has_key(meterid):
            metertemdic[meterid][ftime]=mtem
        else:
            metertemdic[meterid]={}
            metertemdic[meterid][ftime]=mtem
    c=0
    # 对于在真实电表温度数据中存在的电表进行取数据操作
    for key in metertemdic.keys():
        #if c>3:
        #    break
        c=c+1
        #tldic = dicmeterload.get(key)
            #判断电表负载数据中是否存在该电表
            if meterloaddic.has_key(key):
                #path = 'D:/data/meter/result-2daytrain-dayfeature'+str(key)+'.txt'
                #print path
                #output = open(path, 'w+')
                pathf = 'D:/data/meter/date-tem-weatem-load/date-tem-weatem-load-rare/'+str(key)+'.txt'
                #print pathf
                outputf = open(pathf, 'w+')
                dateloaddic = meterloaddic[key]
                #print(len(dateloaddic))

                datetemdic = metertemdic[key] #得到特定电表对应的时间温度词典
                WY = []
                TY = []
                X = []
                datefs = []
                train = []
                datetemp = ''
                temtemp = 0
                #按照特定电表的日期升序递归筛选数据
                for date in sorted(datetemdic.keys()):
                    #调整日期格式
                    minutes = long(date) % 100
                    head = long(date)/100
                    end = minutes/30
                    if end > 1:
                        print date
                    datef = head*100+end*30

                    #选择2014年的电表数据
                    if str(datef).startswith('2014'):
                        #print (date)
                        #print (datef)
                        weatem0 = 60  #如果环境温度为空，则默认为60度
                        load01 = 0.1   #如果负载数据为空，则默认为0.1
                        item = []   #存放本次循环要记录的数据
                        #判断负载数据中有没有改日期，没有的话则跳过
                        if dateloaddic.has_key(str(datef)):
                            print str(datef)
                            weatem = weatem0
                            if(weadic.has_key(str(datef))):
                                weatem = weadic[str(datef)]
                                weatem0 = weatem
                            tem = datetemdic[date]

                            WY.append(float(weatem))
                            TY.append(float(tem))
                            datefs.append(float(datef))
                            item.append(str(datef/100))
                            item.append(float(tem))
                            item.append(float(weatem))
                            item.append(float(dateloaddic[str(datef)]))
                            train.append(item)
                            datetemp = datef
                            temtemp = float(tem)
                            load01= float(dateloaddic[str(datef)])
                trainnum = 0
                featurenum = 1  #
                for j in range(trainnum+featurenum-1, len(train), 1):
                    temp = []
                    for p in range(j, j-featurenum, -1):
                        temp.append(train[p][0])
                        temp.append(train[p][1])
                        temp.append(train[p][2])
                        temp.append(train[p][3])
                    outputf.write(str(temp[0])+'\t'+str(temp[1])+'\t'+str(temp[2])+'\t'+str(temp[3])+'\n')
                outputf.close()


