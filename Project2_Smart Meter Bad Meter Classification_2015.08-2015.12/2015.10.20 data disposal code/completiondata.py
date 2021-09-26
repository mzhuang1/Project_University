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
        #print meterload
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

    metertemdic = {}
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
    for key in metertemdic.keys():
        #if c>3:
        #    break
        c=c+1
        if c < 160:
            print(c)
        #tldic = dicmeterload.get(key)
            if meterloaddic.has_key(key):
                #path = 'D:/data/meter/result-2daytrain-dayfeature'+str(key)+'.txt'
                #print path
                #output = open(path, 'w+')
                pathf = 'D:/data/meter/date-tem-weatem-load/date-tem-weatem-load-completion/'+str(key)+'.txt'
                #print pathf
                outputf = open(pathf, 'w+')
                dateloaddic = meterloaddic[key]
                #print(len(dateloaddic))

                datetemdic = metertemdic[key]
                WY = []
                TY = []
                X = []
                datefs = []
                train = []
                test = []
                datetemp = '' # record the previous date
                temtemp = 0  # record the previous data
                for date in sorted(datetemdic.keys()):
                    minutes = long(date) % 100
                    head = long(date)/100
                    end = minutes/30
                    #if end > 1:
                        #print date
                    datef = head*100+end*30
                    if str(datef).startswith('2013123123'):  # use the meter temperature of 2013123123 and 2014010101 to generate the meter temperature of 2014010100
                        temtemp = datetemdic[str(date)]
                        datetemp = str(datef)
                    if str(datef).startswith('2014'):
                        #print (date)
                        #print (datef)
                        weatem0 = 60
                        load01 = 0.1
                        item = []
                        if dateloaddic.has_key(str(datef)):
                            weatem = weatem0
                            if(weadic.has_key(str(datef))):
                                weatem = weadic[str(datef)]
                                weatem0 = weatem
                            tem = datetemdic[date]
                            if len(str(datetemp))>0:
                                #print str(datetemp)[:-4]
                                #print str(datef)[:-4]
                                # generate the null data between this hour and the previous hour
                                if str(datetemp)[:-4]==str(datef)[:-4]:
                                    starth = str(datetemp)[-4:-2]
                                    endh=str(datef)[-4:-2]
                                    dis = int(endh)-int(starth)
                                    #print dis
                                    if dis > 1:
                                        unit = (float(tem)-temtemp)/dis
                                        #print unit
                                        weatem01 = 60
                                        tem1 = temtemp

                                        for j in range(int(starth)+1, int(endh)):
                                            #print j
                                            item1 = []
                                            #print tem1
                                            weatem1 = weatem01
                                            if(weadic.has_key(str((datef/10000*100+j)*100))):
                                                weatem1 = weadic[str((datef/10000*100+j)*100)]
                                                weatem01 = weatem1
                                            load1 = load01
                                            if dateloaddic.has_key(str((datef/10000*100+j)*100)):
                                                load1 = float(dateloaddic[str((datef/10000*100+j)*100)])
                                            item1.append(str(datef/10000*100+j))
                                            tem01 = tem1 + unit
                                            item1.append('%.2f'%tem01)
                                            item1.append(float(weatem1))
                                            item1.append(load1)
                                            train.append(item1)
                                            tem1 = tem01
                                # if the date hour is 01, then generate the data of 00
                                if str(datef/100).endswith('01'):
                                    print str(datef)
                                    item2 = []
                                    tem02 = (temtemp+tem)/2
                                    weatem02 = 60
                                    load02 = 0.1
                                    weatem2 = weatem02
                                    if(weadic.has_key(str((datef/10000*100)*100))):
                                        weatem2 = weadic[str((datef/10000*100)*100)]
                                        weatem02 = weatem2
                                    load2 = load02
                                    if dateloaddic.has_key(str((datef/10000*100)*100)):
                                        load2 = float(dateloaddic[str((datef/10000*100)*100)])
                                    item2.append(str(datef/10000*100))
                                    item2.append('%.2f'%tem02)
                                    item2.append(float(weatem2))
                                    item2.append(load2)
                                    train.append(item2)
                            WY.append(float(weatem))
                            TY.append(float(tem))
                            datefs.append(float(datef))
                            item.append(str(datef/100))
                            item.append('%.2f'%float(tem))
                            item.append(float(weatem))
                            item.append(float(dateloaddic[str(datef)]))
                            train.append(item)
                            datetemp = datef
                            temtemp = float(tem)
                            load01= float(dateloaddic[str(datef)])
                            #print str(datef)
                real = []
                predict = []
                trainnum = 0 #49+7  7+2
                featurenum = 1
                for j in range(trainnum+featurenum-1, len(train), 1):
                    TR = []
                    TE = []
                    #TE.append(train[j])
                    temp = []
                    for p in range(j, j-featurenum, -1):
                        temp.append(train[p][0])
                        temp.append(train[p][1])
                        temp.append(train[p][2])
                        temp.append(train[p][3])
                    TE.append(temp)
                    for k in range(j-trainnum-featurenum, j, 1):
                        temp1 = []
                        for p in range(k, k-featurenum, -1):
                            temp1.append(train[p][0])
                            temp1.append(train[p][1])
                            temp1.append(train[p][2])
                        TR.append(temp1)
                    #regressor = RandomForestRegressor(n_estimators=150, min_samples_split=1)
                    #nptr = np.asarray(TR)
                    #npte = np.asarray(TE)
                    #print (temp)
                    #regressor.fit(nptr[:, 1:17], nptr[:, 0])
                    #result = []
                    #result.append(npte[0, 0])
                    #pre = regressor.predict(npte[:, 1:17])
                    #result.append(pre[0])

                    #print result
                    #real.append(str(result[0]))
                    #predict.append(str(result[1]))
                    #output.write(str(result[0])+'\t'+str(result[1])+'\n')
                    outputf.write(str(temp[0])+'\t'+str(temp[1])+'\t'+str(temp[2])+'\t'+str(temp[3])+'\n')
                outputf.close()
                #X = range(0, len(real), 1)
                #plt.plot(X, real, 'r-', label = 'Real temperature')
                #plt.plot(X, predict, 'b-', label = 'Prediction temperature')
                #plt.title(key+'-2daytrain-dayfeature')
                #plt.xticks(X, datefs)
                #print (X)
                #legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
                #plt.savefig('D:/data/meter/prediction-2daytrain-dayfeature'+str(key)+'.png', dpi=75)
                #plt.show()

