__author__ = 'ray+zhuang'
import datetime
import numpy as np
import csv_io
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt, savetxt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
    sc1 = 0
    scr1 = 0
    sc2 = 0
    scr2 = 0
    sc3 = 0
    scr3 = 0
    spr = 0
    sp = 0
    pathout = 'D:/data/meter/date-tem-weatem-load/date-tem-weatem-load-completion-cla-precision.txt'
    outputo = open(pathout, 'w+')
    for key in metertemdic.keys():
        #if c>3:
        #    break
        c1 = 0
        cr1 = 0
        c2 = 0
        cr2 = 0
        c3 = 0
        cr3 = 0
        pr = 0
        p = 0
        if meterloaddic.has_key(key):
            #path = 'D:/data/meter/result-2daytrain-dayfeature'+str(key)+'.txt'
            #print path
            #output = open(path, 'w+')
            pathf = 'D:/data/meter/date-tem-weatem-load/date-tem-weatem-load-completion-cla/'+str(key)+'.txt'
            print pathf
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
            datetemp = ''
            temtemp = 0
            for date in sorted(datetemdic.keys()):
                minutes = long(date) % 100
                head = long(date)/100
                end = minutes/30
                #if end > 1:
                    #print date
                datef = head*100+end*30
                if str(datef).startswith('2013123123'):
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
                                        cla1 = 1
                                        if float(tem01)>=120:
                                            cla1 = 3
                                        elif float(tem01)>=100:
                                            cla1 = 2
                                        else:
                                            cla1 = 1
                                        item1.append(int(cla1))
                                        train.append(item1)
                                        tem1 = tem01

                            if str(datef/100).endswith('01'):
                                #print str(datef)
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
                                cla2 = 1
                                if float(tem02)>=120:
                                    cla2 = 3
                                elif float(tem02)>=100:
                                    cla2 = 2
                                else:
                                    cla2 = 1
                                item2.append(int(cla2))
                                train.append(item2)
                        WY.append(float(weatem))
                        TY.append(float(tem))
                        datefs.append(float(datef))
                        item.append(str(datef/100))
                        item.append('%.2f'%float(tem))
                        item.append(float(weatem))
                        item.append(float(dateloaddic[str(datef)]))
                        cla = 1
                        if float(tem)>=120:
                            cla = 3
                        elif float(tem)>=100:
                            cla = 2
                        else:
                            cla = 1
                        item.append(int(cla))
                        train.append(item)
                        datetemp = datef
                        temtemp = float(tem)
                        load01= float(dateloaddic[str(datef)])
                        #print str(datef)
            real = []
            predict = []
            trainnum = 49 #49+7  7+2
            featurenum = 3
            for j in range(trainnum+featurenum-1, len(train), 1):
                TR = []
                TE = []
                #TE.append(train[j])
                temp = []
                temp.append(train[j][4])
                temp.append(train[j][0])
                temp.append(train[j][1])
                temp.append(train[j][2])
                temp.append(train[j][3])
                for p in range(j-1, j-featurenum, -1):
                    temp.append(train[p][1])
                    temp.append(train[p][2])
                    temp.append(train[p][3])
                TE.append(temp)
                for k in range(j-trainnum-featurenum, j, 1):
                    temp1 = []
                    temp1.append(train[k][4])
                    temp1.append(train[k][0])
                    temp1.append(train[k][1])
                    temp1.append(train[k][2])
                    for p in range(k-1, k-featurenum, -1):
                        temp1.append(train[p][1])
                        temp1.append(train[p][2])
                        temp1.append(train[p][3])
                    TR.append(temp1)
                lr = LogisticRegression()
                gnb = GaussianNB()
                svc = LinearSVC(C=1.0)
                rfc = RandomForestClassifier(n_estimators=50)
                lend = 3*featurenum+1
                nptr = np.asarray(TR)
                npte = np.asarray(TE)

                for clf, name in [(rfc, 'Random Forest')]: #(lr, 'Logistic'),(gnb, 'Naive Bayes'),(svc, 'Support Vector Classification'),
                    clf.fit(nptr[:, 2:lend], nptr[:, 0])
                    clf_c = clf.predict(npte[0, 2:lend])

                    #print npte[0, 0]
                    #print clf_c[0]
                    outputf.write(npte[0, 1]+'\t'+npte[0, 2]+'\t'+npte[0, 0]+'\t'+clf_c[0]+'\n')
                    if int(npte[0,0])==int(clf_c[0]):
                        spr = spr+1
                        pr = pr+1
                    sp = sp+1
                    p = p+1
                    if int(npte[0,0]) ==1:
                        c1 = c1+1
                        sc1 = sc1+1
                        if int(clf_c[0])==1:
                            cr1 = cr1+1
                            scr1 = scr1+1
                    if int(npte[0,0]) ==2:
                        c2 = c2+1
                        sc2 = sc2+1
                        if int(clf_c[0])==2:
                            scr2 = scr2+1
                            cr2 = cr2+1
                    if int(npte[0,0]) ==3:
                        sc3 = sc3+1
                        c3 = c3+1
                        if int(clf_c[0])==3:
                            scr3 = scr3+1
                            cr3 = cr3+1

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
            outputf.write('p:'+str(p)+'\tpr:'+str(pr)+'\tc1:'+str(c1)+'\tcr1:'+str(cr1)+'\tc2:'+str(c2)+'\tcr2:'+str(cr2)+'\tc3:'+str(c3)+'\tcr3:'+str(cr3)+'\n')
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
    outputo.write('sp:'+str(sp)+'\tspr:'+str(spr)+'\tsc1:'+str(sc1)+'\tscr1:'+str(scr1)+'\tsc2:'+str(sc2)+'\tscr2:'+str(scr2)+'\tsc3:'+str(sc3)+'\tscr3:'+str(scr3)+'\t')
    print('sp:'+str(sp)+'\tspr:'+str(spr)+'\tsc1:'+str(sc1)+'\tscr1:'+str(scr1)+'\tsc2:'+str(sc2)+'\tscr2:'+str(scr2)+'\tsc3:'+str(sc3)+'\tscr3:'+str(scr3)+'\t')

