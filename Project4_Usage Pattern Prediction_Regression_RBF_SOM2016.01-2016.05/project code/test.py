from dataBase import *
from dataExtrc import *
import matplotlib.pyplot as plt
import numpy as np
user=dataBase()
user.welcome()
user.preDB()
dataext=dataExtrc()
result=user.openDB(dataext)

#plt.plot(result['total_power'])
#plt.show()
y_list=list(result['total_power'])
time_std=list(result.index)

x=np.matrix(time_std).T
y=np.matrix(result['total_power']).T

mean_x=np.mean(x)
std_x=np.std(x)

mean_y=np.mean(y)
std_y=np.std(y)

for i in range(len(list(result.index))):
    time_std[i]=(time_std[i]-mean_x)/std_x

for i in range(len(y)):
    y[i]=-(y[i]-mean_y)/std_y

x=np.array(time_std).T.reshape(-1, 1)
length=len(y)

plt.plot(x,y,'r+')
#plt.show()
