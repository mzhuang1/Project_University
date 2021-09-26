import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
yesterday=np.load('raw_yesterday.npy')
today=np.load('raw_today.npy')
yesterday=yesterday[0:1000]
yesterday=yesterday.reshape(len(yesterday),1)
today=today[0:1000]
today=today.reshape(len(today),1)

#combine=np.vstack((yesterday,today)).T
target=np.load('raw_tomorrow.npy')
target=target[0:1000]
#target=target.reshape(len(target),1)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
#y_rbf = svr_rbf.fit(combine, target).predict(combine)
y_rbf = svr_rbf.fit(yesterday, target).predict(yesterday)
lw = 2
y_rbf=y_rbf.reshape(len(y_rbf),1)
plt.scatter(yesterday, target, color='darkorange', label='data')
plt.hold('on')
plt.plot(yesterday, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('Total_power')
plt.ylabel('prediction')
plt.title('Support Vector Regression')
plt.legend()
plt.hold('off')
plt.show()


y_rbf2 = svr_rbf.fit(yesterday, target).predict(today)
lw = 2
y_rbf2=y_rbf.reshape(len(y_rbf),1)
plt.scatter(yesterday, target, color='green', label='data')
plt.hold('on')
plt.plot(yesterday, y_rbf2, color='pink', lw=lw, label='RBF model')
plt.xlabel('Total_power')
plt.ylabel('prediction')
plt.title('Support Vector Regression')
plt.legend()
plt.hold('off')
plt.show()
