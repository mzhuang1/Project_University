import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
combine=np.load('date13.npy')
for i in range(14,32):
    filename='date'+str(i)+'.npy'
    temp=np.load(filename)
    combine=np.vstack((combine,temp))
train=combine[:-2,1:3374]
target=combine[:-2,-3]

predictd=combine[1:-1,1:3374]

time=np.array([i for i in range(17)])
#print(combine.shape)    
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
y_rbf = svr_rbf.fit(train,target).predict(predictd)
plt.scatter(time, target, color='darkorange', label='data')
plt.plot(time, target, color='darkorange', label='RBF model')
plt.hold('on')
plt.plot(time, y_rbf, color='navy', label='RBF model')
plt.xlabel('Total_power')
plt.ylabel('prediction')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
