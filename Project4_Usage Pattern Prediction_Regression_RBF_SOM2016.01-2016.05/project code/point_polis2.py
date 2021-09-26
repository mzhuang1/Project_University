import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
start=2
combine=np.load('date13.npy')[start]
test=np.load('date13.npy')[start+1]
for i in range(14,32):
    filename='date'+str(i)+'.npy'
    temp=np.load(filename)[start]
    temp1=np.load(filename)[start+1]
    test=np.hstack((test,temp1))
    combine=np.hstack((combine,temp))
train=combine[0:10]
result_record=train
target=combine[10:11]
predictd=test[0:10]
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
time=np.array([i+1 for i in range(19)])
for i in range(len(train)+1,len(combine)+1):
    y_rbf = svr_rbf.fit(train.reshape(1,-1),target).predict(predictd.reshape(1,-1))
    train=combine[0:i]
    target=combine[i:i+1]
    predictd=test[0:i]
    result_record=np.hstack((result_record,y_rbf))
plt.scatter(time,combine , color='darkorange', label='data')
plt.hold('on')
plt.plot(time, result_record, color='navy', label='RBF model')
plt.ylabel('Total_power')
plt.xlabel('prediction in days')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

print('compute the error rate of the model:')
count=0
if len(combine) is not len(result_record):
        raise ValueError
for i in range(len(combine)):
    if combine[i]-result_record[i]<= 10:
        count=count+1
print('the program correctly predict '+str(count/len(combine)*100)+' % of total test data')
    
        
