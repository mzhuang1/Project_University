import socket
from multiprocessing import Process, Queue
import numpy as np
import time, pickle
from save2Database import *
import os
from sklearn.decomposition import PCA
from sklearn import mixture

exec(open('parameters_new.py').read())
def recv(q, addr):
	try:
		# Set the socket parameters
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.bind(addr)
		s.listen(10)
		conn, addr = s.accept()
		print("connected by client:", addr)
		data_buf = bytes()
		databuffer=pd.DataFrame(columns=['u','i','p','q','harmony','emi'])
		#s.listen(1)
		#conn, addr = s.accept()
		#print('Connected by', addr)
		# Receive message
		ui_len = 8230
		while 1:
			start = time.time()
			#print(len(data))
			if len(data_buf) >= ui_len:
				data = pickle.loads(data_buf[:ui_len], encoding="bytes")
				data_buf = data_buf[ui_len:]
				#print(len(data_buf))
				try:
					databuffer = ev_detect(data,databuffer, q)
				except:
					print('something wrong happened, trying to reconstruct algorithm!')
					databuffer=pd.DataFrame(columns=['u','i','p','q','harmony','emi'])
				saveRaw2mdb(db,data)
				#print("consume time:", time.time()-start, "in process :")
			else:
				data_buf = data_buf + conn.recv(2**16)
				#conn.send(1)
				#print("fail to get the data at :", time.time())
		# Close socket
	except:
		pass
	try:
		s.close()
		print("waiting for client now!")
	except:
		pass



     
def classifier(q):
#    if os.path.exists(r'features.pkl'):
#        feature = pickle.load(open('features.pkl', 'rb'))
#    else:
#        feature=pd.DataFrame([], columns=['dp_tr','dp_t','dq_tr','dq_t','du_tr','du_t','di_tr','di_t','dp_s','dq_s','du_s','di_s','dp_dq','first_h','third_h','fifth_h','demi','time_stamp','p_n'])
    while 1:
        start = time.time()
        data=q.get(True)
        saveFeature2mdb(db,data)
        if data.p_n[0] == 1:
            print('I know something was started up, let me guess what it is!')
        elif data.p_n[0] == 0:
            print('I know something was shut down, let me guess what it is!')
        #feature=pd.concat([feature,data]) 
        try:
	        model = readModel(db)
	        if data.p_n[0] == 1:
	            cl_up = model["af_up"]
	            sample = data.loc[:,data_index_woemi]

	            print('The event number of starting applicances is:  +++++++',cl_up.predict(data.loc[:,data_index_woemi])[0], 
	            '+++++++, at time:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data.time_stamp)))
	            
	        elif data.p_n[0] == 0:
	            neigh = model["KNN"]
	            sample = np.abs(data.loc[:,data_index_woemi_stable])
	            
	            print('The event number of stoping applicances is:   -------',neigh.predict(np.abs(data.loc[:,data_index_woemi_stable]))[0], 
	            '-------, at time:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data.time_stamp)))
	#                print('The event number of stoping applicances is:   -------',cl_down.predict(sample_norm)[0], 
	#                '-------, at time:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data.time_stamp)))
                
        except:
            pass
        #print('consume time:', time.time()-start)

if __name__ == '__main__':
	while 1:
		q = Queue()
		host = '104.194.124.231'#'192.168.1.111'#
		port = 9999
		addr = (host,port)
		# Create socket and bind to address
		
		p_recv = Process(target = recv, args = (q, addr, ))
		p_classifier = Process(target = classifier, args = (q,))
		p_recv.start()
		p_classifier.start()
		p_recv.join()
		p_classifier.terminate()
		time.sleep(10)
		print('restart to be ready for new client!')
