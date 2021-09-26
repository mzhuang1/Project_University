# -*- coding: utf-8 -*-

import pickle
from user_profile import *
import threading
import time, datetime

def bbb_sort(df):
    return np.r_[df[0],df[1],df[2],df[3],df[6],1]

def bbb_tcplink(sock, addr):
    u = User()

    counter = 0
    buffer_size = 8232
    data = sock.recv(65536)
    while True:
        while len(data) <= buffer_size:
           data = data+sock.recv(65536)
           if not data:
               break
           continue
        #print('the time is:', time.time())
        databuffer = data[:buffer_size]
        data = data[buffer_size:]
        #print('socket data length is:',len(databuffer))
        try:
            socket_data = pickle.loads(databuffer, encoding="bytes")
        except EOFError:
            continue
        print('get data and the couunter is:', counter)
        counter += 1
        socket_data = bbb_sort(socket_data)
        #print(socket_data)
        real_power, reactive_power, device_id, time_stamp = online_receiving_save2mongodb(socket_data)
        u.device_id = device_id
        u.concat_new_data(real_power,real_power)
        while u.databuffer['p'].size >= base_window:
            #print(u.databuffer.size)
            ev = u.event_detect()
            
            if ev == -1:
                break
            
            if not ev:
                continue
            label = u.predict_save_feature(ev, time_stamp)
            print('label is', label)

    sock.close()
    print('connection from %s:%s closed.' % addr)

def tcplink(sock, addr):

    u = User()


    while True:

        databuffer = sock.recvfrom(4096)[0]
        #print('socket data length is:',len(databuffer))
        try:
            socket_data = pickle.loads(databuffer, encoding="bytes")
        except:
            print('error, please check')
            continue
        #print(len(socket_data))
        real_power, reactive_power, device_id, time_stamp = online_receiving_save2mongodb(socket_data)
        #print('extraction compeleted')
        u.device_id = device_id
        u.concat_new_data(real_power,real_power)
        while u.databuffer['p'].size >= base_window:
            #print(u.databuffer.size)
            ev = u.event_detect()
            #print(ev)
            if ev == -1:
                break
            
            if not ev:
                continue
            label = u.predict_save_feature(ev, time_stamp)
            print('label is', label)
            print(datetime.datetime.now())

    sock.close()
    print('connection from %s:%s closed.' % addr)
    


#UDP
s = socket(AF_INET, SOCK_DGRAM)
s.bind(addr)

print('ready to recevie data')

t = threading.Thread(target=tcplink, args=(s, addr))
t.start()
    