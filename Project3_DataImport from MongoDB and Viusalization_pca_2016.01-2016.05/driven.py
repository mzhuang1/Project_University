from pymongo import MongoClient
import datetime
import pickle,numpy
#import pandas as pd
print("###############")
print("speed-up-input!")
print("###############")
print("Please make sure that a MongoDB instance is running on a host!!!!!!!!\n")
print("please input your hostname,d for default")
hostname=input()
if hostname[0]=='d':
    hostname="localhost"
port=input("please input port number,d for default\n")
try:
    port=int(port)
except Exception as err:
    port=27017

client = MongoClient(hostname,port)

db=client.lab
collection=db.main
print(collection.count())
collector = collection.find()
begin = datetime.datetime.now()         ##check the start time
print(collector)
n=0
voltage_matrix=[]
current_matrix=[]
real_power_matrix=[]
reactive_power_matrix=[]
time_matrix=[]

#combine_matrix=[]                       ##bad way for implementation as required


##entry=collection.find_one()
##data = [pickle.loads(entry['rawData'])][0]
##print(type(entry))
for entry in collection.find():
    n=n+1
    if(n>70):
        break
    data = [pickle.loads(entry['rawData'])][0]
    voltage_matrix.append(data[0])
    current_matrix.append(data[1])
    real_power_matrix.append(data[2])
    reactive_power_matrix.append(data[3])
    time_matrix.append(data[-1])
    ##Actually this is a bad way to meet requirements, but in case you like, I 
    ##leave the code here, but I will not use it.
##    combine_matrix.append(data[0])
##    combine_matrix.append(data[1])
##    combine_matrix.append(data[2])
##    combine_matrix.append(data[3])
##    combine_matrix.append(data[-1])
    
end = datetime.datetime.now()           ##check the end time
print("time usage: ")
print(end-begin)                        ##output result
combined_table=pd.DataFrame({'voltage':voltage_matrix,'current':current_matrix,'real_power':real_power_matrix,'reactive_power':reactive_power_matrix},index=time_matrix)
print(combined_table)

def output_parse(matrix):
    ##function defined for output data formating.It is optional.
    ##I will comment when making representation
    i=0
    for ele in matrix:
        i=i+1
        print("-----------------next entry-----------------")
        print("entry ",i,ele)
        




































#dict=collection.find_one()
#print(dict)

