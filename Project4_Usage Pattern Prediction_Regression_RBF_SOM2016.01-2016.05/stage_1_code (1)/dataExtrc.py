import pandas as pd
import pickle
from time_trans import *
class dataExtrc(object):
    def __init__(self):
        self.voltage_matrix=[]
        self.current_matrix=[]
        self.real_power_matrix=[]
        self.reactive_power_matrix=[]
        self.time_matrix=[]

    def result(self,cursor,end):
        n=-1
        for entry in cursor:
            print(n)
            n=n+1
            data = [pickle.loads(entry['rawData'])][0]
            self.voltage_matrix.append(data[0])
            self.current_matrix.append(data[1])
            self.real_power_matrix.append(data[2])
            self.reactive_power_matrix.append(data[3])
            self.time_matrix.append(data[-1])                ##check
            print(end-self.time_matrix[-1])

        dict={"voltage":self.voltage_matrix,"current":self.current_matrix,"real_power":self.real_power_matrix,"reactive_power":self.reactive_power_matrix}
        result=pd.DataFrame(dict,index=self.time_matrix)

        return result

    def linear_approximation(self,startDate,control_const=100):
        estimate_const=0.3
        #60047 for a day/1786652 for whole database
        max_Reach=control_const**2
        default_Date="2016-01-13 07:00:00"
        default_Date=datetime_timestamp(default_Date)
        skip_entry=int((startDate-default_Date)*estimate_const)
        if skip_entry>max_Reach:
            skip_entry=max_Reach
        if skip_entry<0:
            skip_entry=0
        return  skip_entry
        
    def output_parse(matrix):
    ##function defined for output data formating.It is optional.
    ##I will comment when making representation
        i=0
        for ele in matrix:
            i=i+1
            print("-----------------next entry-----------------")
            print("entry ",i,ele)
        
