"""Statistic variables

frequency = 60 							
steady_before_window = 10				
steady_after_window = 10				
base_window = int(frequency*5)			
fixed_window = 1000						
delta_p = 9								
min_event_interval = int(frequency*0.2)


encoder_layer_1_num = 500
encoder_layer_2_num = 16


from pymongo import MongoClient

client = MongoClient()
db = client.nilm

addr = ("216.47.129.145", 9999)
receiving_data_time = 1
data_per_sec = 64

voltage_index = data_per_sec*receiving_data_time
current_index = 2*data_per_sec*receiving_data_time
real_power_index = 3*data_per_sec*receiving_data_time
reactive_power_index = 4*data_per_sec*receiving_data_time
