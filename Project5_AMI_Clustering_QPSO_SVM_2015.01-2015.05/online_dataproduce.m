function [MIX_train,MIX_test,DATA_train,DATA_test]=online_dataproduce()

shu1=xlsread('focus2.xlsx');
MIX_train=shu1(179:3378,(2:3))';
DATA_train=shu1(179:3378,1)';
shu2=xlsread('focus2.xlsx');
MIX_test=shu2(1:178,(2:3))';
DATA_test=shu2(1:178,1)';
