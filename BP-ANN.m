clear all
clc
data2015 = xlsread('2015_smd_hourly.xls','NEMASSBost');
day = 31; %selected day number
datanum = day*24;
index1 = 1:1:datanum; %1/1/2015-1/31/2015
%index2 = 1:1:24;    %24 h
load1 = data2015(index1,3);         %Jan. load
tempdry1 = data2015(index1,12);    %Jan. dry temp
tempwet1 = data2015(index1,13);    %Jan. wet temp
%P
P1 = zeros(1, day-2);       %because of D-1 and D does not include so -2.
for n = 1:day-2             %P1 reads day #
    P1(n) = n+2;            %starting from day 3
end
P2 = zeros(24,day-2);       %p2 reads hourly load
for n = 1:day-2
    for m = 1:1:24
        x = (n-1)*24;
        P2(m,n) = load1(x+m);
    end
end
P3 = zeros(24,day-2);       %p3 reads actual dry temp.
for n = 1:day-2
    for m = 1:1:24
        x = (n-1)*24;
        P3(m,n) = tempdry1(x+m);
    end
end
P4 = zeros(24,day-2);       %P4 reads forcatsted dry temp.
for n = 1:day-2             
    for m = 1:1:24
        x = (n-1)*24+48;    %plus 2 days
        P4(m,n) = tempdry1(x+m);
    end
end
P5 = zeros(24,day-2);       %P5 reads actual wet temp.
for n = 1:day-2             
    for m = 1:1:24 
        x = (n-1)*24;
        P5(m,n) = tempwet1(x+m);
    end
end
P6 = zeros(24,day-2);       %P6 reads forcasted wet temp.
for n = 1:day-2
    for m = 1:1:24
        x = (n-1)*24+48;
        P6(m,n) = tempdry1(x+m);
    end
end
P = [P1;P2;P3;P4;P5;P6];

%T
T = zeros(24,day-2);        %training results
for n = 1:day-2
    for m = 1:1:24
        x = (n-1)*24+48;
        T(m,n) = load1(x+m);
    end
end

%%
%read data of year 2016 from excel
data2016 = xlsread('2016_smd_hourly.xls','NEMASSBost');

day2 = 9; %selected day number
datanum2 = day2*24;
indexf = 1:1:datanum2; %1/1/2015-1/31/2015
load2 = data2016(indexf,3);         %Jan. 2016 load
tempdry2 = data2016(indexf,12);     %Jan. 2016 dry temp.
tempwet2 = data2016(indexf,13);     %Jan. 2016 wet temp.
%
%Pf
P11 = zeros(1, day2-2);       %because of D-1 and D does not include so -2.
for n = 1:day2-2              %P11 reads day #
    P11(n) = n+2;            %starting from day 3
end
P22 = zeros(24,day2-2);       %p22 reads hourly load
for n = 1:day2-2
    for m = 1:1:24
        x = (n-1)*24;
        P22(m,n) = load2(x+m);
    end
end
P33 = zeros(24,day2-2);       %p33 reads actual dry temp.
for n = 1:day2-2
    for m = 1:1:24
        x = (n-1)*24;
        P33(m,n) = tempdry2(x+m);
    end
end
P44 = zeros(24,day2-2);       %P44 reads forcatsted dry temp.
for n = 1:day2-2             
    for m = 1:1:24
        x = (n-1)*24+48;    %plus 2 days
        P44(m,n) = tempdry2(x+m);
    end
end
P55 = zeros(24,day2-2);       %P55 reads actual wet temp.
for n = 1:day2-2             
    for m = 1:1:24 
        x = (n-1)*24;
        P5(m,n) = tempwet2(x+m);
    end
end
P66 = zeros(24,day2-2);       %P66 reads forcasted wet temp.
for n = 1:day2-2
    for m = 1:1:24
        x = (n-1)*24+48;
        P66(m,n) = tempdry2(x+m);
    end
end
Pf = [P11;P22;P33;P44;P55;P66];

%Tf
Tf = zeros(24,day2-2);        %forecasting results
for n = 1:day2-2
    for m = 1:1:24
        x = (n-1)*24+48;
        Tf(m,n) = load2(x+m);
    end
end


%training

net = newff(P,T,20);             % TF1=140
net.trainParam.epochs = 20;
net.trainParam.goal = 0.001;

net = train(net,P,T);
%%
%forecasting
Y = sim(net,Pf);

Pfoutput = Y;
%
[a,b] = size(Tf);
X1 = 1:a*b;

Y1 = [];
T1 = [];
for i = 1:b
    T1 = [T1 Tf(:,i)'];%T actual
    Y1 = [Y1 Pfoutput(:,i)'];%Y forecast
end

fr_mape = f_mean(abs(Y1-T1)./T1)
pause;
plot(X1,Y1,X1,T1)