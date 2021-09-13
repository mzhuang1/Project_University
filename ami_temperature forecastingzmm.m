% Load data and trainning network
P=xlsread('temperature_AMI_ANN.xls','P');
T=xlsread('temperature_AMI_ANN.xls','T');
P0=xlsread('temperature_AMI_ANN.xls','P0');
T0=xlsread('temperature_AMI_ANN.xls','T0');

% creat and train network
%NodeNum1 = 10; %number of nodes of the second layer in hidden layer
%NodeNum1 = 20; %number of nodes of the second layer in hidden layer
[pn,minp,maxp,tn,mint,maxt]=premnmx(P,T);%normalizaiton of data
TF1 = 'tansig'; 
TF2 = 'tansig'; 
TF3 = 'tansig';
net=newff(minmax(pn),[90,24],{TF1 TF2 TF3},'traingdx');
net.trainParam.epochs=5000;
net.trainParam.show=50;
net.trainParam.epochs=5000; 
net.trainParam.goal=1e-5;
net.trainParam.lr=0.001;
net=train(net,pn,tn);

% data forecasting;
p2n=tramnmx(P0,minp,maxp);%Normalization of test data
Y0=sim(net,p2n);
[a]= postmnmx(Y0,mint,maxt); %Reverse normalization of Data

%data analysis
[nrow,ncol]=size(T0);
X1=[1:nrow*ncol];
Y1=[];
T1=[];
for i=1:ncol;
    T1=[T1 T0(:,i)'];
    Y1=[Y1 Y0(:,i)'];
end
AMI_Result = perform(net,Y1,T1) % calculate statstics
a=length(Y0); %The length of vector 
error=T0'-a; %error vector
figure;
hist(error,length(error)-1)
legend('Foracasting errors temperature')
xlabel('Error in (^{o}C)')
ylabel('Number of occurances')
title('Forecasting change of error')
pause;
%plot(X1,Y1,X1,T1);