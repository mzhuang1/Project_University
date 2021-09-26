% Agg:aggregate power dataset--Agg(index, time, Pa, Qa, Pb, Qb)
% C: ground truth dataset--C(time; index, active appliance ID, phase of appliance (65->Phase A,66->Phase B))
% C_A: ground truth data in Phase A--C_A(time, index, appliance ID)
% K2: aggregate power after median filter--K2(active power)
% w: the start and end time of each contiguous segement (isabove=1/0:show steady/transient event)--w(start time, end time)
% w_mean_time: average power in each contiguous segment--w_mean_time(start time, end time, average active power, average reactive power)
% P_background: background active power->data
% active_events: the active events after filter background level--active_events(turn_on_time,turn_off_time,active_averpower,react_averpower)
% change: the P/Q change during state transitions(only consider steady state)--change(time,P,Q,action); action -1/1:turn off/on

clear all;clc
%% 0.Input data
numfiles=8;
data=cell(1,numfiles);
for i=1:numfiles
    myfilename=sprintf('location_001_matlab_%d.mat',i);
    data{i}=load(myfilename);
end;
% Construct aggregate power dataset: Agg (index, time, Pa, Qa, Pb, Qb)                              
%           ground truth dataset: events_Plugs, events_Cir, events_Env (time, index, active appliance ID, phase of appliance)
Agg=[];events_Plugs=[];events_Cir=[];events_Env=[];events_Unknown=[];events_Unknown;events_total=0;m=0;
for k=1:numfiles
    n=size(data{1,k}.data.Pa,2);
    B=[(data{1,k}.data.t_power)';data{1,k}.data.Pa;data{1,k}.data.Qa;data{1,k}.data.Pb;data{1,k}.data.Qb];
    B=B';
    Agg=[Agg;B];
    %%%%%%%%%  double phase A->65, phase B->66
    if isfield(data{1,k}.data.events,'Plugs')==0
        C1=[];
    else
         C1=[data{1,k}.data.events.Plugs.t;data{1,k}.data.events.Plugs.index+m*ones(size(data{1,k}.data.events.Plugs.index));data{1,k}.data.events.Plugs.MacAddr;double(char(data{1,k}.data.events.Plugs.phase))']';
    end;
    if isfield(data{1,k}.data.events,'CLGT')==0
        C2=[];
    else
        C2=[data{1,k}.data.events.CLGT.t;data{1,k}.data.events.CLGT.index+m*ones(size(data{1,k}.data.events.CLGT.index));data{1,k}.data.events.CLGT.channel;double(char(data{1,k}.data.events.CLGT.phase))']';
    end;
    if isfield(data{1,k}.data.events,'Env')==0
        C3=[];
    else
        C3=[data{1,k}.data.events.Env.t;data{1,k}.data.events.Env.index+m*ones(size(data{1,k}.data.events.Env.index));data{1,k}.data.events.Env.MacAddr;double(char(data{1,k}.data.events.Env.phase))']';
    end;
    if isfield(data{1,k}.data.events,'Unknown')==0
        C4=[];
    else 
        C4=[data{1,k}.data.events.Unknown.t;data{1,k}.data.events.Unknown.index+m*ones(size(data{1,k}.data.events.Unknown.index));data{1,k}.data.events.Unknown.MacAddr;double(char(data{1,k}.data.events.Unknown.phase))']';
    end;
    events_Plugs=[events_Plugs;C1];events_Cir=[events_Cir;C2];events_Env=[events_Env;C3];events_Unknown=[events_Unknown;C4];
    events_total=size(C1,1)+size(C2,1)+size(C3,1)+size(C4,1)+events_total;
    m=m+n;
    %plot summary of data sets
    %fprintf('Data number:%d data size: %d plugs events: %d Circuit events: %d Env events: %d Unknown events: %d.\n',k,size(B,1),size(C1,1),size(C2,1),size(C3,1),size(C4,1));
end;
C=[events_Plugs;events_Cir;events_Env;events_Unknown];
C=sortrows(C,2);                                           % array arrages as index
C_A=C(find(C(:,4)==65),1:3);                               % ground truth of phase A events
index=[1:size(Agg,1)]';
Agg=[index Agg];

%% 1. event detection
%% a) Contiguous and transient portion Detector
%K2=Agg(:,3);
K2=medfilt1(Agg(:,3),300);
%Agg_filt=[Agg(:,1) Agg(:,2) K2];
Thrh=15;
for i=1:size(index)-1
    Kt(i)=K2(i+1)-K2(i);
    if abs(Kt(i))<=Thrh
        Kt(i)=1;
    else
        Kt(i)=0;
    end;
end;
Kt=[0 Kt 0];
edge=diff(Kt);
rise=find(edge==1);
fall=find(edge==-1);
spanWidth=fall-rise;
NumZeros=180;            % The min time defined as contiguous
isabove=1;               % Choose steady(1) or transient(0) portion
if isabove 
    wideEnough=find(spanWidth>=NumZeros);
else
    wideEnough=find(spanWidth<=NumZeros);
end;
start=rise(wideEnough);
ends=fall(wideEnough)-1;
w=[start;ends]';

%% b) Background and Active Segment Labeling
w_temp_P=[];w_temp_Q=[];w_mean=[];w_mean_P=[];w_mean_Q=[];
for i=1:size(w,1)
    w_start=w(i,1);
    w_ends=w(i,2);
    w_duration=w_start:w_ends;
    w_temp_P=Agg(w_duration,3);
    %w_temp=K2(w_duration);
    w_temp_Q=Agg(w_duration,4);
    w_temp_P=mean(w_temp_P);
    w_temp_Q=mean(w_temp_Q);
    w_mean_P=[w_mean_P, w_temp_P];
    w_mean_Q=[w_mean_Q, w_temp_Q];
    w_mean=[w_mean_P;w_mean_Q];
end;
w_mean=w_mean';
w_mean_time=[start' ends' w_mean];               % complete steady-state segment without considering background power

%% K-means
% decide K
km_sum=[];km_idx=[];
for i=1:round(size(w_mean,1)/10)
    [idx_temp,Center,sumd]=kmeans(w_mean,i);
    km_sum=[km_sum;sum(sumd)];
    km_idx=[km_idx,idx_temp];
end;
%plot(1:size(km_sum,1),km_sum);hold on;plot(1:size(km_sum,1),km_sum,'.k','Markersize',12);
n_cluster=find(km_sum==min(km_sum));    % Show the cluster number of K which has the min sum of distance 
[idx,Center]=kmeans(w_mean,n_cluster);
P1=[idx w_mean];
P_background=mean(w_mean((find(idx==min(idx)))',1));

%% c) find detection time
active_events=w_mean_time(find(w_mean_time(:,3)>P_background),1:4);         % steady-state segment considering background power 
background_events=w_mean_time(find(w_mean_time(:,3)<=P_background),1:4);    % steady-state background segment

% Compare detection with ground-truth
index_de=sort([active_events(:,1);active_events(:,2)]);
plot(K2);hold on;plot(C_A(:,2),K2(C_A(:,2)),'.','MarkerSize',12);hold on; plot(index_de,K2(index_de),'.','MarkerSize',12)
legend('aggregate signal','ground truth in Phase A','detection in Phase A');title('event detection results'),xlabel('index'),ylabel('active power (W)')

%% 2. Clusters matching

%% a) calculate the P/Q change in each event 
% w_mean_time represents the whole continguous parts of aggregate signal,active_events represents the events beyond background power, others are
% treated as background contingous part. In other words, in this method only continguous part is considered while transient power is assumed to be 0 in order to aviod inrush current. 
P_ch=[];Q_ch=[];t_temp=[];label=[];
for i=1:size(w_mean_time)-1
    P_ch1=w_mean_time(i+1,3)-w_mean_time(i,3);
    if P_ch1>0
        t1=w_mean_time(i+1,1);
        lb=1;                                 % lb is the indicator of state transition: 1/-1->turn on/off 
    else
        t1=w_mean_time(i,2);
        lb=-1;
    end;
    Q_ch1=w_mean_time(i+1,4)-w_mean_time(i,4);
    P_ch=[P_ch, P_ch1];
    Q_ch=[Q_ch, Q_ch1];
    t_temp=[t_temp,t1];
    label=[label,lb];
end;
change=[t_temp',P_ch',Q_ch',label'];

events_on=change(find(change(:,4)==1),1:3);
events_off=change(find(change(:,4)==-1),1:3);

%% b) agglomerative hierarchical cluster
%%   1) turn on detection
Y=pdist(events_on(:,2:3),'mahalanobis');
%squareform(Y);
Z=linkage(Y,'complete');                        
%dendrogram(Z,150);
figure(1);plot(Z(:,3));title('Tree-level distance');xlabel('Tree combine time');ylabel('distance');

dis=[];
for i=1:size(Z,1)-1                           % the row size of Z indicates the tree combination times
    Z_diff=(Z(i+1,3)-Z(i,3))/Z(i,3);
    dis=[dis, Z_diff];                        % dis: the difference of cluster distance
end;
figure(2);plot(dis);title('Tree-level distance variation');xlabel('Tree combine time');ylabel('distance variation');
[var_thrh,dis_index]=max(dis);

I=inconsistent(Z);
dis_thrh=max(I(dis_index,4),I(dis_index+1,4));
T=cluster(Z,'cutoff',dis_thrh);
event_clusters=[events_on,T];
%%
turn_on_events=event_clusters(find(event_clusters(:,2)>0),:);
turn_off_events=event_clusters(find(event_clusters(:,2)<0),:);
