clear all
clc

    %% ��������
[MIXtrain,MIXtest,DATAtrain,DATAtest]=online_dataproduce();
p_train = MIXtrain';
t_train = DATAtrain';

p_test = MIXtest'; 
t_test = DATAtest';



%% ���ݹ�һ��

% ���뼯
[pn_train,inputps] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',inputps);
pn_test = pn_test';
% �����
[tn_train,outputps] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',outputps);
tn_test = tn_test';
%% SVMģ�ʹ���/ѵ��

[bestCVmse,bestc,bestg,pso_option]=psoSVMcgForRegress(tn_train,pn_train);

% ����/ѵ��SVM  
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];

model = svmtrain(tn_train,pn_train,cmd);

%% SVM����Ԥ��
[Predict_1,error_1,tt1] = svmpredict(tn_train,pn_train,model);
[Predict_2,error_2,tt2] = svmpredict(tn_test,pn_test,model);
% ����һ��
predict_1 = mapminmax('reverse',Predict_1,outputps);
predict_2 = mapminmax('reverse',Predict_2,outputps);
% ����Ա�
result_1 = [t_train predict_1];
result_2 = [t_test predict_2];

%% ��ͼ
figure(4)
plot(1:length(t_train),t_train,'r-*',1:length(t_train),predict_1,'b:o')
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('�������')
ylabel('�ź�')
string_1 = {'ѵ����Ԥ�����Ա�';
           ['mse = ' num2str(error_1(2)) ' R^2 = ' num2str(error_1(3))]};
title(string_1)
figure(5)
plot(1:length(t_test),predict_2,'r-*',1:length(t_test),t_test,'b:o')
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('�������')
ylabel('�ź�')
string_2 = {'���Լ�Ԥ�����Ա�';
           ['mse = ' num2str(error_2(2)) ' R^2 = ' num2str(error_2(3))]};
title(string_2)


k=0;
for i=1:178
    if abs(t_test(i)-predict_2(i))<1
        k=k+1;
    end
end
k

