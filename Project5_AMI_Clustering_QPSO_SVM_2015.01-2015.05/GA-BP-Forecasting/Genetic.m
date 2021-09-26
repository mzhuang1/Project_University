% ��ջ�������
clc
clear
% 
%% ����ṹ����
%��ȡ����
data=xlsread('focus2.xlsx');

%�ڵ����
inputnum=2;
hiddennum=11;
outputnum=1;
data_train=data(79:3378,1:3); 
data_test=data(1:78,1:3);

input_train=data_train(:,2:3)';
output_train=data_train(:,1)';

input_test=data_test(:,2:3)';
output_test=data_test(:,1)';
%%[inputn,mininput,maxinput,outputn,minoutput,maxoutput]=premnmx(input_train,output_train); %��p��t�����ֱ�׼��Ԥ���� 
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
net=newff(minmax(inputn),[11,1],{'tansig','purelin'},'trainlm');

%ѵ�����ݺ�Ԥ������
%%input_train=input(1:1900,:)';
%%input_test=input(1901:2000,:)';
%%output_train=output(1:1900)';
%%output_test=output(1901:2000)';

%ѡ����������������ݹ�һ��
%%[inputn,inputps]=mapminmax(input_train);
%%[outputn,outputps]=mapminmax(output_train);

%��������
%%net=newff(inputn,outputn,hiddennum);

%% �Ŵ��㷨������ʼ��
maxgen=100;                         %��������������������
sizepop=50;                        %��Ⱥ��ģ
pcross=0.9;                       %�������ѡ��0��1֮��
pmutation=0.1;                    %�������ѡ��0��1֮��

%�ڵ�����
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);        
bound=[-5*ones(numsum,1) 5*ones(numsum,1)];    %���ݷ�Χ

%------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��

%��ʼ����Ⱥ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary��grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    x=individuals.chrom(i,:);
    %������Ӧ��
    [individuals.fitness(i),hh]=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn); %Ⱦɫ�����Ӧ��
end

%����õ�Ⱦɫ��
[bestfitness, bestindex]=max(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
% ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
trace=[avgfitness bestfitness]; 
 
%% ���������ѳ�ʼ��ֵ��Ȩֵ
% ������ʼ
for i=1:maxgen
    
    % ѡ��
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %����
        [individuals.fitness(j),cc(j)]=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    sum_squreerror(i)=sum(cc);
    
  %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=max(individuals.fitness);
    [worestfitness,worestindex]=min(individuals.fitness);
    % ������һ�ν�������õ�Ⱦɫ��
    if bestfitness<newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    aa(i)=avgfitness;
    bb(i)=bestfitness;
    trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��

end
%% �Ŵ��㷨������� 
 figure(1)
%%[r, c]=size(trace);
%%plot([1:r]',trace(:,2),'b--');
%%plot(aa,'b--');
%%hold on
%% �Ŵ��㷨������� 
plot(aa,'b+');
hold on
plot(bb,'r-');
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
legend('ƽ����Ӧ��','�����Ӧ��');
figure(2)
plot(1./aa,'b+');
hold on
plot(1./bb,'r-');
xlabel('��������');ylabel('GA-BP���ƽ����');
legend('ƽ�����ƽ����','��С���ƽ����');
x=bestchrom;

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=1000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.001;

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);


Yn=test_simu;                 
figure(3)                        %��ͼ
plot(Yn,'r*-')                %����Ԥ��ֵ����
hold on                       %������ͼ
plot(output_test,'bo')                  %ʵ��ֵ����
legend('Ԥ��ֵ','ʵ��ֵ')      %ͼ��

xx=0;

    for j=1:1:78
    if abs(output_test(j)-Yn(j))<2
      xx=xx+1;        
    end   
end
k=xx/78;
disp(k);


