%% SOM??????????????????--??????????????

%% ????????????
clc
clear

%% ????????????
% ????????
P=xlsread('Load Temperature','Sheet1');

P1 = P(1:8760,:);
P2 = P(8761:end,:);

%????????????????????????????
First_Year_Load_Data = P(1:8760,2);
Second_Year_Load_Data = P(8761:end,2);

% Each Part represents load data for three respective months
First_Part = First_Year_Load_Data(1:90*24,:);
Second_Part = First_Year_Load_Data((90*24)+1:(181*24),:);
Third_Part = First_Year_Load_Data((181*24)+1:(273*24),:);
Fourth_Part = First_Year_Load_Data((273*24)+1:end,:);

First_Part=First_Part';

% Create SOM Maps


%plotsom(network1.layers{1}.positions)

a=[10 30 50 100 200 500 1000];

% Train SOM net work using passed data


%% ????????????????
% ????????????
num_sheets = 30;
filename = 'Input_Data.xlsx';

A = P1(:,1);        % Hours [1-24]
B = P1(:,2);        % Load Data
C = P1(:,3);        % Temperature
vari = [];
vari2 = [];
for i=1:num_sheets
    sheet = i;
    
    %Algorithm for 24 rows per sheet
    A_data = A(((i-1)*24)+1:i*24, :);
    B_data = B(((i-1)*24)+1:i*24, :);
    C_data = C(((i-1)*24)+1:i*24, :);
    
    Excel_Data = [num2cell(A_data), num2cell(B_data), num2cell(C_data)];
    xlswrite(filename,Excel_Data,sheet);
end

i=1; data1 = []; data2 = []; data3 = []; data4 = [];
rowofj=[]; varOfary=[];

for i=1:num_sheets
    i_str = ['Sheet',num2str(i)];
    t = xlsread('Input_Data.xlsx',i_str);
    Load_Data2 = t(:,2);
    Load_Data2=Load_Data2';
    %%vari(i)= var(Load_Data2);
    sortt = sortrows(t,2);
   max5=[];
    for k=1:5
        max5(k)=sortt(k,1);
    end
    vari1(i)=  var(max5) ;
  
    min5=[];
    for l=1:5
        min5(k)=sortt(25-l,1);
    end
    vari2(i)=  var(min5) ;
   vari(i)= vari1(i)+vari2(i);
    
end
network1=newsom(minmax(vari),[2 2]);% ????6*6?36?????
network1.trainparam.epochs=a(1);
network1=train(network1,vari);
r=sim(network1,vari);
plotsom(network1.IW{1,1},network1.layers{1}.distances)



    rr = vec2ind(r);
    [rows,cols] = size(rr);

    
filename = 'clusteredData1.xlsx';
filename2 = 'clusteredData2.xlsx';
filename3 = 'clusteredData3.xlsx';
filename4 = 'clusteredData4.xlsx';
    
    for j = 1:num_sheets
        %  rowofj= t(j,:);   % Extract third row
        % rowofi=linalg:row(A, i);
        k = rr(j);

        if(k == 1)
          %  data1 = [data1 ; rowofj];
            
        
          i_str = ['Sheet',num2str(j)];
          data1=  xlsread('Input_Data.xlsx',i_str);
        
        A = data1(:,1); B = data1(:,2); C = data1(:,3);
      
          figure(2);
         hold on;
          plot(A,B);
            figure(6);
         hold on;
          plot(A,B,'b');
      
        Excel_Data = [num2cell(A), num2cell(B), num2cell(C)];
        xlswrite(filename,Excel_Data,i_str);
      elseif (k == 2)
           i_str = ['Sheet',num2str(j)];
          data1=  xlsread('Input_Data.xlsx',i_str);
        
            A = data1(:,1); B = data1(:,2); C = data1(:,3);
             figure(3);
              hold on;
             plot(A,B);
             figure(6);
         hold on;
          plot(A,B,'k');
          
      
          Excel_Data = [num2cell(A), num2cell(B), num2cell(C)];
          xlswrite(filename2,Excel_Data,i_str);
    elseif(k == 3)
           i_str = ['Sheet',num2str(j)];
          data1=  xlsread('Input_Data.xlsx',i_str);
        
            A = data1(:,1); B = data1(:,2); C = data1(:,3);
      
           figure(4);
          hold on;
          plot(A,B);
          
            figure(6);
         hold on;
          plot(A,B,'r');
          Excel_Data = [num2cell(A), num2cell(B), num2cell(C)];
          xlswrite(filename3,Excel_Data,i_str);
    elseif(k == 4)
       
       
        
          i_str = ['Sheet',num2str(j)];
          data1=  xlsread('Input_Data.xlsx',i_str);
        
          A = data1(:,1); B = data1(:,2); C = data1(:,3);
         figure(5);
         hold on;
          plot(A,B);
      
          figure(6);
         hold on;
          plot(A,B,'y');
        
          Excel_Data = [num2cell(A), num2cell(B), num2cell(C)];
          xlswrite(filename4,Excel_Data,i_str);
    else
        disp('Hello');
    
    
      end
    
   end
            
 %  time = data1(:,1);        
            
 
            
      




