%Bismillah ArRahmaan ArRaheem
%Automation Code - Text File to Excel

% Reading the text file '239828534_G.txt'

clc;
clear all;

directory = 'E:\1555 497\Data\';

MyFolderInfo = dir(directory);
file_names = [];

number_of_files = length(MyFolderInfo) - 2;

for i = 1:number_of_files
    
    file_names = [file_names ; cellstr(MyFolderInfo(i+2).name)];
    
    fileID = fopen([directory file_names{i}]);
    Data = textscan(fileID,'%d %f %f %f');
    fclose(fileID);

    A = Data{1}; B = Data{2}; C = Data{3}; D = Data{4};

    filename = 'testData.xlsx';

    Excel_Data = [num2cell(A), num2cell(B), num2cell(C), num2cell(D)];
    sheet = i;
    xlswrite(filename,Excel_Data,sheet);
end