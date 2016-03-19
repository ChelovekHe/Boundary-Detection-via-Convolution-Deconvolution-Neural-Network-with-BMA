fnames = dir('val/*.mat');
delete('val_label_flat.txt');
path = 'val/';
numfids = length(fnames);
pattern = '.mat';
replacement = '';
for K = 1:numfids
  disp(fnames(K).name);
  filename = strcat(path,regexprep(fnames(K).name,pattern,replacement),'.csv');
  groundT = load(strcat(path,fnames(K).name));
  dlmwrite(filename,groundT.groundTruth{1,1}.Boundaries)
end