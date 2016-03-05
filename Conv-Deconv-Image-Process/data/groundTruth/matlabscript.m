fnames = dir('train/*.mat');
delete('train_label_flat.txt');
path = 'train/';
numfids = length(fnames);
fid = fopen('train_label_flat.txt','wt');
for K = 1:numfids
  groundT = load(strcat(path,fnames(K).name));
  grayimage_str = sprintf('%i,',groundT.groundTruth{1,1}.Boundaries);
  grayimage_str = grayimage_str(1:end-1);
  fprintf(fid, grayimage_str);
end
fclose(fid);