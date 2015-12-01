img = imread('GF.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));
% YOUR (TWO LINE) CODE HERE
[idx,ctrs,~]=kmeans(fea,2);
fea=ctrs(idx,:);
% imshow(uint8(reshape(fea, size(img))));
imwrite(uint8(reshape(fea, size(img))),'GF_out2.jpg');
