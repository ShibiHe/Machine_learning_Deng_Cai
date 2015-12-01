function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));
imshow(uint8(img_r));
X=[];
% YOUR CODE HERE
for i=1:size(img_r,1)
    for j=1:size(img_r,2)
        if img_r(i,j)~=255
            X=[X;[i,j,img_r(i,j)]];
        end
    end
end
XProject=X(:,1:2);
m=mean(XProject);
XProject=XProject-repmat(m,size(XProject,1),1);
[P,~]=pca(XProject);
XProject=XProject*P;
m=[abs(min(XProject(:,1)))+1, abs(min(XProject(:,2)))+1];
XProject=XProject+repmat(m,size(XProject,1),1);
XProject=[XProject,X(:,3)];
img=zeros(size(img_r));
for i=1:size(XProject,1)
    img(ceil(XProject(i,1)),ceil(XProject(i,2)))=XProject(i,3);
end
figure;
imshow(uint8(img));
end