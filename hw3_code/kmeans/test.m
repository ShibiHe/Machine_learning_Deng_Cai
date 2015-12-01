%% Kmeans for 1000 times
load('kmeans_data.mat');
max_dis=0;
min_dis=10000;
max_idx=[];
max_ctrs=[];
max_iter_ctrs=[];
min_idx=[];
min_ctrs=[];
min_iter_ctrs=[];

for i=1:1000
    dis=0;
    [idx,ctrs,iter_ctrs]=kmeans(X,2);
    D=pdist2(X,ctrs);
    for j=1:500
        dis=dis+D(j,idx(j));
    end
    if dis>max_dis
        max_dis=dis;
        max_idx=idx;
        max_ctrs=ctrs;
        max_iter_ctrs=iter_ctrs;
    end
    if dis<min_dis
        min_dis=dis;
        min_idx=idx;
        min_ctrs=ctrs;
        min_iter_ctrs=iter_ctrs;
    end
end
kmeans_plot(X, max_idx, max_ctrs, max_iter_ctrs);
figure;
kmeans_plot(X, min_idx, min_ctrs, min_iter_ctrs);

%% Kmeans for digit_data.mat
load('digit_data.mat');

Ks=[10, 20, 50];
for i=1:3
    [idx, ctrs, iter_ctrs] = kmeans(X, Ks(i));
    figure;
    show_digit(ctrs);
end