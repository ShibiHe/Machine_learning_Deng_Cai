load('TDT2_data', 'fea', 'gnd');
N=100;
options = [];
options.NeighborMode = 'KNN';
options.k=10;
options.WeightMode = 'Binary';
W = constructW(fea,options);
% YOUR CODE HERE
accuracy1=0;
accuracy2=0;
NMI1=0;
NMI2=0;
for i=1:N
    idx=spectral(W,2);
    res=bestMap(gnd,idx);
    accuracy1=length(find(gnd==res))/length(gnd)+accuracy1;
    NMI1=NMI1+MutualInfo(gnd,res);

    idx=litekmeans(fea,2);
    res=bestMap(gnd,idx);
    accuracy2=length(find(gnd==res))/length(gnd)+accuracy2;
    NMI2=NMI2+MutualInfo(gnd,res);
end
fprintf('Spectral Clustering:Accuracy=%f, normalized mutual information=%f\n',accuracy1/N,NMI1/N);
fprintf('Litekmeans:Accuracy=%f, normalized mutual information=%f\n',accuracy2/N,NMI2/N);