clear;
load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE

% 1. Feature preprocessing
% 2. Run PCA
% 3. Visualize eigenface
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
% 6. Recover face images form low dimensional space, visualize them

X=fea_Train;
show_face(X);

centralize=mean(X);
X=X-repmat(centralize,size(X,1),1);
figure;
show_face(X);

[u,s,v]=mySVD(X);
figure;
show_face(v');
dims=[8,16,32,64,128];
for i=1:size(dims,2)
    s_re=s(1:dims(i),1:dims(i));
    u_re=u(:,1:dims(i));
    v_re=v(:,1:dims(i));
    X_train=X*v_re;
    X_test=(fea_Test-repmat(mean(fea_Test),size(fea_Test,1),1))*v_re;
    y=knn(X_test',X_train',gnd_Train',1);
    fprintf('Knn k=1, The error rate of reduced dimensionality=%d is %f.\n',...
    dims(i),length(find(y' ~= gnd_Test))/length(y));
end

%%
X=fea_Train;
show_face(X);

centralize=mean(X);
X=X-repmat(centralize,size(X,1),1);
% figure;
% show_face(X);
dims=[8,16,32,64,128];
for i=1:length(dims)
    MAX_EIGENS=dims(i);
    s_new=s(1:MAX_EIGENS,1:MAX_EIGENS);
    u_new=u(:,1:MAX_EIGENS);
    v_new=v(:,1:MAX_EIGENS);
    recoverX=u_new*s_new*v_new';
    recoverX=recoverX+repmat(centralize,size(X,1),1);
    figure;
    show_face(recoverX);
end

