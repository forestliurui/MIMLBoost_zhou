clear;

dbstop if error;
%addpath('~/MIMLBoost/LIBSVM_weighted');
addpath('~/MIMLBoost/libsvm-mat-2.86-1');
addpath('~/MIMLBoost/auxiliary');
load('miml_data.mat');
num_bags=length(bags);

rand_sample=randperm(num_bags);
rand_sample_training=rand_sample(1:floor(num_bags*0.8));
rand_sample_test=rand_sample(floor(num_bags*0.8)+1:end);
train_bags=bags(rand_sample_training,1);
train_target=targets(:,rand_sample_training);



rounds=15;


svm.type='RBF';
svm.para=2;
cost=1;
[classifiers,c_values,Iter_train,tr_time]=MIMLBoost_train(train_bags,train_target,rounds,svm,cost);

test_bags=bags(rand_sample_test,1);
test_target=targets(:,rand_sample_test);


Iter=1;
[HammingLoss1,RankingLoss1,OneError1,Coverage1,Average_Precision1,Outputs1,Pre_Labels1,te_time1]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter);
[HammingLoss1_train,RankingLoss1_train,OneError1_train,Coverage1_train,Average_Precision1_train,Outputs1_train,Pre_Labels1_train,te_time1_train]=MIMLBoost_test(train_bags,train_target,classifiers,c_values,Iter);


for i=1:3
    Iter=i*5;
    [HammingLoss{i},RankingLoss{i},OneError{i},Coverage{i},Average_Precision{i},Outputs{i},Pre_Labels{i},te_time{i}]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter);
    [HammingLoss_train{i},RankingLoss_train{i},OneError_train{i},Coverage_train{i},Average_Precision_train{i},Outputs_train{i},Pre_Labels_train{i},te_time_train{i}]=MIMLBoost_test(train_bags,train_target,classifiers,c_values,Iter);


end
