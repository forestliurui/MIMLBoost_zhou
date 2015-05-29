clear;

dbstop if error;
%addpath('~/MIMLBoost/LIBSVM_weighted');
addpath('~/MIMLBoost/libsvm-mat-2.86-1');
addpath('~/MIMLBoost/auxiliary');
load('miml_data.mat');
num_bags=length(bags);

%rand_sample=randperm(num_bags);
%rand_sample_training=rand_sample(1:floor(num_bags*0.8));
%rand_sample_test=rand_sample(floor(num_bags*0.8)+1:end);
%train_bags=bags(rand_sample_training,1);
%train_target=targets(:,rand_sample_training);


rounds=15;


svm.type='RBF';

for trainset_index=0:9
svm.para=2;
cost=1;
    trainset_name=['natural_scene.fold_000',num2str(trainset_index),'_of_0010.train' ];
    testset_name=['natural_scene.fold_000',num2str(trainset_index),'_of_0010.test' ];
    fprintf(1,'Training set: %s\n',trainset_name);
    fprintf(1,'Test set: %s\n',testset_name);
    [classifiers,c_values,Iter_train,tr_time]=MIMLBoost_train(bags,targets,trainset_name  ,rounds,svm,cost);



    [temp_prefix, bag_test_index]=textread(['folds/',testset_name,'.view'],'%s%d','delimiter',',');
     test_target=targets(:,bag_test_index);
     test_bags=bags(bag_test_index,1);
    
    Iter=1;
    [HammingLoss1(trainset_index+1),RankingLoss1(trainset_index+1),OneError1(trainset_index+1),Coverage1(trainset_index+1),Average_Precision1(trainset_index+1),Outputs1,Pre_Labels1,te_time1]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter);
    for Iter=1:rounds
        i=Iter;
        [HammingLoss(trainset_index+1,i),RankingLoss(rainset_index+1,i),OneError(trainset_index+1,i),Coverage(trainset_index+1,i),Average_Precision(trainset_index+1,i),Outputs,Pre_Labels,te_time]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter);
    end
    save(['rounds_',num2str(rounds),'_trainset_index_',num2str(trainset_index),'.mat']);
end

%[HammingLoss1_train,RankingLoss1_train,OneError1_train,Coverage1_train,Average_Precision1_train,Outputs1_train,Pre_Labels1_train,te_time1_train]=MIMLBoost_test(train_bags,train_target,classifiers,c_values,Iter);


