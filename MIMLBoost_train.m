function [classifiers,c_values,Iter,tr_time]=MIMLBoost_train(bags,targets,trainset_name,rounds,svm,cost)
%MIMLBoost_train implements the training procedure of MIMLBOOST as shown in [1].
%
%N.B.: MIMLBoost employs the Matlab version of Libsvm [2] (available at http://sourceforge.net/projects/svm/) as the base learners
%
%    Syntax
%
%       function [classifiers,c_values,Iter,tr_time]=MIMLBoost_train(train_bags,train_target,rounds,svm,cost)
%
%    Description
%
%       MIMLBoost_train takes,
%           train_bags     - An Mx1 cell, the jth instance of the ith training bag is stored in train_bags{i,1}(j,:)
%           train_target   - A QxM array, if the ith training bag belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           rounds         - The number of boosting rounds used
%           svm            - svm.type gives the type of svm used in training, which can take the value of 'RBF', 'Poly' or 'Linear'; svm.para gives the corresponding parameters used for the svm:
%                            1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-Gamma*|x1-x2|^2) for two vectors x1 and x2
%                            2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x1,x2>+coefficient)^degree.
%                            3) if svm.type is 'Linear', then svm.para is [].
%           cost           - The cost parameter used for the base svm classifier
%      and returns,
%           classifiers    - An Iterx1 structure, where parameters of the ith trained base learner is stored in classifiers(i)
%           c_values       - A 1xIter array, where the ith c value (as defined in [1]) is stored in c_values(i)
%           Iter           - The number of trained base learners
%           tr_time        - The training time
%
% [1] Z.-H. Zhou and M.-L. Zhang. Multi-instance multi-label learning with application to scene classification. In: Advances in Neural Information Processing Systems 19 (NIPS'06) (Vancouver, Canada), B. Sch?0?2lkopf, J. Platt, and T. Hofmann, eds. Cambridge, MA: MIT Press, 2007.
% [2] C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector machines, Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan, Technical Report, 2001.

%Preparing data
    
     start_time=cputime;
     
     [temp_prefix, bag_train_index]=textread(['folds/',trainset_name,'.view'],'%s%d','delimiter',',');
     train_target=targets(:,bag_train_index);
     train_bags=bags(bag_train_index,1);
     
     
     
     
     [num_class,num_bags]=size(train_target);
     
     Label=cell(num_bags,1);
     not_Label=cell(num_bags,1);
     Label_size=zeros(1,num_bags);
     for i=1:num_bags
         temp=train_target(:,i);
         Label_size(1,i)=sum(temp==ones(num_class,1));
         for j=1:num_class
             if(temp(j)==1)
                 Label{i,1}=[Label{i,1},j];
             else
                 not_Label{i,1}=[not_Label{i,1},j];
             end
         end
     end
    
    
     inst_num=zeros(1,num_bags*num_class);
     
     num_inst=0;
     for i=1:num_bags
         temp_bag=train_bags{i,1};
         tempsize=size(temp_bag,1);
         num_inst=num_inst+tempsize*num_class;
     end
     Dim=length(train_bags{1,1}(1,:))+1;
     instances=zeros(Dim,num_inst);
     inst_labels=zeros(1,num_inst);
     
     if(Dim<=50)
         ER=1;
     else
         ER=log(Dim)/log(50);
     end

     for i=1:num_bags
         temp_bag=train_bags{i,1};
         tempsize=size(temp_bag,1);
         for j=1:num_class
             inst_num(1,(i-1)*num_class+j)=tempsize;
             tempvec=[temp_bag,ER*(j-1)/(num_class-1)*ones(tempsize,1)];
             low=sum(inst_num(1:((i-1)*num_class+j-1)))+1;
             high=sum(inst_num(1:((i-1)*num_class+j)));
             instances(:,low:high)=tempvec';
             if(ismember(j,Label{i,1}))
                 inst_labels(1,low:high)=ones(1,tempsize);
             else
                 inst_labels(1,low:high)=-ones(1,tempsize);
             end
         end
     end
     
%Initializing weights     
     Weights=(1/(num_bags*num_class))*ones(1,num_bags*num_class);
     num_inst=size(instances,2);

%Begin boosting
     Iter=0;
     for round=1:rounds
         
         disp(strcat('Boosting round: ',num2str(round)));
         
         %reweighting instances         
         vec1=zeros(1,num_inst);
         vec2=zeros(1,num_inst);
         for i=1:num_bags
             for j=1:num_class
                 low=sum(inst_num(1:((i-1)*num_class+j-1)))+1;
                 high=sum(inst_num(1:((i-1)*num_class+j)));
                 vec1(1,low:high)=Weights(1,(i-1)*num_class+j)*ones(1,high-low+1);
                 vec2(1,low:high)=inst_num(1,(i-1)*num_class+j)*ones(1,high-low+1);
             end
         end
         inst_weight=vec1./vec2;
         
         
         
        for inner_CV_index=0:4 %inner loop used for tuning the parameters
            inner_trainset_name=[trainset_name,'.fold_000',num2str(inner_CV_index), '_of_0005.train'];
            inner_testset_name=[trainset_name,'.fold_000',num2str(inner_CV_index), '_of_0005.test'];
        
            
            
            [temp_prefix, inner_train_bag_index]=textread(['folds/',inner_trainset_name,'.view'],'%s%d','delimiter',',');
            inner_trainset_bag=bags(inner_train_bag_index,1); %bag index starts from 1 to 2000
            inner_trainset_target=targets(:,inner_train_bag_index);
        
       %     for ii=1:length(inner_train_bag_index)
                
             %   inner_train_inst_weight(1,ii) = inst_weight(find(bag_train_index==inner_train_bag_index(ii)));
            %end
           
	    inner_train_inst_weight=[];
            for ii=1:length(inner_train_bag_index)
                i=find(bag_train_index==inner_train_bag_index(ii));
                for j=1:num_class
                    low=sum(inst_num(1:((i-1)*num_class+j-1)))+1;
                    high=sum(inst_num(1:((i-1)*num_class+j)));
                    inner_train_inst_weight=[inner_train_inst_weight,inst_weight(low:high)];
                end
%                 inner_test_inst_weight(1,ii) = inst_weight(find(bag_train_index==inner_test_bag_index(ii)));
            end
 
            [temp_prefix, inner_test_bag_index]=textread(['folds/',inner_testset_name,'.view'],'%s%d','delimiter',',');
            inner_testset_bag=bags(inner_test_bag_index,1); %bag index starts from 1 to 2000
            inner_testset_target=targets(:,inner_test_bag_index);
        
            %for ii=1:length(inner_test_bag_index)
                
             %   inner_test_inst_weight(1,ii) = inst_weight(find(bag_train_index==inner_test_bag_index(ii)));
            %end
                  
	    inner_test_inst_weight=[];
            for ii=1:length(inner_test_bag_index)
                i=find(bag_train_index==inner_test_bag_index(ii));
                for j=1:num_class
                    low=sum(inst_num(1:((i-1)*num_class+j-1)))+1;
                    high=sum(inst_num(1:((i-1)*num_class+j)));
                    inner_test_inst_weight=[inner_test_inst_weight,inst_weight(low:high)];
                end
%                 inner_test_inst_weight(1,ii) = inst_weight(find(bag_train_index==inner_test_bag_index(ii)));
            end


            
            %% construct relevant instances     
            [inner_num_class,inner_num_bags]=size(inner_trainset_target);
     
            [inner_inst_num,inner_instances,inner_inst_labels ]=construction_basic(inner_trainset_bag, inner_trainset_target);
            %%
           
            
            %build base learner using resampling
            rand('twister',sum(100*clock));
            inner_index=randsample(sum(inner_inst_num),sum(inner_inst_num),true,inner_train_inst_weight)';
            inner_Samples=instances(:,inner_index);
            inner_Labels=inst_labels(:,inner_index);  
        
            
            %try different combination of parameters
            for para_index=1:15
                fprintf(1,'Inner Training set: %s\n',inner_trainset_name);
                fprintf(1,'Inner Test set: %s\n',inner_testset_name);
                fprintf(1,'Inner CV parameter index %d/15\n', para_index)
                C_list(para_index)=exp(8*rand()-3);
                gamma_list(para_index)=exp(7*rand()-6);
                t=2;
                str=['-t ',num2str(t),' -g ',num2str(gamma_list(para_index)),' -c ',num2str(C_list(para_index))];
                model=svmtrain(inner_Labels',inner_Samples',str);
                
                inner_classifiers(1).model=model;
                inner_c_values(1)=1;
                
                [inner_HammingLoss(para_index),inner_RankingLoss(para_index),inner_OneError(para_index),inner_Coverage(para_index),inner_Average_Precision(inner_CV_index+1,para_index),inner_Outputs,inner_Pre_Labels,inner_te_time]=MIMLBoost_test(inner_testset_bag,inner_testset_target,inner_classifiers,inner_c_values,1);
            end
          
     end
         [inner_AP_max_v, inner_AP_max_index]=max(mean(inner_Average_Precision,1));
         
         svm.para=gamma_list(inner_AP_max_index);
         cost=C_list(inner_AP_max_index);
         
         %build base learner using resampling
         rand('twister',sum(100*clock));
         index=randsample(sum(inst_num),sum(inst_num),true,inst_weight)';
         Samples=instances(:,index);
         Labels=inst_labels(:,index);         
         
         if(strcmp(svm.type,'RBF'))
             t=2;
             gamma=svm.para;
             str=['-t ',num2str(t),' -g ',num2str(gamma),' -c ',num2str(cost)];
         else
             if(strcmp(svm.type,'Poly'))
                 t=1;
                 gamma=svm.para(1);
                 coefficient=svm.para(2);
                 degree=svm.para(3);
                 str=['-t ',num2str(t),' -d ',num2str(degree),' -g ',num2str(gamma),' -r ',num2str(coefficient),' -c ',num2str(cost)];
             else
                 t=0;                 
                 str=['-t ',num2str(t),' -c ',num2str(cost)];
             end
         end
         display('begin training...');
%          model=svmtrain(Labels',Samples',str);
         model=svmtrain(ones(length(Labels)), Labels',Samples',str);
         errors=[];
         for i=1:num_bags
             for j=1:num_class
                 low=sum(inst_num(1:((i-1)*num_class+j-1)))+1;
                 high=sum(inst_num(1:((i-1)*num_class+j)));
                 Samples=instances(:,low:high);
                 Labels=inst_labels(:,low:high);
%                  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias, Parameters, nSV, nLabel);
                 [predict_label, accuracy, dec_values] = svmpredict(Labels', Samples', model);
                 errors=[errors,1-(accuracy(1)/100)];
             end
         end
         
         if(sum(errors<=0.5)==num_bags*num_class)
             break;
         end
         
         %find c value
         tic;
         c_value=fminunc(@MIML_error,0,optimset('Gradobj','on'),Weights,errors);
         toc;
         
         if(c_value<=0)
             break;
         end
         
         %Update weights
         for i=1:num_bags*num_class
             Weights(i)=Weights(i)*exp((2*errors(i)-1)*c_value);
         end
         Weights=Weights/sum(Weights);         

         classifiers(round).model=model;
         classifiers(round).str=str;
         c_values(round)=c_value;
         
         Iter=Iter+1;
     end
     
     tr_time=cputime-start_time;
