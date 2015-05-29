function [inst_num,instances,inst_labels ]=construction_basic(train_bags, train_target)

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