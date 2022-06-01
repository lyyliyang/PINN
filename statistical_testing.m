clear
path='.\data';
dro_path='.\dropout data';
files =dir(path);
folder_nums = size(files,1);
folder_names = { };
% 文件中子文件夹的名称是从第3位开始的，这里需要注意
for i=3:folder_nums
    folder_names{i-2} = files(i,1).name;
end
name={'boston','concrete','energy','kin8nm','naval','power','protein','wine','yacht','year'};
metric={'RMSE','PLL','time'};
p_all=[];h_all=[];
m_all=[];
s_all=[];t_all=[];
for i=1:size(metric,2)
    for j=1:size(name,2)
        for k=1:size(folder_names,2)

            if ~isempty(strfind(folder_names{k},name{j})) & ~isempty(strfind(folder_names{k},metric{i}))
                if strcmp(metric{i},'time')
                    data=load([path,'\',folder_names{k}]);  
                    t=mean(data);
                    t_all=[t_all,t];
                else
                    
                    data=load([path,'\',folder_names{k}]);  
                    data_dro=load([dro_path,'\',folder_names{k}]);
                    m=mean(data);
                    s=std(data);
                    m_all=[m_all,m];
                    s_all=[s_all,s];
                    [h,p]  = ttest2(data,data_dro);
                    p_all=[p_all,p];h_all=[h_all,h]
                end
                break

            end
        end
    end
end



