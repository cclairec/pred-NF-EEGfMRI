%patients={'P002';'P003'; 'P004'; 'P015'; 'P017'};
%sessions={'S1s1'; 'S1s2'; 'S2s1'; 'S3s1'};
%learn_runs={'NF1'; 'NF2'; 'NF3'};
%test_runs=learn_runs;

patients={'P002'};
sessions={'S1s1'};
learn_runs={'NF2'};
test_runs={'NF2'};
for p=1:length(patients)
    for s=1:length(sessions)
        for l=1:length(learn_runs)
            for t=1:length(test_runs)
                res_path=['C:/Users/cpinte/Documents/Results/Res_', patients{p} ,'_s' ,sessions{s}, '_l', learn_runs{l}, '_t', test_runs{t}, '/Res_', patients{p} ,'_s' ,sessions{s}, '_l', learn_runs{l}, '_t', test_runs{t}, '.mat'];
                Res=pred_NF_from_eeg_fmri_1model_AVC(patients{p}, sessions{s}, learn_runs{l}, test_runs{t},'fmri');              
                save(res_path,'Res');
            end
        end
    end
end