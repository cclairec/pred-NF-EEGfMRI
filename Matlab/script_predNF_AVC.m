%learn_runs={'run-01'; 'run-02'; 'run-03'};
patients={'sub-xp216'};
sessions={'task-2dNF'};
learn_runs={'run-03'};
test_runs={'run-03'};
for p=1:length(patients)
    for s=1:length(sessions)
        for l=1:length(learn_runs)
            for t=1:length(test_runs)
                res_path=['C:/Users/cpinte/Documents/Results/Sujets_sains/Res_', patients{p} ,'_' ,sessions{s}, '_', learn_runs{l}, '_', test_runs{t}, '/Res_', patients{p} ,'_' ,sessions{s}, '_', learn_runs{l}, '_', test_runs{t}, '.mat'];
                Res=pred_NF_from_eeg_fmri_1model_AVC(patients{p}, sessions{s}, learn_runs{l}, test_runs{t},'fmri');              
                save(res_path,'Res');
            end
        end
    end
end