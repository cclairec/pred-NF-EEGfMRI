%% Load Res
prompt='Input name (ex: Res_P002_sS1s1_lNF1_tNF2) :';
path=input(prompt,'s');
  
if strcmp(path,'')
    load('C:/Users/cpinte/Documents/Results/Best_Results/Res_P002_sS1s1_lNF1_tNF1/Res_P002_sS1s1_lNF1_tNF1.mat'); % Res matrix
    ResPath=['C:/Users/cpinte/Documents/Results/Best_Results/Res_P002_sS1s1_lNF1_tNF1/']; % path for saving figures
else
    load(['C:/Users/cpinte/Documents/Results/Best_Results/',path,'/',path,'.mat']); % Res matrix
    ResPath=['C:/Users/cpinte/Documents/Results/Best_Results/',path,'/']; % path for saving figures
end
%% Plot gtruth with prediction
plotPredNF(Res);

%% Save the correlation value
[correlation_value, correlation_value_2] = correlationPredNF(Res);
disp(correlation_value)
disp(correlation_value_2)

writematrix(correlation_value, ['',ResPath,'correlation_value.txt'])

% Case lNF = tNF
if ~isnan(correlation_value_2)
    writematrix(correlation_value_2, ['',ResPath,'correlation_value_2.txt'])
end

%% Save Figure
saveas(gcf,['',ResPath,'plot_gtruth_NFpred.png'])