%% Load Res
load('C:/Users/cpinte/Documents/Results/Res_P002_sS1s1_lNF2_tNF1/Res_P002_sS1s1_lNF2_tNF1.mat'); % Res matrix
ResPath=['C:/Users/cpinte/Documents/Results/Res_P002_sS1s1_lNF2_tNF1/']; % path for saving figures
 
%% Plot gtruth with prediction
plotPredNF(Res);

%% Save Figure
saveas(gcf,['',ResPath,'plot_gtruth_NFpred.png'])