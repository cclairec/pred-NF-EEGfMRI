function [] = plotPredNF(Res)
%PLOTPREDNF Plot the NF prediction with the ground truth
%   Res : Res object, output of pref_NF_from_eeg_fmri_1model_AVC.m

if size(Res.NF_estimated_fMRI) == size(Res.NF_fMRI_test)
    NF_estimated_fMRI = Res.NF_estimated_fMRI;
    NF_fMRI_test = Res.NF_fMRI_test;
    NF_EEG_test = Res.NF_EEG_test;
else
    half = size(Res.NF_estimated_fMRI,2);
    NF_estimated_fMRI(1:half) = Res.NF_estimated_fMRI;
    NF_estimated_fMRI(half+1:half+half) = Res.NF_estimated_fMRI;
    NF_fMRI_test = Res.NF_fMRI_test;
    NF_EEG_test = Res.NF_EEG_test;
end

blocsize=160;nbloc=8;
minval = -300;
maxval = 300;

figure;
% fMRI pred VS fMRI gtruth
subplot(2,2,1);
hold off;
for i=1:nbloc % plot Task bands
    x = [blocsize*(i-1)+blocsize/2+1:1:blocsize*i+1];
    y = maxval*ones(size(x));
    basevalue = minval;
    ha = area(x, y, basevalue);
    set(ha(1), 'FaceColor', [0.93 0.93 0.93]);
    set(ha, 'LineStyle', 'none');
    set(gca, 'Layer', 'top');
    hold on;
end
hold on;
plot1=plot(zscore(NF_estimated_fMRI).*50, 'LineWidth', 1); plot1.Color(4)=0.7;
plot2=plot(zscore(NF_fMRI_test).*50, 'LineWidth', 1); plot2.Color(4)=0.7;
title('fMRI pred (Red) vs fMRI gtruth (Yellow)');
correlation_value = corr2(zscore(NF_estimated_fMRI),zscore(NF_fMRI_test));
legend(sprintf('Correlation = %0.3f',correlation_value))

% fMRI pred + EEG gtruth VS fMRI gtruth + EEG gtruth
subplot(2,2,2);
hold off;
for i=1:nbloc % plot Task bands
    x = [blocsize*(i-1)+blocsize/2+1:1:blocsize*i+1];
    y = maxval*ones(size(x));
    basevalue = minval;
    ha = area(x, y, basevalue);
    set(ha(1), 'FaceColor', [0.93 0.93 0.93]);
    set(ha, 'LineStyle', 'none');
    set(gca, 'Layer', 'top');
    hold on;
end
hold on;
plot3=plot((zscore(NF_estimated_fMRI)+zscore(NF_EEG_test)).*50, 'LineWidth', 1); plot3.Color(4)=0.7;
plot4=plot((zscore(NF_fMRI_test)+zscore(NF_EEG_test)).*50, 'LineWidth', 1); plot4.Color(4)=0.7;
title('fMRI pred + EEG gtruth (Red) vs fMRI gtruth + EEG gtruth (Yellow)');
correlation_value = corr2(zscore(NF_estimated_fMRI)+zscore(NF_EEG_test),zscore(NF_fMRI_test)+zscore(NF_EEG_test));
legend(sprintf('Correlation = %0.3f',correlation_value))

% EEG gtruth VS fMRI gtruth + EEG gtruth
subplot(2,2,3);
hold off;
for i=1:nbloc % plot Task bands
    x = [blocsize*(i-1)+blocsize/2+1:1:blocsize*i+1];
    y = maxval*ones(size(x));
    basevalue = minval;
    ha = area(x, y, basevalue);
    set(ha(1), 'FaceColor', [0.93 0.93 0.93]);
    set(ha, 'LineStyle', 'none');
    set(gca, 'Layer', 'top');
    hold on;
end
hold on;
plot5=plot((zscore(NF_EEG_test)).*50, 'LineWidth', 1); plot5.Color(4)=0.7;
plot6=plot((zscore(NF_fMRI_test)+zscore(NF_EEG_test)).*50, 'LineWidth', 1); plot6.Color(4)=0.7;
title('EEG gtruth (Red) vs fMRI gtruth + EEG gtruth (Yellow)');
correlation_value = corr2(zscore(NF_EEG_test),zscore(NF_fMRI_test)+zscore(NF_EEG_test));
legend(sprintf('Correlation = %0.3f',correlation_value))
end

