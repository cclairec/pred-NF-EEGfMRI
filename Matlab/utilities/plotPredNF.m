function [] = plotPredNF(Res)
%PLOTPREDNF Plot the NF prediction with the ground truth
%   Res : Res object, output of pref_NF_from_eeg_fmri_1model_AVC.m

blocsize=160;nbloc=8;
minval = -300;
maxval = 300;

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
plot1=plot(zscore(Res.NF_estimated_fMRI).*50, 'LineWidth', 1); plot1.Color(4)=0.7;
plot2=plot(zscore(Res.NF_fMRI_test).*50, 'LineWidth', 1); plot2.Color(4)=0.7;
plot3=plot(zscore((Res.NF_EEG_test)+(Res.NF_fMRI_test)).*50, 'LineWidth', 1); plot3.Color(4)=0.7;

title('NF estimated fMRI (Red), test NF fMRI score (Yellow), test NF EEG+fMRI score(Purple)');

end

