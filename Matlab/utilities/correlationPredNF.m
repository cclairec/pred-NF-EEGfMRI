function [correlation_value, correlation_value_2] = correlationPredNF(Res)
%CORRELATIONPREDNF Find the correlation value between the NF prediction and the ground truth
%   Res : Res object, output of pref_NF_from_eeg_fmri_1model_AVC.m

if isequal(size(Res.NF_estimated_fMRI), size(Res.NF_fMRI_test))
    correlation_value = corr2(zscore(Res.NF_estimated_fMRI),zscore(Res.NF_fMRI_test));
    correlation_value_2 = NaN;
else % case where lNF = tNF
    gtruth = Res.NF_fMRI_test; % size 1224
    pred = Res.NF_estimated_fMRI; % size 612
    half = size(pred,2);
    gtruth_firstPart = gtruth(1:half); % first half of the ground truth
    gtruth_secondPart = gtruth(half+1:end); % second half of the ground truth
    correlation_value = corr2(zscore(pred),zscore(gtruth_firstPart));
    correlation_value_2 = corr2(zscore(pred),zscore(gtruth_secondPart));
end

end

