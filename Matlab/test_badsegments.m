load('C:/Users/cpinte/Documents/Data/Patients/P002/P002_NFEEG/S1s1/NF1/EEG_features_Laplacian.mat'); 

badseg_learning = EEG_FEAT.bad_segments;
badseg_learning_ind = find(badseg_learning);

EEG_signal_reshape_learning(:,:) = reshape(EEG_FEAT.data, 64, 64000);
data = EEG_signal_reshape_learning(5,:);

for j=1:length(badseg_learning_ind)
data(1,badseg_learning_ind) = 0;
end

blocsize=160;nbloc=8;
minval = -300;
maxval = 300;
figure;
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
plot5=plot((data)*50, 'LineWidth', 1); plot5.Color(4)=0.7;
plot6=plot((EEG_FEAT.bad_segments).*500, 'LineWidth', 1); plot6.Color(4)=0.7;
title('C3 data (Red) and badsegments (Yellow)');
