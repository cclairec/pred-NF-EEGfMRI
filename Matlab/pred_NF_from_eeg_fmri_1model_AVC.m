% suj_ID
% session: 'S1s1', 'S1s2',... must contain subfolders MI_PRE, NF1, NF2, NF3
% learn_run and test_run: NF1 NF2 or NF3
% mod: model to learn 'eeg', 'fmri' or 'both'. Here 'both' means 2 models.
% nb_bandfreq: number of freq bands (default 10)
% size_bandfreq: width of freq bands for the design matrice
% reg: regularisation function: 'lasso' (matlab), 'fistaL1' or 'L12'
% clean_test: boolean, to clean or not the design matrix of data test.

function Res = pred_NF_from_eeg_fmri_1model_AVC(suj_ID, session, learn_run, test_run,mod, nb_bandfreq, reg_function,clean_test)
DataPath='C:/Users/cpinte/Documents/Data/Sujets_sains'; % Where subjects are stored
ResPath=['C:/Users/cpinte/Documents/Results/Sujets_sains/Res_', suj_ID ,'_' ,session, '_', learn_run, '_', test_run, '/']; % Where figures are saved
if not(isfolder(ResPath))
    mkdir(ResPath)
end
if nargin ==0
error('Some parameters are needed : suj_ID \n session: S1s1, S1s2,.. must contain subfolders nf1, nf2, nf3, mi_pre \n learn_run and test_run: nf1 nf2 or nf3 \n mod: eeg, fmri or both. Here both means 2 models. \n nb_bandfreq: number of freq bands (default 10) \n size_bandfreq: width of freq bands for the design matrice \n reg: regularisation function: lasso (matlab), fistaL1 (default) or L12 \n clean_test: boolean, to clean or not the design matrix of data test. (not used) \n')
end

if nargin >=4
        disp(['  *  Loading EEG data for subject : ' suj_ID ' session ' session ' learning run ' learn_run]);
        suj_learning_EEG=load([DataPath '/' suj_ID '/NF_eeg/d_' suj_ID '_' session '_' learn_run '_NFeeg_scores.mat']);
        disp(['  *  Loading fMRI data for subject : ' suj_ID ' session ' session ' learning run ' learn_run]);
        suj_learning_fMRI=load([DataPath '/' suj_ID '/NF_bold/d_' suj_ID '_' session '_' learn_run '_NFbold_scores.mat']);
        disp(['  *  Loading EEG data for subject : ' suj_ID ' session ' session ' testing run ' test_run]);
        suj_testing_EEG=load([DataPath '/' suj_ID '/NF_eeg/d_' suj_ID '_' session '_' test_run '_NFeeg_scores.mat']);
        disp(['  *  Loading fMRI data for subject : ' suj_ID ' session ' session ' testing run ' test_run]);
        suj_testing_fMRI=load([DataPath '/' suj_ID '/NF_bold/d_' suj_ID '_' session '_' test_run '_NFbold_scores.mat']);
        suj_all = load([DataPath '/' suj_ID '/S21_all_infos.mat']);

        if strcmp(learn_run,'run-01')
            badseg_learn = suj_all.S21_all.nf1.eeg.pp2.data.data.event;
        end
        if strcmp(test_run,'run-01')
            badseg_test = suj_all.S21_all.nf1.eeg.pp2.data.data.event;
        end
        
        if strcmp(learn_run,'run-02')
            badseg_learn = suj_all.S21_all.nf2.eeg.pp2.data.data.event;
        end
        if strcmp(test_run,'run-02')
            badseg_test = suj_all.S21_all.nf2.eeg.pp2.data.data.event;
        end
        
        if strcmp(learn_run,'run-03')
            badseg_learn = suj_all.S21_all.nf3.eeg.pp2.data.data.event;
        end
        if strcmp(test_run,'run-03')
            badseg_test = suj_all.S21_all.nf3.eeg.pp2.data.data.event;
        end
        
end

if nargin <5
    mod = 'fmri';
end

if nargin <6
    nb_bandfreq = 10;
end

if nargin <7
    reg_function = 'fistaL1'; % 'fistaL1';
end

if nargin <8
    clean_test = 1;
end
disp_fig = 1;
%%

smooth_param = 7; % window size >3 and odd
nbloc=8;
blocsize=160;
f_m= 7; % minimum freq to consider
f_M=30; % maximum freq to consider
f_win=ceil((f_M-f_m)/nb_bandfreq); % windows frequency size

disp(['  *  Reshaping EEG signals of learning and testing sessions']);
clear EEG_signal_reshape_learning EEG_signal_reshape_test

EEG_signal_reshape_learning(:,:) = reshape(suj_learning_EEG.NF_eeg.eegdata, 64, 64000);
EEG_signal_reshape_test(:,:) = reshape(suj_testing_EEG.NF_eeg.eegdata, 64, 64000);

% removing bad segments from signals:
disp('  * Bad segments already removed from EEG signal');
% badseg_learning = [suj_learning.eeg.pp2.data.data.event(arrayfun(@(x)strcmp(x.type,'BAD'),suj_learning.eeg.pp2.data.data.event)).latency];
% badseg_testing = [suj_testing.eeg.pp2.data.data.event(arrayfun(@(x)strcmp(x.type,'BAD'),suj_testing.eeg.pp2.data.data.event)).latency];

badseg_learning = [badseg_learn(arrayfun(@(x)strcmp(x.type,'BAD'),badseg_learn)).latency];
badseg_testing = [badseg_test(arrayfun(@(x)strcmp(x.type,'BAD'),badseg_test)).latency];

for jj=1:length(badseg_learning)
EEG_signal_reshape_learning(:,badseg_learning(jj):badseg_learning(jj)+199) = 0;
end
for jj=1:length(badseg_testing)
EEG_signal_reshape_test(:,badseg_testing(jj):badseg_testing(jj)+199) = 0;
end

% Removing the corresponding removed times to the NF scores
disp('  * Identifying bad segmentation');
%badseg_learning=suj_learning_EEG.EEG_FEAT.bad_segments(1:50:end);
%bad_scores_learning_ind=find(badseg_learning);
%bad_scores_learning_ind(bad_scores_learning_ind>1280)=1280
badseg_learning = [badseg_learn(arrayfun(@(x)strcmp(x.type,'BAD'),badseg_learn)).latency];
bad_scores_learning_ind = floor(badseg_learning/50);
bad_scores_learning_ind(bad_scores_learning_ind>1280)=1280

% badseg_testing=suj_testing_EEG.EEG_FEAT.bad_segments(1:50:end);
% bad_scores_testing_ind=find(badseg_testing);
% bad_scores_testing_ind(bad_scores_testing_ind>1280)=1280
badseg_testing = [badseg_test(arrayfun(@(x)strcmp(x.type,'BAD'),badseg_test)).latency];
bad_scores_testing_ind = floor(badseg_testing/50);
bad_scores_testing_ind(bad_scores_testing_ind>1280)=1280

% Load Channel names
load([ 'C:/Users/cpinte/Documents/Data/Patients/Chanlocs.mat']);
k=0;
for i=[1:31 33:64]
    k=k+1;
    elect_names = Chanlocs(i).labels;
end

% Extract NF_EEG:
disp(['  *  Extracting NF_EEG scores']);
X_eeg_learn_smooth_Lap = zscore(suj_learning_EEG.NF_eeg.smoothnf);
X_eeg_test_smooth_Lap = zscore(suj_testing_EEG.NF_eeg.smoothnf);

% extract NF_fMRI:
disp(['  *  Extracting NF_fMRI scores']);
clear fmri_NF*
fmri_NF_learn = suj_learning_fMRI.NF_bold.sma.smoothnf; 
fmri_NF_test= suj_testing_fMRI.NF_bold.sma.smoothnf;
kk=0;
for ii=1:4:length(X_eeg_learn_smooth_Lap)-3 
    kk=kk+1;
    for k=ii:ii+3
        fmri_NF_reshape(k) = fmri_NF_learn(kk);
        fmri_NF_reshape_test(k) = fmri_NF_test(kk);
    end
end
X_fmri_reshape_learn_smooth= zscore(fmri_NF_reshape);%(fmri_NF_reshape-mean(fmri_NF_reshape))./(max( fmri_NF_reshape)-min(fmri_NF_reshape));
X_fmri_reshape_test= zscore(fmri_NF_reshape_test);%((fmri_NF_reshape_test - mean(fmri_NF_reshape_test))./(max( fmri_NF_reshape_test)-min(fmri_NF_reshape_test)));

% Smooth NF scores:
%X_fmri_reshape_learn_smooth = sgolayfilt(X_fmri_reshape, 3, smooth_param);
%X_fmri_reshape_test = sgolayfilt(X_fmri_reshape_test, 3, smooth_param);

%%%%%%%%%%%%%%%%%%%%%%%%%% Test
% fmri_NF_learn = suj_learning_fMRI.NF_bold.sma.smoothnf; 
% fmri_NF_test= suj_testing_fMRI.NF_bold.sma.smoothnf;
% kk=0;
% for ii=1:4:length(X_eeg_learn_smooth_Lap)-3 
%     kk=kk+1;
%     for k=ii:ii+3
%         fmri_NF_reshape(k) = fmri_NF_learn(kk);
%         fmri_NF_reshape_test(k) = fmri_NF_test(kk);
%     end
% end
% X_fmri_reshape= zscore(fmri_NF_reshape);%(fmri_NF_reshape-mean(fmri_NF_reshape))./(max( fmri_NF_reshape)-min(fmri_NF_reshape));
% X_fmri_reshape_test= zscore(fmri_NF_reshape_test);%((fmri_NF_reshape_test - mean(fmri_NF_reshape_test))./(max( fmri_NF_reshape_test)-min(fmri_NF_reshape_test)));
% 
% %X_fmri_reshape_learn_smooth = sgolayfilt(X_fmri_reshape, 3, smooth_param);
% %X_fmri_reshape_test = sgolayfilt(X_fmri_reshape_test, 3, smooth_param);
% 
% figure;
% hold on;
% plot1=plot(X_fmri_reshape_learn_smooth, 'c','LineWidth', 1); plot1.Color(4)=0.7;
% plot2=plot(X_fmri_reshape, 'b','LineWidth', 1); plot2.Color(4)=0.7;
% title('fMRI nf + sgolayfilt (Cyan) vs fMRI smoothnf (Blue)');
%%%%%%%%%%%%%%%%%%%%%%%%%%

% X are NF values:
disp(['  *  NF score to be learned, with mod: ' mod]);
clear X X_test; pause(0.5);

X_fmri_reshape_learn_smooth = X_fmri_reshape_learn_smooth.*50;

X_fmri_reshape_learn_smooth(bad_scores_learning_ind)=0; %%%%%%%%%%%%% 00
X_fmri_reshape_test = X_fmri_reshape_test.*50;
X_fmri_reshape_test(bad_scores_testing_ind)=0;

X_eeg_test_smooth_Lap = X_eeg_test_smooth_Lap.*50;
X_eeg_test_smooth_Lap(bad_scores_testing_ind)=0;

X_eeg_learn_smooth_Lap = X_eeg_learn_smooth_Lap.*50;
X_eeg_learn_smooth_Lap(bad_scores_learning_ind)=0;

if strcmp(mod,'eeg')
    X = X_eeg_learn_smooth_Lap;
    X_test = X_eeg_test_smooth_Lap;
end
if strcmp(mod,'fmri')
    X = X_fmri_reshape_learn_smooth;
    X_test = X_fmri_reshape_test;
end
if strcmp(mod,'both')
    X =  [X_eeg_learn_smooth_Lap + X_fmri_reshape_learn_smooth]; % weight_eeg.*X_eeg_smooth + (1-weight_eeg).*X_fmri_reshape_smooth;  % [X_eeg_smooth ; X_fmri_reshape_smooth];
    X_test = [X_eeg_test_smooth_Lap+ X_fmri_reshape_test];
    weight=0.5;
end


% compute the design matrix for learning step and test:
disp(['  *  Compute the design matrices for learning and test:']);
for ff=1:nb_bandfreq
       freq_band_learning{ff} = zeros(length([1:50:64000]),size(EEG_signal_reshape_learning,1)); 
       freq_band_test{ff} = zeros(length([1:50:64000]),size(EEG_signal_reshape_test,1)); 
end
k=1; clear freq_band_*;
steps = 400; 
for i=1:50:64000, % shift for 1/4 of second
    k=k+1;
    f_interval{1} = [f_m f_m+f_win]; 
    for ff=1:nb_bandfreq
        freq_band_learning{ff}(k,:)=(bandpower(EEG_signal_reshape_learning(:,i:min(size(EEG_signal_reshape_learning,2),i+steps))',200,f_interval{ff}));
        freq_band_test{ff}(k,:)=(bandpower(EEG_signal_reshape_test(:,i:min(size(EEG_signal_reshape_test,2),i+steps))',200,f_interval{ff}));
        f_interval{ff+1} = [(max(f_interval{ff})-1) (max(f_interval{ff})-1) + f_win];
    end
end

% Removing some electrods (the noisy ones, like occipital) from the design matrix of both steps:
disp(['  *  Removing some electrods (the noisy ones, like occipital) from the design matrix of both steps:']);

motor_channels = [5,6,17,18,21,22,23,24,25,26,27,28,33,34,35,36,41,42,43,44,49,50,64]; % electrods to keep, base 64 already
%frontal_channels = [33 34 17]; % electrods to keep, base 64 already. Removed for patients.
all_channels = [1:64];
ind_elect_eeg_exclud = 1:64; % electrodes to exclude
ind_elect_eeg_exclud([motor_channels ])=[];
%ind_elect_eeg_exclud([all_channels ])=[];

% clear Emaps; k=0;
% for i=[1:31 33:64]
%     k=k+1;
%     Emaps{1,k}= Chanlocs(i).labels;
%     Emaps{3,k}= Chanlocs(i).X;
%     Emaps{2,k}= -Chanlocs(i).Y;
% end

% elect_kept=ones(1,64);
% elect_kept(ind_elect_eeg_exclud)=0;
% disp(elect_kept)
% plotElecPotentials(Emaps,elect_kept([1:31 33:end])',1), title(['motor_channel']); 

% alternative using variance of electrodes
% var_elect = var(EEG_signal_reshape_learning');
% var_ref = var_elect(5)+0.1*var_elect(5);
% ind_elect_eeg_exclud = 1:64; % electrodes to exclude
% ind_elect_eeg_exclud(var_elect<=var_ref)=[]; % cancel electrodes to kepp

% elect_chosen=ones(1,64);
% elect_chosen(ind_elect_eeg_exclud)=0;
% disp(elect_chosen)
% plotElecPotentials(Emaps,elect_chosen([1:31 33:end])',1), title(['alternative']); 

clear D_*;
index_freq_band_used = [1:length(freq_band_learning)];
D_learning = double(cell2mat(freq_band_learning(index_freq_band_used)));
D_learning = D_learning(2:end,:);
sz=size(D_learning,1);

D_learning = reshape(D_learning,sz,64,nb_bandfreq);
D_learning(:,ind_elect_eeg_exclud,:)=0;
%D_learning = reshape(D_learning_tmp,sz,64*nb_bandfreq);


D_test = double(cell2mat(freq_band_test(index_freq_band_used)));
D_test = D_test(2:end,:);
sz=size(D_test,1);

D_test = reshape(D_test,sz,64,nb_bandfreq);
D_test(:,ind_elect_eeg_exclud,:)=0;
%D_test = reshape(D_test_tmp,sz,64*nb_bandfreq);

% Estimate design matrix for fMRI model:
p = [4 16 1 1 3 0 32];
hrf4 = spm_hrf( 1/4, p ); % 4Hz
p = [5 16 1 1 3 0 32];
hrf5 = spm_hrf( 1/4, p );
p = [3 16 1 1 3 0 32];
hrf3 = spm_hrf( 1/4, p );

for i=1:size(D_learning,2) % elect
    
    for j = 1 : size(D_learning,3)
        
        clear resp3 resp4 resp5;
        
        resp3=conv(D_learning(:,i,j),hrf3);
        resp3 = resp3(1:length(D_learning(:,i,j)));
        D_learning_(:,i,j)= resp3;
        clear resp3;
        resp3=conv(D_test(:,i,j),hrf3);
        resp3 = resp3(1:length(D_test(:,i,j)));
        D_test_(:,i,j)= resp3;
        
        resp4=conv(D_learning(:,i,j),hrf4);
        resp4 = resp4(1:length(D_learning(:,i,j)));
        D_learning_(:,i+size(D_learning,2),j)= resp4;
        clear resp4;
        resp4=conv(D_test(:,i,j),hrf4);
        resp4 = resp4(1:length(D_test(:,i,j)));
        D_test_(:,i+size(D_learning,2),j)= resp4;
        
        resp5=conv(D_learning(:,i,j),hrf5);
        resp5 = resp5(1:length(D_learning(:,i,j)));
        D_learning_(:,i+2*size(D_learning,2),j)= resp5;
        clear resp5;
        resp5=conv(D_test(:,i,j),hrf5);
        resp5 = resp5(1:length(D_test(:,i,j)));
        D_test_(:,i+2*size(D_learning,2),j)= resp5;
        

    end
end
D_learning(bad_scores_learning_ind,:,:)=0; %%%%%%%%0
D_test(bad_scores_testing_ind,:,:)=0;
D_learning_(bad_scores_learning_ind,:,:)=0;
D_test_(bad_scores_testing_ind,:,:)=0;



% cleaning the design matrix for learning step, by removing bad observations:
disp(['  *   cleaning the design matrix for learning step, by removing bad observations:']);
mean_3std_learn = (mean(mean(D_learning(D_learning~=0))))+3*mean(std(D_learning(D_learning~=0)));
D_learning(D_learning>mean_3std_learn)=mean_3std_learn;

mean_3std_learn = (mean(mean(D_learning_(D_learning_~=0))))+3*mean(std(D_learning_(D_learning_~=0)));
D_learning_(D_learning_>mean_3std_learn)=mean_3std_learn;

if clean_test
    disp(['  *   cleaning the design matrix for TEST step, by thresholding bad observations:']);
    D_test(D_test>(mean(mean(D_test(D_test~=0)))+3*mean(std(D_test(D_test~=0)))))= mean(mean(D_test(D_test~=0)))+3*mean(std(D_test(D_test~=0)));
    D_test_(D_test_>(mean(mean(D_test_(D_test_~=0)))+3*mean(std(D_test_(D_test_~=0)))))= mean(mean(D_test_(D_test_~=0)))+3*mean(std(D_test_(D_test_~=0)));
end

if strcmp(mod,'both') || strcmp(mod,'fmri')
    for i = 1 : size(D_learning,1)
        D_learning_old(i,:,:) = [squeeze(D_learning(i,:,:)); squeeze(D_learning_(i,:,:))];
    end
    size(D_test)
    size(D_test_)
    for i = 1 : size(D_test,1)
        D_test_old(i,:,:) = [squeeze(D_test(i,:,:)); squeeze(D_test_(i,:,:))];
    end
else
    D_learning_old = D_learning;
    D_test_old = D_test;
end

%% execution
disp(['  **  Execution...']);
clear alpha D_test_fmri D_learning_fmri;

% if strcmp(learn_run,test_run) % then cut the session in 2 blocks:
%     learning_block=blocsize*1+1:round(length(D_learning_old)/2);
%     testing_block = learning_block(end)+1:length(D_test_old);
% else
%     learning_block=blocsize*1+1:length(D_learning_old);
%     testing_block=1:length(D_test_old);
% end

learning_block=blocsize*1+1:length(D_learning_old);
testing_block=1:length(D_test_old);

tic
testing_dummy_data = 0;
clear rep_learning rep_test;
if testing_dummy_data
    coeff_NF = [0.05 0.5 10 -5 3 2];
    rep_learning = D_learning_old(learning_block,[1 2 5 6 17 34])*coeff_NF'; % build the NF variable for learning
    D_learning = D_learning_old(learning_block,:);
    rep_test = D_test_old(testing_block,[1 2 5 6 17 34])*coeff_NF'; % build the rep test variable
    D_test = D_test_old(testing_block,:);
else
    rep_learning = X(learning_block); % removing the first bloc from learning phase
    rep_test = X_test(testing_block); %(block_task_test);
    
    D_learning = D_learning_old(learning_block,:,:);
    D_learning(:,:,end+1) = ones(size(D_learning,1), size(D_learning,2));
    D_test = D_test_old(testing_block,:,:);
    D_test(:,:,end+1) = ones(size(D_test,1), size(D_test,2));

end

clear alpha p;

disp(['  **  Estimating regularisation parameter lambda for method ' reg_function ':']);
if strcmp(reg_function, 'lasso')
    lambdas=[0.1:0.2:10];
elseif strcmp(reg_function, 'fistaL1')
    lambdas=[0:80:2000]; %initial values
    %lambdas=[0:500:50000]; % test
    %lambdas=5000;
end

% Creating object input for testing
lambda_choice_input.D_learning = D_learning;
lambda_choice_input.rep_learning = rep_learning;
lambda_choice_input.nb_bandfreq = nb_bandfreq;
lambda_choice_input.reg_function = reg_function;
lambda_choice_input.lambdas = lambdas;
lambda_choice_input.disp_fig = disp_fig;
save(['',ResPath,'lambda_choice.mat'],'lambda_choice_input');

[regul_eeg] = lambda_choice(D_learning,rep_learning,nb_bandfreq, reg_function,lambdas,disp_fig);
saveas(gcf,['',ResPath,'Fig1.png'])
% end

disp(['  **  EEG lambda parameter for method ' reg_function ' is ' num2str(regul_eeg)]);
if strcmp(reg_function, 'lasso')
    [alpha FitInfo_eeg] = lasso(D_learning,rep_learning,'Lambda',regul_eeg,'Standardize',false); % , 'CV', 400 'Standardize',false
    
    predicted_values = D_learning*alpha + FitInfo_eeg.Intercept;
    NF_estimated = D_test*alpha + FitInfo_eeg.Intercept;

    
elseif strcmp(reg_function, 'fistaL1')
    alpha = forward_backward_optimisation(D_learning, rep_learning', regul_eeg);
    
    for t = 1:size(D_test,1)
        NF_estimated(t) = trace(squeeze(D_test(t,:,:))'*alpha);
    end
    
    for t = 1:size(D_learning,1)
        predicted_values(t) = trace(squeeze(D_learning(t,:,:))'*alpha);
    end
end


t=toc;
fprintf('%f secondes\n',toc);

smooth_NF = 1;
if smooth_NF
    smooth_wind_test = 2;
    NF_estimated_notsmoothed = NF_estimated;
    for i=1:length(NF_estimated)
        if i<=smooth_wind_test
            NF_estimated(i)=mean(NF_estimated_notsmoothed(1:i));
        else
            NF_estimated(i)=mean(NF_estimated_notsmoothed(i-smooth_wind_test:i));
        end
    end
end

if strcmp(mod,'eeg')
    weight = 0;
end

clear filter_estimated;
if strcmp(reg_function, 'fistaL1')
    % EEG start at 65+64+1=130, since it's design matrix it at the end:
    % fMRI4s, fMRI5s, EEG.
    nb_mat_design = size(alpha,1)/64;
    ch1=1;
    ch64=64;
    filter_estimated = {};
    for nmd = 1: nb_mat_design
        alpha_1 = full(alpha(ch1:ch64,1:end-1));
        alpha_1(32,:)=[];
        filter_estimated{nmd}=alpha_1;
        ch1 = ch64+1;
        ch64 = ch1+63;
    end
    filter_estimated_eeg=filter_estimated{nb_mat_design};
    filter_estimated_fmri = filter_estimated(1:nb_mat_design-1);
end

elect_kept=ones(1,64);
elect_kept(ind_elect_eeg_exclud)=0;
%length(nonzeros(filter_estimated_eeg))
%length(nonzeros(filter_estimated_fmri))
% electrods involved per frequency bands:

% plotElecPotentials(Emaps, suj_learning.eeg.pp2.feat(eeg_feat_ind).filter(1,[1:31 33:end]),1), title(['CSP filter used by Lorraine']);
if disp_fig
    
    clear Emaps; k=0;
for i=[1:31 33:64]
    k=k+1;
    Emaps{1,k}= Chanlocs(i).labels;
    Emaps{3,k}= Chanlocs(i).X;
    Emaps{2,k}= -Chanlocs(i).Y;
end
%     for i = 1:nb_bandfreq
%         plotElecPotentials(Emaps,filter_estimated_eeg(1:63,i)',1);
%         title([reg_function ': lambda ' num2str(regul_eeg) '. estimated filter for band freq for EEG ' num2str(f_interval{index_freq_band_used(i)})]);
%     end
    
    plotElecPotentials(Emaps,sum(abs(filter_estimated_eeg(1:63,:)),2)',1), title(['X0 estimated abs filter all band of freq for EEG']);
    saveas(gcf,['',ResPath,'Fig2.png'])
    plotElecPotentials(Emaps,sum((filter_estimated_eeg(1:63,:)),2)',1), title(['X0 estimated filter all band of freq for EEG']);
    saveas(gcf,['',ResPath,'Fig3.png'])
    
%     kk=1;
%     for i = 1:nb_bandfreq
%         plotElecPotentials(Emaps,filter_estimated_fmri{1}(1:63,i)',1);
%         title([reg_function ': lambda ' num2str(regul_eeg) '. estimated filter for band freq for fMRI (4s) ' num2str(f_interval{index_freq_band_used(kk)})]);
%                 plotElecPotentials(Emaps,filter_estimated_fmri{2}(1:63,i)',1);
%         title([reg_function ': lambda ' num2str(regul_eeg) '. estimated filter for band freq for fMRI (5s) ' num2str(f_interval{index_freq_band_used(kk)})]);
%         kk=kk+1;
%     end
    
    plotElecPotentials(Emaps,sum(abs(cell2mat(filter_estimated_fmri)),2)',1), title(['X3 X4 X5 estimated abs filter all band of freq for fMRI']);
    saveas(gcf,['',ResPath,'Fig4.png'])
    plotElecPotentials(Emaps,sum((cell2mat(filter_estimated_fmri)),2)',1), title(['X3 X4 X5 estimated filter all band of freq for fMRI']);
    saveas(gcf,['',ResPath,'Fig5.png'])

    plotElecPotentials(Emaps,elect_kept([1:31 33:end])',1), title(['Electrodes kept in the model']);  
    saveas(gcf,['',ResPath,'Fig6.png'])
        
    length(nonzeros(filter_estimated_eeg))
    length(nonzeros(cell2mat(filter_estimated_fmri)))
end

%%
Res.learning_session = learn_run;
Res.test_session = test_run;
Res.alpha = alpha;
%Res.alpha_fMRI = alpha_fmri;
Res.NF_estimated_fMRI = NF_estimated;
Res.filter_estimated_e = filter_estimated_eeg;
Res.filter_estimated_f = filter_estimated_fmri;

Res.NF_fMRI_test = X_fmri_reshape_test;
Res.NF_EEG_test = X_eeg_test_smooth_Lap;
Res.NF_EEG_learn = X_eeg_learn_smooth_Lap;
Res.NF_fMRI_learn = X_fmri_reshape_learn_smooth;
%Res.rep_test_f = rep_test_fmri;
Res.rep_test = rep_test;

Res.nb_bandfreq = nb_bandfreq;
Res.lambda = regul_eeg;
Res.time = t;
Res.D_test = D_test;
Res.D_learn = D_learning;
Res.index_freq_band_used = index_freq_band_used;
Res.f_interval = f_interval;
%Res.weight = weight;
Res.elect_used = elect_kept;
Res.bad_scores_testing_ind = bad_scores_testing_ind;
Res.bad_scores_learning_ind = bad_scores_learning_ind;
% Res.eeg_filter_ind = eeg_filter_ind;
% Res.eeg_feat = eeg_feat(eeg_feat_ind);
% Res.fmri_filter_ind = fmri_filter_ind;
% Res.fmri_feat = fmri_feat(fmri_feat_ind);
end
