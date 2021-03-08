
function [regul_lambda rho] = lambda_choice(Dtrain, rep_train, nb_freq_band, reg_function, lambdas, disp_)


if nargin < 4
    reg_function = 'lasso';
end
if nargin < 5
    lambdas = [0:0.1:10];
end
if nargin <6
    disp_=0;
end


clear sparsity R2_train
K = 10;

if size(rep_train,1) == 2 || size(rep_train,2)==2
    weights = [1/3 1/2 2/3];
    if size(rep_train,1) == 2
        NF_1 = rep_train(1,:);
        NF_2 = rep_train(2,:);
    elseif size(rep_train,2)==2
        NF_1 = rep_train(:,1);
        NF_2 = rep_train(:,2);
    end
    clear rep_train;
else
    weights = -1;
end

sparsity_threshold = size(Dtrain,2)*40/100 % we want at most 40 % of nonzeros coefficients

size_cv_dataset = ceil(size(Dtrain,1)/10);
delay_cv = (size(Dtrain,1)-size_cv_dataset)/K;

for w = 1:length(weights)
    if weights(w) ~= -1
        rep_train = (1-weights(w)).*NF_1 + weights(w).*NF_2;
    end
    ind = 1;
    clear CV_mean_ sparsity_ CV Cost_train Cost_train_mean sparsity_mean
    for l=lambdas
        l
	rho=l;
        k_ind = 1;
        for k = 1:delay_cv:size(Dtrain)-size_cv_dataset
            cv_set = [floor(k):floor(k)+size_cv_dataset];
            train_set = [1:size(Dtrain,1)];
            train_set(cv_set) = [];
            
            Dtrain_k = Dtrain(train_set,:,:);
            rep_train_k = rep_train(train_set);
            Dcv_k = Dtrain(cv_set,:,:);
            rep_cv_k = rep_train(cv_set);
            
            SStot = sum( (rep_cv_k(:)-mean(rep_cv_k(:))).^2 );
            SStot_train = sum( (rep_train_k(:)-mean(rep_train_k(:))).^2 );
            
            if strcmp(reg_function, 'lasso')
                clear alpha FitInfo predicted_values;
                [alpha FitInfo] = lasso(Dtrain_k,rep_train_k,'Lambda',l,'Standardize',false);
                alpha_=reshape(alpha,size(alpha,1)/nb_freq_band, nb_freq_band);
                filter_estimated=full(alpha_);
                filter_estimated(32,:)=[];
                sparsity_(ind,k_ind)=length(nonzeros(filter_estimated));
                
                predicted_values = Dcv_k*alpha + FitInfo.Intercept;
                SSres = sum( (rep_cv_k(:)-predicted_values(:)).^2 );
                CV(ind,k_ind) =  SSres/SStot;
                
                clear predicted_values;
                predicted_values = Dtrain_k*alpha + FitInfo.Intercept;
                SSres_train = sum((rep_train_k(:)-predicted_values(:)).^2);
                Cost_train(ind,k_ind) = SSres_train/SStot_train;
                
            elseif strcmp(reg_function, 'fistaL1')
                clear alpha FitInfo predicted_values;
                method_ = 2;
                alpha = forward_backward_optimisation(Dtrain_k, rep_train_k', l,method_, rho);
                sparsity_(ind,k_ind)=length(nonzeros(full(alpha)));
                
                if length(size(Dcv_k)) == 3
                    for t = 1:size(Dcv_k,1)
                        predicted_values(t) = trace(squeeze(Dcv_k(t,:,:))'*alpha);
                    end
                else
                    predicted_values = Dcv_k*alpha;
                end
                
                SSres = sum( (rep_cv_k(:)-predicted_values(:)).^2 );
                CV(ind,k_ind) =  SSres/SStot;
                
                clear predicted_values;
                if length(size(Dtrain_k)) == 3
                    for t = 1:size(Dtrain_k,1)
                        predicted_values(t) = trace(squeeze(Dtrain_k(t,:,:))'*alpha);
                    end
                else
                    predicted_values = Dtrain_k*alpha;
                end
                SSres_train = sum((rep_train_k(:)-predicted_values(:)).^2);
                Cost_train(ind,k_ind) = SSres_train/SStot_train;
            end
            k_ind=k_ind+1;
        end
        CV_mean_(ind) = mean(CV(ind,:));
        Cost_train_mean(ind) = mean(Cost_train(ind,:));
        sparsity_mean(ind) = mean(sparsity_(ind,:));
        
        if sparsity_mean(ind) <=2
            disp(['breaking at ' num2str(l)]);
            ind = ind + 1;
            break;
        end
        ind = ind + 1;
    end
    
    biais_var = (CV_mean_ + Cost_train_mean)/2;
    
    [sort_biais_var ind_sorted]=sort(biais_var);
    j=2;
    % enforcing sparsity
    while  j<=length(sort_biais_var) && sparsity_mean(ind_sorted(j))> sparsity_threshold
        j=j+1;
    end
    % even more, if models are similar:
    j_ref=j;
    while j<=length(sort_biais_var) && abs(sort_biais_var(j_ref)-sort_biais_var(j))<=0.001 
        j=j+1;
    end
    %[m, mi_biais_var]=min(biais_var)
    regul_lambda_w(w) = lambdas(ind_sorted(j-1));
    biais_var_w(w) = sort_biais_var(j-1);
    if disp_
        figure(),
        plot(lambdas(1:ind-1),CV_mean_), hold on,
        plot(lambdas(1:ind-1),Cost_train_mean); plot(lambdas(1:ind-1),biais_var, '.-');
        title('blue: CV error, red: training error, .- : cv error + training error');
    end
end
biais_var_w
regul_lambda_w
[sort_biais_var_w, ind_sorted]=sort(biais_var_w)
j=2;
if j<=length(weights)
    while abs(sort_biais_var_w(1)-sort_biais_var_w(j))<=0.001 && j<=length(weights)
        j=j+1;
    end
end
regul_lambda = regul_lambda_w(ind_sorted(j-1));
    weight = weights(ind_sorted(j-1));
end
