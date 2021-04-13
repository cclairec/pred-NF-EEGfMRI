%% Basic Pursuit with Forward-Backward

function x = forward_backward_optimisation(A, y, lambda, method_i, rho)

addpath('optim_function/');

% Matrix and observation.
if nargin < 2
    n = 20;
    p = 1280;
    A = randn(p,64,n);
    A(:,5,[1:50 300:400]) = A(:,5,[1:50 300:400]).*3;
    A(:,30,[100:200 300:400]) = A(:,30,[100:200 300:400]).*2;
    A(:,50,[200:500]) = A(:,5,[200:500]).*1.5+randn;
    y = randn(p,1);
end

% Dimension of the problem.
[p,n] = size(A);

% Regularization parameter.
if nargin < 3
    lambda = 0.2;
end

if nargin < 4
    method_i = 2;
end
if nargin < 5
    rho = 1500;
end

    % Lipschitz constant.
    if length(size(A)) == 2
        L = norm(A);
    elseif length(size(A)) == 3
        % size AA: d2 x d1*d3
        AA = [];
        for i = 1:size(A,1)
            AA = [AA squeeze(A(i,:,:))]; 
        end
        L = max(svd(AA))^2;
    end


% List of benchmarked algorithms.
methods = {'fb', 'fista', 'nesterov'};


% operator callbacks
F = @(x)lambda*norm_L21(x)+rho*norm(x,1);
G = @(x)1/2*norm(y-A*x)^2;
if length(size(A)) == 3
    G = @(x) G3(A, x, y);
end
% Proximal operator of F. 
clear ProxF;
% ProxF = @(x,tau)perform_soft_thresholding(x,lambda*tau); % ProxF = @(x,tau)prox_power(x, lambda*tau, 1);
ProxF = @(x,tau)prox_L21_1(x, lambda*tau, rho*tau);

% Gradient operator of G.
GradG = @(x)A'*(A*x-y);
if length(size(A)) == 3
    GradG = @(x) gradG3(A, x, y);
end

% Function to record the energy.
options.report = @(x)F(x)+G(x);

% Bench the algorithm
options.niter = 6000;

options.method = methods{method_i};
clear x predicted_values;
x_init = zeros(n,1);
if length(size(A)) == 3
    x_init = zeros(size(A,2), size(A,3));
    %ProxF_21 = @(x,tau)prox_L21(x, lambda*tau);
    %x_init = perform_fb(x_init, ProxF_21, GradG, L, options);
end
%disp(['  Constant L before perform fb : ' num2str(L)]);
[x,e] = perform_fb(x_init, ProxF, GradG, 2*L, options); % in toolbox_optim



disp_=0;
if disp_
    nbloc=8;
    blocsize=160;
    % error estimation
    disp(['  **  Correlation between learning and the model, R^2 = ' num2str(R2_train)]);
    clear SStot SSres;
    SStot = sum( (y(:)-mean(y(:))).^2 );
    predicted_values = A*x ;
    SSres = sum( (y(:)-predicted_values(:)).^2 );
    R2_train = 1 - SSres/SStot
    titre = ['learning ' options.method ':  lambda ' num2str(lambda) '. R^2 = ' num2str(R2_train)];
    feature_disp2b(y,blocsize,nbloc); hold on, plot(predicted_values,'LineWidth',2), title(titre);
    
end

end

% for 3D matrices:
function gradG = gradG3(A, x, y)
% A: TxExF
% x: ExF
% y: Tx1
    gradG=zeros(size(x));
    for i=1:size(A,1)
        gradG=gradG+squeeze(A(i,:,:))*(trace(squeeze(A(i,:,:))'*x) - y(i));
    end
end

function G = G3(A, x, y)
    G=0;
    for i=1:size(A,1)
        G=G+(y(i) - trace(squeeze(A(i,:,:))'*x))^2;
    end
end