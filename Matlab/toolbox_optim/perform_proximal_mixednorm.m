function y = perform_proximal_mixednormL21(x, tau)
% Added by Claire Cury, 2018 July. from Gramfort, 2012

% perform_soft_thresholding - soft thresholding
%
%   y = perform_soft_thresholding(x, tau);
%
%   y = prox_{tau*|.|_1}(x) = max(0,1-tau/|x|)*x
%
%   Proximal operator for the scalar L1 norm.
%
%   Copyright (c) 2010 Gabriel Peyre

for s = 1 : size(x,1)
    xs = x(s,:);
    for t = 1 : size(x,2)
        y(s,t) = max(0,1-tau./max(norm2(xs)^2,1e-10)).*x(s,t);
    end
end
