function n21 = norm_L21(x)
% dim 1 : L1
% dim 2 : L2
% n21 = || x ||_21 = sum_i ( sum_j |x(i,j)|^2 )^(1/2) 
n21=0;
for i = 1 : size(x,1)
    for j = 1 : size(x,2)
        n21 = n21 + x(i,j)^2;
    end
    n21 = n21 + sqrt(n21);
end
end

