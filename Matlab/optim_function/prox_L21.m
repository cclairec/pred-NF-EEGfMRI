 function prox = prox_L21(x, lambda, group_weight)
 % proximal operator for  lambda||.||21
 if nargin ==2
     group_weight = ones(1,size(x,1));
 end
 for p=1:size(x,1)
     for k = 1:size(x,2)
         prox(p,k) = x(p,k)* max(1-(lambda*sqrt(group_weight(p)))/norm(x(p,:),2),0);
     end
 end

 end