 function prox = prox_L21_1(x, lambda, mu)
 % proximal operator for  mu||.||1 + lambda||.||21
 % x: ExF
 % mu: large value will lead to spatially very sparse solution
 % lambda: large value will promote sources with smooth time series
 % Proximal operator added by Claire Cury
 
 for p=1:size(x,1)
     for k = 1:size(x,2)
         div_=0;
         for kk = 1:size(x,2)
             div_ = div_ +max(abs(x(p,kk))-mu,0)^2;
         end
         div_ = sqrt(div_);
         if div_ ==0
             frac = 0;
         else
             frac = lambda/div_;
         end
         
         prox(p,k) = sign(x(p,k)) * max(abs(x(p,k))-mu,0) * max((1 - frac), 0);
     end
 end
 
 end