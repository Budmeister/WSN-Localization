function xhat = TDOALocalization(y,t)
% Function to estimate target using TDOA measurements
% Quantities:
%   q: Number of dimensions
%   m: Number of anchors
% Inputs: 
%   y: anchor locations; mxq matrix
% Output:
%   xhat: target location; 1xq vector

 [m,~] = size(y); % counting number of anchors
 
 c=340.29; % 340.29 m/s Signal Propagation speed
 
 H=y(2:m,:)-y(1,:); % Hessian matrix 
 
 tN1=t(2:m)-t(1); % time difference measurement matrix
 
%  b=zeros(m-1,2);
%   
% for time=1:(m-1) % estimate b
%     b(time,2) = 1/2.*[norm(y(time+1,:)).^2 - norm(y(1,:)).^2 - c^2*(tN1(time)^2+2*t(1)*tN(time))];
% end

r_2 = sum(y.^2,2);
b = 0.5*(r_2(2:end)-r_2(1)-c^2*(tN1.^2+2*t(1)*tN1));

H_pseudo=pinv(H); % pseudo inverser
 
xhat=H_pseudo*b; % compute localization estimate

end
