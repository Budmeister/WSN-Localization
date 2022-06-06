function I = MI_Shift(X,Y,NumTaus);

I = zeros(NumTaus,1);

for i = 1:NumTaus
    x = X(1:end-(i-1));
    y = Y(1+(i-1):end);
    
    C = corr([x,y]);
    I(i) = -0.5*log(det(C));
end
