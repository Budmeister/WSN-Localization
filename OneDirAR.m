function [X,Y] = OneDirAR(T,Tau);


Y = normrnd(0,1,T,1);
X = zeros(T,1);

X(1:Tau) = normrnd(0,1,Tau,1);
b = unifrnd(-1,1);
for i = Tau+1:T
    X(i) = b*Y(i-Tau)+normrnd(0,1);
end

