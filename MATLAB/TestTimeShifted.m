clear
close all
clc


T = 1000;
Tau = 5;

NumTaus = 20;

[X,Y] = OneDirAR(T,Tau);
I = MI_Shift(Y,X,20);

Z = 0:NumTaus-1;
plot(Z,I)
xline(Tau,'LineWidth',2)
xlabel('tau')
ylabel('I(X(t),Y(t-tau))')
