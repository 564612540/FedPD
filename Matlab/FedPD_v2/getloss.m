function [F]=getloss(X,D,f)
    [m,~]=size(X);
    F=zeros(m,1);
    for i=1:m
        F(i)=f(X(i,:)',D(i),m);
    end
end