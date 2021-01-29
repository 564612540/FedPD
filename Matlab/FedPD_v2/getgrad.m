function [G]=getgrad(X,D,g,alpha)
    [m,n]=size(X);
    G=zeros(m,n);
    for j=1:m
        Y=X(j,:);
        G(j,:)=alpha*g(Y',D(j),m);
    end
end