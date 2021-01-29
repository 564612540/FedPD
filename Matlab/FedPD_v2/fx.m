function [f]=fx(x,data,node)
    %% gradient function
    %% parameters
    lambda=0.1;
    alpha=1;
    [m,n]=size(data.features);
    %% g1=sigmoid; g2=regularizer
    f2=sum(lambda*alpha*x.^2./(1+alpha*x.^2));
    f1=log(1+exp(-(data.features'*x).*data.labels'));
    f1=mean(f1);
    f=(f1+f2)/node;
end