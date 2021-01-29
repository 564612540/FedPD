function [grad]=gradx(x,data,node)
    %% gradient function
    %% parameters
    lambda=0.1;
    alpha=1;
    [m,n]=size(data.features);
    %% g1=sigmoid; g2=regularizer
    g2=(2*lambda*alpha*x)./((1+alpha*x.^2).^2);
%     g2=0;
    g1_num=-data.features.*(ones(m,1)*data.labels);
    g1_den=(1+exp((data.features'*x).*data.labels'));
    g1=mean(g1_num./(ones(m,1)*g1_den'),2);
    grad=(g1+g2)/node;
end