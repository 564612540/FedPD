function loss=ComputeLog(X,D,g,f,alpha)
    G=getgrad(X,D,g,alpha);
    F=getloss(X,D,f);
    g=norm(sum(G,1))^2;
    f=mean(F);
    %norm(A*X,'fro')^2;
    loss=[f,g];
end