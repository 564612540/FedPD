function [logs]=mainiter_FedPD(para,data,gra,fnc,init_x)
    logs=zeros(para.iter,2);
    X_0=init_x;
    Dataset=transform(data,para.node,para.bs,1);
    G_2=getgrad(X_0,Dataset,gra,para.stepsize);
    X=X_0;
    X_old=X;
    %%main iteration
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(X_0,Dataset,gra,fnc,1);
        G_0=getgrad(X_0,Dataset,gra,para.stepsize);
        Dataset=transform(data,para.node,para.bs,para.mini_bs);
%         G_1=G_0;
        G_1=getgrad(X_0,Dataset,gra,para.stepsize);
        G_2=getgrad(X_old,Dataset,gra,para.stepsize);
        X_old=X_0;
        X=X_0-(G_1-G_2);
%         G_2=G_1;
        for j=2:para.localiter
            Dataset=transform(data,para.node,para.bs,para.mini_bs);
            G_1=getgrad(X,Dataset,gra,para.stepsize);
            G_2=getgrad(X_old,Dataset,gra,para.stepsize);
            X_old=X;
%             G_1=G_0+(getgrad(X,Dataset,gra,para.stepsize)-getgrad(X_0,Dataset,gra,para.stepsize));
%             norm(G_1-G_2)
            X=X_0-(G_1-G_2);
%             G_2=G_1;
        end
        Dataset=transform(data,para.node,para.bs,para.mini_bs);
        G_1=getgrad(X,Dataset,gra,para.stepsize);
        G_2=getgrad(X_0,Dataset,gra,para.stepsize);
        X_old=X;
        X=X-(G_0+(G_1-G_2));
        X_0=para.W*X;
    end
end