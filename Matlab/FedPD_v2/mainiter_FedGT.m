function [logs]=mainiter_FedGT(para,data,gra,fnc,init_x)
    logs=zeros(para.iter,2);
    X=init_x;
    Y=zeros(size(X));
    %%main iteration
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(X,Dataset,gra,fnc,1);
        X_0=X;
        G_0=getgrad(X,Dataset,gra,para.stepsize);
        G_0=para.W*G_0;
        X=X-G_0;
        for j=2:para.localiter
            Dataset=transform(data,para.node,para.bs,para.mini_bs);
            G=getgrad(X,Dataset,gra,para.stepsize);
            G_1=getgrad(X_0,Dataset,gra,para.stepsize);
%             X_0=X;
            G_2=G_0+(G-G_1);
            X=X-G_2;
        end
        X=para.W*X;
    end
end