function [logs]=mainiter_FedPD_1(para,data,gra,fnc,init_x)
    logs=zeros(para.iter,2);
    X_0=init_x;
    X=X_0;
    X_old=X;
    lam = X_0-X;
%     Dataset=transform(data,para.node,para.bs,1);
%     lam=getgrad(X_old,Dataset,gra,para.stepsize);
    %%main iteration
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(X_0,Dataset,gra,fnc,1);
        if (mod(i,10)==1)
            X_ex=X_0;
            V_ex=getgrad(X_ex,Dataset,gra,para.stepsize);
        end
        for j=1:para.localiter
            Dataset=transform(data,para.node,para.bs,para.mini_bs);
            G_1=getgrad(X,Dataset,gra,para.stepsize);
%             G_2=getgrad(X_old,Dataset,gra,para.stepsize);
            G_3=getgrad(X_old,Dataset,gra,para.stepsize);
            X_old=X;
            V_ex=V_ex+G_1-G_3;
            X=X_0-(V_ex-lam);
        end
%         Dataset=transform(data,para.node,para.bs,para.mini_bs);
% %         G_2=getgrad(X_old,Dataset,gra,para.stepsize);
%         G_3=getgrad(X_ex,Dataset,gra,para.stepsize);
%         X_old=X;
        lam_t=-V_ex;
        X_0=para.W*(X+(lam_t));
        lam=-lam_t;
    end
end