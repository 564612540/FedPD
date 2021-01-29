function [logs]=mainiter_SGD(para,data,gra,fnc,init_x)
    logs=zeros(para.iter,2);
    X=init_x;
    %%main iteration
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(X,Dataset,gra,fnc,1);
        Dataset=transform(data,para.node,para.bs,para.mini_bs);
        G=getgrad(X,Dataset,gra,para.stepsize);
        if(para.mini_bs<1)
            X=X-G/sqrt((i-1)*para.iter/5+1);
        else
            X=X-G;
        end
        for j=2:para.localiter
            Dataset=transform(data,para.node,para.bs,para.mini_bs);
            G=getgrad(X,Dataset,gra,para.stepsize);
            if(para.mini_bs<1)
                X=X-G/((i-1)*para.iter+j);
            else
                X=X-G;
            end
        end
        X=para.W*X;
    end
end