function [logs]=Fed_Prox(para,data,gra,fnc,init_x)
    logs=zeros(para.iter,2);
    X_0=init_x;
    X=X_0;
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(para.W*X_0,Dataset,gra,fnc,1);
        if (para.VR==1)
            g_ex=getgrad(X,Dataset,gra,para.stepsize);
        end
        for j=1:para.localiter
            Dataset=transform(data,para.node,para.bs,para.mini_bs);
            if (para.VR==0)
                g_ex=getgrad(X,Dataset,gra,para.stepsize);
                if (para.mini_bs<1)
                    eta = 1.0/sqrt(para.localiter*(i-1)/5+j);
                else
                    eta=1;
                end
            else 
                eta=1;
            end
            X_old=X;
            X = X - eta*(g_ex + para.mu*(X-X_0));
            if (para.VR==1)
                g_ex=g_ex+getgrad(X,Dataset,gra,para.stepsize)-getgrad(X_old,Dataset,gra,para.stepsize);
            end
        end
        X_0=para.W*X;
    end