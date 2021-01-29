function logs=FirstOrder(para,data,alpha,init_x,gra,fnc,WX,WG)
    %% Distributed Gradient Descent with Dynamic Weight
    X=init_x;
    Y=zeros(size(X));
    X_O=zeros(size(X));
    logs=zeros(para.iter,3);
    
    %% main iterations
    for i=1:para.iter
        Dataset=transform(data,para.node,para.bs,1);
        logs(i,:)=ComputeLog(X,Dataset,gra,fnc,1);
        Dataset=transform(data,para.node,para.bs,para.mini_bs);
        G=getgrad(X,Dataset,gra,alpha.grad/sqrt(i));
        ss=alpha.ss;
        if(para.GT)
            if(i==1)
                G_O=G;
                Y=G;
            end
            [Y,G_O]=GT(Y,G,G_O,WG);
            [X]=updateX(X,WX,Y,ss);
        elseif(para.EXTRA)
            if(i==1)
                G_O=G;
                X_O=X;
                Y=G;
            else
                [Y,X_O,G_O]=EXTRA(X,X_O,G,G_O,WG);
            end
            [X]=updateX(X,WX,Y,ss);
        elseif(para.xFilter)
            G=G*5;
            alpha_grad=alpha.grad*5;
            cons=0.01;
            logs(i,1)=0;
            if(i==1)
                logs(i,1)=1;
                j=1;
                [m,n]=size(WG);
                mu=zeros(m,1);
                d=X-G-alpha_grad/sqrt(j)*WG'*mu;
                R=alpha_grad/sqrt(j) * (alpha.ss * cons * WX) + eye(n);
                t = 2 / (min(eig(R)) + max(eig(R)));
                rho = (1-1/cond(R))/(1+1/cond(R));
                Q=ceil(-1/4*log10((sqrt(j)^2/(cond(R)*alpha_grad^2))^2/(16+128*n*max(max(eig(R/alpha_grad*sqrt(j))),1)))*sqrt(cond(R)));
                U=zeros(size(G));
                ss_1=0;
                [X,Q,U,ss_1]=Chebyshev(R, d, X, Q, U, ss_1, t, rho);
            else
                if (Q==0)
                    logs(i,1)=1;
                    j=j+1;
                    mu=mu+alpha.ss * cons*WG*X;
                    d=X-G-alpha_grad/sqrt(j)*WG'*mu;
                    R=alpha_grad/sqrt(j) * (alpha.ss*cons * WX) + eye(n);
                    t = 2 / (min(eig(R)) + max(eig(R)));
                    rho = (1-1/cond(R))/(1+1/cond(R));
                    Q=ceil(-1/4*log10((sqrt(j)/(cond(R)*alpha_grad))^2/(16+128*n*max(max(eig(R/alpha_grad*sqrt(j))),1)))*sqrt(cond(R))*2);
                    U=zeros(size(G));
                end
                [X,Q,U,ss_1]=Chebyshev(R, d, X, Q, U, ss_1, t, rho);
            end
        else
            [X]=updateX(X,WX,G,ss);
        end
    end
end

function [Y,X_O,G_O]=EXTRA(X_N,X,G_N,G,W)
    %% EXTRA
    Y=W*X+(G_N-G);
    G_O=G_N;
    X_O=X_N;
end

function [Y_N,G_O]=GT(Y,G_N,G,W)
    %% Gradient Tracking
    Y_N=W*Y+(G_N-G);
    G_O=G_N;
end

function [X_N, Q_N, U_N, ss_N] = Chebyshev(W, d, X, Q, U, ss, t, rho)
    Q_N=Q-1;
    if (any(U)==0)
        X_N = (eye(size(W))-t*W) * X + t*d;
        U_N = X;
        ss_N = 2;
    else
        ss_N=4/(4-rho^2*ss);
        X_N=ss_N*(eye(size(W))-t*W) * X +(1-ss_N)*U+t*ss_N*d;
        U_N=X;
    end
end