% clc; clear; close all;
%% initial workspace
rng('default')
rng('shuffle')
linewidth = 2.5;
fontsize = 14;
MarkerSize = 10;

%% setup hyperparameters
para.dimx=20;
para.node=100;
para.bs=ones(para.node,1)*400;%round(randi([10,500],para.node,1));
para.iter=600;
para.repeat=1;

K=sum(para.bs);

%% setup data

% data.features=[randn(para.dimx-1,K);ones(1,K)];

data.features = [randn(para.dimx-1,K);ones(1,K)];
% data.labels=randi([0,1],K,1);
% data.labels(data.labels==0) = -1;
data.labels=zeros(K,1);
past=1;

for i=1:para.node
    x=rand(para.dimx,1)*40+randi([-10,10],para.dimx,1);
    current=para.bs(i);
    data.labels(past:past+current-1) = data.features(:,past:past+current-1)'*x;
    past=past+current;
end
threshold = randn(K,1)*max(data.labels);
data.labels((data.labels>0)&(data.labels>threshold)) = 1;
data.labels(data.labels>0&data.labels<threshold) = 2;
data.labels(data.labels<0&(-data.labels)>threshold) = 2;
data.labels(data.labels<0&(-data.labels)<threshold) = 1;
data.labels(data.labels==2) = -1;

data.labels=data.labels';

%% setup parameters (stepsizes)


%% setup logs
log_SGD = zeros(para.iter,2,para.repeat);
log_GD = zeros(para.iter,2,para.repeat);
log_GD_1 = zeros(para.iter,2,para.repeat);
log_VR = zeros(para.iter,2,para.repeat);
log_LGD = zeros(para.iter,2,para.repeat);
log_LSGD = zeros(para.iter,2,para.repeat);
log_Prox = zeros(para.iter,2,para.repeat);

g=@gradx;
f=@fx;
%% main iteration
for loop_i=1:para.repeat
    init_x=zeros(para.node,para.dimx);
    %Deterministic
    para.W=ones(para.node)/para.node;
    para.stepsize = 4;
    
    para.I = 1;
    para.R = 1;
    para.VR=0;
    para.mini_bs = 0.0025;
    para.localiter = para.iter;
    para.stepsize = 4*2;
    [log_SGD(:,:,loop_i),itrSGD] = Fed_PD(para,data,g,f,init_x);
    
    para.VR=0;
    para.mini_bs = 1;
    para.localiter = 8;
    para.R = 2;
    [log_GD(:,:,loop_i), itr1] = Fed_PD(para,data,g,f,init_x);
    
    para.VR=0;
    para.mini_bs = 1;
    para.localiter = 8;
    para.R = 1;
    [log_GD_1(:,:,loop_i),itr2] = Fed_PD(para,data,g,f,init_x);
    
    para.I = 100;
    para.R =1;
    para.VR=1;
    para.mini_bs = 0.0025;
    para.localiter = 2;
    [log_VR(:,:,loop_i), itrVR] = Fed_PD(para,data,g,f,init_x);
    
    para.mini_bs = 1;
    para.localiter = 8;
    para.stepsize = 4/para.localiter;
    log_LGD(:,:,loop_i) = mainiter_SGD(para,data,g,f,init_x);
    
    para.mini_bs = 0.0025;
    para.localiter = para.iter;
    para.stepsize = 4;
    log_LSGD(:,:,loop_i) = mainiter_SGD(para,data,g,f,init_x);
end

%% draw figures
para.VR=1;
para.mini_bs = 0.0025;
para.localiter = 8;
para.R = 1;
para.mu = 0.1;
log_Prox(:,:,1) = Fed_Prox(para,data,g,f,init_x);
log_GD_2 = zeros(para.iter,2,para.repeat);
para.VR=0;
para.mini_bs = 1;
para.localiter = 8;
para.R = -1;
[log_GD_2(:,:,loop_i),itr3] = Fed_PD(para,data,g,f,init_x);
%% loss
figure(1);
% idx_xfilter=cumsum(log_xFilter(:,1,R=1))*400;
idx = 0:para.iter-1;
semilogy(cumsum(itrSGD)-1, mean(log_SGD(:,2,:),3),'linestyle', ':','linewidth',linewidth,'color', 'b');
hold on
semilogy(cumsum(itr1)-1, mean(log_GD(:,2,:),3),'linestyle', '-.','linewidth',linewidth,'color', 'b');
hold on
semilogy(cumsum(itr2)-1, mean(log_GD_1(:,2,:),3),'linestyle', '--','linewidth',linewidth,'color', 'b');
hold on
plot(cumsum(itr3)-1, mean(log_GD_2(:,2,:),3),'linestyle', '-','linewidth',linewidth,'color', 'g');
hold on
plot(cumsum(itrVR)-1, mean(log_VR(:,2,:),3),'linestyle', '-','linewidth',linewidth,'color', 'b');
hold on
plot((0:para.iter-1), mean(log_LGD(:,2,:),3),'linestyle', '--','linewidth',linewidth,'color', 'r');
hold on
plot((0:para.iter-1), mean(log_LSGD(:,2,:),3),'linestyle', ':','linewidth',linewidth,'color', 'r');
hold on
plot((0:para.iter-1), mean(log_Prox(:,2,:),3),'linestyle', '-.','linewidth',linewidth,'color', 'k');
hold off
xl = xlabel('Communication r','FontSize',fontsize,'interpreter','latex');
yl = ylabel('$\Vert\nabla f(x^r_0) \Vert^2$','FontSize',fontsize,'interpreter','latex');
le = legend('FedPD-SGD','FedPD-GD p=0.5','FedPD-GD p=0','FedPD-GD p=(T-r)/T','FedPD-VR','FedAvg-GD','FedAvg-SGD','FedProx-VR');%'EXTRA','xFilter','DSGD','GNSD' );

%%

figure(2);
% idx_xfilter=cumsum(log_xFilter(:,1,R=1))*400;
semilogy((0:para.iter-1)*600*100, mean(log_SGD(:,2,:),3),'linestyle', ':','linewidth',linewidth,'color', 'b');
hold on
semilogy((0:para.iter-1)*8*40000, mean(log_GD(:,2,:),3),'linestyle', '-.','linewidth',linewidth,'color', 'b');
hold on
semilogy((0:para.iter-1)*8*40000, mean(log_GD_1(:,2,:),3),'linestyle', '--','linewidth',linewidth,'color', 'b');
hold on
plot((0:para.iter-1)*(40000/100+8*100*2*3), mean(log_VR(:,2,:),3),'linestyle', '-','linewidth',linewidth,'color', 'b');
hold on
plot((0:para.iter-1)*8*40000, mean(log_LGD(:,2,:),3),'linestyle', '--','linewidth',linewidth,'color', 'r');
hold on
plot((0:para.iter-1)*600*100, mean(log_LSGD(:,2,:),3),'linestyle', ':','linewidth',linewidth,'color', 'r');
hold on
plot((0:para.iter-1)*(40000+2*3*100*8), mean(log_Prox(:,2,:),3),'linestyle', '-.','linewidth',linewidth,'color', 'k');
hold off
xl = xlabel('Communication r','FontSize',fontsize,'interpreter','latex');
yl = ylabel('$\Vert\nabla f(x^r_0) \Vert^2$','FontSize',fontsize,'interpreter','latex');
le = legend('FedPD-SGD','FedPD-GD R=2','FedPD-GD R=1','FedPD-VR','FedAvg-GD','FedAvg-SGD','FedProx-VR');%'EXTRA','xFilter','DSGD','GNSD' );
