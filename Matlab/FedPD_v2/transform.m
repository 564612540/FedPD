function [D]=transform(data,node,bs,minibs)
    mini_bs=round(minibs*bs(1));
    idx=randperm(bs(1),min(mini_bs,bs(1)));
    s.features=data.features(:,idx);
    s.labels=data.labels(idx);
    D=[s];
    past=bs(1);
    for i=2:node
        mini_bs=round(minibs*bs(i));
        idx=randperm(bs(i),min(mini_bs,bs(i)));
        s.features=data.features(:,past+idx);
        s.labels=data.labels(past+idx);
        D(i)=s;
        past=past+bs(i);
    end
end