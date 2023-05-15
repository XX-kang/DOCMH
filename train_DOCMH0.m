function [W1,W2,CC,HH,BB,DD,FF] = train_DOCMH0(XTrain_new,YTrain_new,LTrain_new,param)
    % parameters
    ita=param.ita;
    alpha = param.alpha;
    beta = param.beta;
    gamma = param.gamma;   
    xi1 = param.xi1;
    xi2 = param.xi2;
    r = param.nbits;
    maxiter = param.maxiter;
     
    %initialize
    X1_t = XTrain_new';
    X2_t = YTrain_new';
    L_t = LTrain_new';
    [doI,nt]=size(X1_t);
    [doT,~]=size(X2_t);
    [c,~]=size(L_t);
       
    R = randn(r, r);
    [U11, ~, ~] = svd(R);
    R = U11(:, 1:r);
    B_t= sign(randn(r, nt)); B_t(B_t==0) = -1; 
    dri1 = zeros(r,1);
    dri2 = zeros(r,1);
    e=ones(1,nt);
    
    %M
    X_t = [X1_t;X2_t];
    M=sylvester(X_t*X_t',ita*(L_t*L_t'+gamma*eye(c)),(1+ita)*X_t*L_t');   
    Y_t=M*L_t;
    %Y_t= L_t;
    U_t=NormalizeFea(Y_t,1);
    
    for jj=1:maxiter %iteration     
        %G
        G= (B_t*Y_t')/(Y_t*Y_t'+(gamma/alpha)*eye(size(Y_t,1)));
        %G=ones(r,size(Y_t,1));
        
        %V
        O = R'*B_t+2*beta*r*(B_t*U_t')*U_t...
            -beta*r*(B_t*ones(nt,1))*ones(1,nt);
  
        OJO = O*O' -1/nt*(O*ones(nt,1)*(ones(1,nt)*O'));
        [~,Omaga,QQ] = svd(OJO); clear OJO;
        idx = (diag(Omaga)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (O'-1/nt*ones(nt,1)*(ones(1,nt)*O')) *  (Q / (sqrt(Omaga(idx,idx))));
        P_ = orth(randn(nt,r-length(find(idx==1))));
        V_t = sqrt(nt)*[Q Q_]*[P P_]';
        
         %W
        W1= ((B_t- dri1*e)*X1_t')/(X1_t*X1_t'+ (gamma/xi1)*eye(doI));
        W2= ((B_t- dri2*e)*X2_t')/(X2_t*X2_t'+ (gamma/xi2)*eye(doT));

        %Add drift
        dri1= xi1*(W1*X1_t-B_t)*e'/size(e,2);
        dri2= xi2*(W2*X2_t-B_t)*e'/size(e,2);
        
        %B
        B_t = sign(R*V_t+alpha*G*Y_t+2*beta*r*(V_t*U_t')*U_t...
            -beta*r*(V_t*ones(nt,1))* ones(1,nt)...
            +xi1*(W1*X1_t-dri1*e)+xi2*(W2*X2_t-dri2*e));
                  
        %R
        [U2,~,V2]=svd(V_t*B_t');
        if c<r
            R=U2(1:r,:)'*V2;
        elseif c>r
            R=U2*V2(:,1:r)';
        end
    end
      
    C1 = X_t*X_t';
    C2 = L_t*L_t';
    C3 = X_t*L_t';
    C4 = V_t*U_t';
    C5 = V_t*ones(nt,1);
    C6 = B_t*Y_t';
    C7 = Y_t*Y_t';
    H1 =(B_t- dri1*e)*X1_t';
    H2 =(B_t- dri2*e)*X2_t';
    F1 = X1_t*X1_t';
    F2 = X2_t*X2_t';
    D1 = dri1;
    D2 = dri2;
    
    CC{1,1} = C1;
    CC{1,2} = C2;
    CC{1,3} = C3;
    CC{1,4} = C4;
    CC{1,5} = C5;
    CC{1,6} = C6;
    CC{1,7} = C7;
    HH{1,1} = H1;
    HH{1,2} = H2;
    FF{1,1} = F1;
    FF{1,2} = F2;
    DD{1,1} = D1;
    DD{1,2} = D2;
    BB{1,1} = B_t;
  end
