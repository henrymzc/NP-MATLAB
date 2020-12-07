clear;
clc;
cd 'D:\Duke\ECON881-06_NON_PARAM\PS3'

S = 10000; %Simulated Dataset number
mu = [0 0];
Sigma = [1 0; 0 1];
n = 100;
C_set =  [0, 0.001, 0.01, 0.1];

Beta = zeros(S,6); %coefficent
% Column 1: cn = 0
% Column 2: cn = 0.001
% Column 3: cn = 0.01 
% Column 4：cn = 0.1
% Column 5: OLS Y on X and a constant
% Column 6: OLS Y on X , X^2 and a constant

%Rejection result
Rej_m2 = NaN(S,4) ; % Ho:\beta == -2
Rej_m1 = NaN(S,4);  % Ho:\beta == -1
Rej_m05 = NaN(S,4); % Ho:\beta == -0.5
Rej_0 = NaN(S,4);   % Ho:\beta == 0
Rej_05 = NaN(S,4);  % Ho:\beta == 0.5
Rej_1 = NaN(S,4);   % Ho:\beta == 1
Rej_2 = NaN(S,4);   % Ho:\beta == 2

fzcount = 1;
%Simuation Starts Here
for i = 1:S
    %Generate Random Data
    Data = mvnrnd(mu, Sigma, n);
    X = Data(:,1);
    U = Data(:,2);
    Z = X.^2;
    Y = Z + U;  %g(z) = z; beta = 0
    
    % Robinson's partial linear estimator
        EX_Z = Local_Constant(X, Z);
        EY_Z = Local_Constant(Y, Z);
        %plot (Z , EY_Z)
        fZ = Kernal_Density(Z);
        X_hat = X - EX_Z;
        Y_hat = Y - EY_Z;
        fzcount = min(fzcount,min(fZ));
        % For each trimming bandwidth
        for j = 1:4
            cn =  C_set(j);
            
            %Trimming
            X_hat_c = X_hat;
            Y_hat_c = Y_hat;
            TF1 = fZ(:,1) <= cn;
            X_hat_c(TF1,:) = [];
            Y_hat_c(TF1,:) = [];
                        
            [b_PL,bint,r] = regress(Y_hat_c,X_hat_c);
            Beta(i,j) = b_PL;
            
            %Hypothesis test
            sigma = sum(r.^2)./n;
            Sigma_X = sum(X_hat_c.^2)./n;            
            Se = sqrt(sigma/(Sigma_X*n));          
            Rej_m2(i,j) = abs(b_PL - (-2)) > 1.96 * Se;
            Rej_m1(i,j) = abs(b_PL - (-1)) > 1.96 * Se;
            Rej_m05(i,j) = abs(b_PL - (-0.5)) > 1.96 * Se;
            Rej_0(i,j) = abs(b_PL - 0) > 1.96 * Se;
            Rej_05(i,j) = abs(b_PL - 0.5) > 1.96 * Se;
            Rej_1(i,j) = abs(b_PL - 1) > 1.96 * Se;
            Rej_2(i,j) = abs(b_PL - 2) > 1.96 * Se;
        end

    % OLS regression of Y on X and a constant 
    X1 =   [ones(n,1),X];
    [b1,bint1,r1] = regress(Y,X1);
    Beta(i,5) = b1(2,1);
    
    % OLS regression of Y on X and X^2 and a constant
    X2 =   [ones(n,1),X,X.^2];
    [b2,bint2,r2] = regress(Y,X2);
    Beta(i,6) = b2(2,1);  
end
Bias = mean(Beta)
Bias2 = Bias.^2
Var = var(Beta)
MSE = Var + Bias2

M_Rej_m2 = mean(Rej_m2)
M_Rej_m1 = mean(Rej_m1)
M_Rej_m05 = mean(Rej_m05)
M_Rej_0 = mean(Rej_0)
M_Rej_05 = mean(Rej_05)
M_Rej_1 = mean(Rej_1)
M_Rej_2 = mean(Rej_2)


function value = Local_Constant(W, Z)
    n = size(W,1);
    h_rot = 0.2 * std(Z) * n^(-1/5);
    value = zeros(n,1);
    for i = 1:n
        w = normpdf((Z(i)-Z)/h_rot) / sum(normpdf((Z(i)-Z)/h_rot));
        value(i,1) = W' * w;
    end
end

function value = Kernal_Density(Z)
    n = size(Z,1);
    h_rot = 0.2 * std(Z) * n^(-1/5);
    value = zeros(n,1);
    for i = 1:n
        value(i) = (1/h_rot) *mean(normpdf((Z(i)-Z)/h_rot));
    end
end