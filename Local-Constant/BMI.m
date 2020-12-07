clear;
clc;
cd 'D:\Duke\ECON881-06_NON_PARAM\PS2(Local_constant_estimator)'

% Read in data
data = csvread('CDC_data_males.csv',1,0);
age = data(:,1);
weight = data(:,2);
height = data(:,3);
n = size(data,1);
bmi = weight./(height.^2);
grid_age = (20:0.1:70)';
X_20_70 = [ones(size(grid_age,1),1),grid_age,grid_age.^2];

%OLS estimation

X = [ones(n,1),age,age.^2];
[b,bint,r] = regress(bmi,X);
var(r)
%asym. var of OLS estimator
asyvar = (X'*X/n)^(-1) * var(r);

%OLS estimator E[Y|X=x]
y_ols = @(x) (x * b);
y_hat_ols = y_ols(X_20_70);

%OLS confidence interval
var_y_ols = zeros(size(grid_age,1),1);
y_ub_ols = zeros(size(grid_age,1),1);
y_lb_ols = zeros(size(grid_age,1),1);
%Delta Method
for i = 1:size(grid_age,1)
    var_y_ols(i,1) = (X_20_70(i,:) * asyvar *  X_20_70(i,:)')/n;
    y_ub_ols(i,1) = y_hat_ols(i,1) + 1.64 * sqrt(var_y_ols(i,1));
    y_lb_ols(i,1) = y_hat_ols(i,1) - 1.64 * sqrt(var_y_ols(i,1));
end

plot(grid_age,y_hat_ols,'black') 
hold on
plot(grid_age,y_ub_ols,'--k')
hold on
plot(grid_age,y_lb_ols,'--k')
    title('Quadratic Specification')
    xlabel('Age')
    ylabel('BMI')
    legend('E[BMI|Age]','90% CI','location','southeast')



    
%NP estimation

%rule of thumb
h_rot = 1.06 * std(age) * n^(-1/5);

y_hat_np_rot = zeros(size(grid_age,1),1);
for x_index = 1:size(grid_age,1)
    w = normpdf((grid_age(x_index)-age)/h_rot) / sum(normpdf((grid_age(x_index)-age)/h_rot));
    y_hat_np_rot(x_index) = bmi' * w;
end



%Cross-Validation
grid_h = (1:0.1:3)';

Obj_cv = inf(size(grid_h,1),1);
for i = 1:size(grid_h,1)
    h = grid_h(i)
    Obj_cv(i) = CV(n, age, bmi,h);
end
[M, I] = min(Obj_cv);
h_cv = grid_h(I);
y_hat_np_cv = zeros(size(grid_age,1),1);
for x_index = 1:size(grid_age,1)
    w = normpdf((grid_age(x_index)-age)/h_cv) / sum(normpdf((grid_age(x_index)-age)/h_cv));
    y_hat_np_cv(x_index) = bmi' * w;
end


%AIC
Obj_AIC = inf(size(grid_h,1),1);
for i = 1:size(grid_h,1)
    h = grid_h(i)
    Obj_AIC(i) = AIC(n, age, bmi,h);
end
[M, I] = min(Obj_AIC);
h_AIC = grid_h(I);
y_hat_np_aic = zeros(size(grid_age,1),1);
for x_index = 1:size(grid_age,1)
    w = normpdf((grid_age(x_index)-age)/h_AIC) / sum(normpdf((grid_age(x_index)-age)/h_AIC));
    y_hat_np_aic(x_index) = bmi' * w;
end
plot( grid_age, y_hat_np_rot,'r')
hold on
plot( grid_age, y_hat_np_cv,'g')
hold on
plot( grid_age, y_hat_np_aic,'b')
    title('Nonparametric Estimator')
    xlabel('Age')
    ylabel('BMI')
    legend('rule of thumb(1.06\sigma n^{-1/5}:h=4.47)','Cross-validation(h=1.8)','AIC(h=2.9)')



%Confidence Interval(%undersmooting); us: undersmoothing
h_us =  1.06 * std(age) * n^(-1/4); 
y_hat_np_us = zeros(size(grid_age,1),1);
y_hat_np_us_ub = zeros(size(grid_age,1),1);
y_hat_np_us_lb = zeros(size(grid_age,1),1);
y2_hat = zeros(size(grid_age,1),1); %E[Y^2|X=x]
sigma2_hat = zeros(size(grid_age,1),1);
fhat_NP = zeros(size(grid_age,1),1);
B = (2 * sqrt(pi)) ^ (-1);
bmi2 = bmi.^2;

for x_index = 1:size(grid_age,1)
    w = normpdf((grid_age(x_index)-age)/h_us) / sum(normpdf((grid_age(x_index)-age)/h_us));
    y_hat_np_us(x_index) = bmi' * w;
    y2_hat(x_index) = bmi2' * w;
    sigma2_hat(x_index) = y2_hat(x_index) - y_hat_np_us(x_index)^2;
    fhat_NP(x_index) = (1/h_rot) *mean(normpdf((grid_age(x_index)-age)/h_rot));
    y_hat_np_us_ub(x_index) = y_hat_np_us(x_index) + 1.64 * sqrt(sigma2_hat(x_index)*B/(fhat_NP(x_index)*n*h_us));
    y_hat_np_us_lb(x_index) = y_hat_np_us(x_index) - 1.64 * sqrt(sigma2_hat(x_index)*B/(fhat_NP(x_index)*n*h_us));
end

plot(grid_age,y_hat_np_us,'black') 
hold on
plot(grid_age,y_hat_np_us_ub,'--k')
hold on
plot(grid_age,y_hat_np_us_lb,'--k')
    title('Under Smoothing(h=2.9345)')
    xlabel('Age')
    ylabel('BMI')
    legend('E[BMI|Age]','90% CI','location','southeast')



function value = CV(n,age,bmi,h)
    g_leave_one_out = zeros(n,1);
    for i = 1:n
        age_i = age;
        bmi_i = bmi;
        bmi_i(i) =[]; %leave one out data(bmi)
        age_i(i) = []; %leave one out data(age)
        w_i = normpdf((age(i)-age_i)/h) / sum(normpdf((age(i)-age_i)/h));
        g_leave_one_out(i) = bmi_i' * w_i;
    end
    value = mean((bmi - g_leave_one_out).^2);
end

function value = AIC(n,age,bmi,h)
    First_term = log(CV(n,age,bmi,h));
    diag = NaN(n,1);
    for i = 1:n
        diag(i) = normpdf(0) / sum(normpdf((age(i)-age)/h));
    end
    trace = sum(diag);
    Second_term = (1 + trace/n) / (1 - (2+trace)/n);
    value = First_term + Second_term;
end

