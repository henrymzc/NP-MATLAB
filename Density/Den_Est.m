clear;
clc;
cd 'D:\Duke\ECON881-06_NON_PARAM\MORG'
options=optimset('MaxFunEvals', 1000, 'MaxIter', 1000,'TolX',1e-10,'TolFun',1e-10, 'Display', 'off' );

CPI = [172.2/72.6 ;     %2000/1979
       172.2/124.0];     %2000/1989

for year = 79:10:89
    filename = sprintf('data%d.csv',year);
    wage = csvread(filename,1,0);
    if year == 79
        log_wage =  log(CPI(1,1) * wage);
    end
    if year == 89
        log_wage =  log(CPI(2,1) * wage);
    end
    n = size(log_wage,1);
    
    %grid
    grid = 0.1;
    grid_x =(min(log_wage):grid:max(log_wage))';
    
    %descriptive statistics
    mu_hat = mean(log_wage);
    sigma2_hat = var(log_wage,1);
    sigma_hat = std(log_wage,1);
    
    %F_hat(Parametric)
    fhat_P = normpdf(grid_x,mu_hat,sigma_hat);
    
    %Non-Parametric
    fhat_NP = NaN(size(grid_x,1),1);
    h_rot = 1.06 * sigma_hat * size(log_wage,1)^(-1/5); %Optimal bandwidth
    for x_index = 1:size(grid_x,1)
        fhat_NP(x_index) = (1/h_rot) *mean(normpdf((grid_x(x_index)-log_wage)/h_rot)); %second order gaussian
    end
    
    %Plug-in Estimator
    fhat_NP_plugin = NaN(size(grid_x,1),1);
    
    fhat_fo = diff(fhat_NP)/grid;   % first derivative
    fhat_so = diff(fhat_fo)/grid;   % second derivative
    A = 1;
    B = (2 * sqrt(pi))^ (-1);
    
    %Integral approximation
    Int = 0 ; 
    for index = 1:(size(grid_x,1)-2)
        Int = Int + (fhat_so(index,1)^2) * grid;
    end
    
    h_plugin =  size(log_wage,1)^(-1/5) * (B / (A^2 * Int)) ^ (1/5);
    for x_index = 1:size(grid_x,1)
        fhat_NP_plugin(x_index) = (1/h_plugin) *mean(normpdf((grid_x(x_index)-log_wage)/h_plugin)); %second order gaussian
    end
    
    %{
        %Cross Validation
      
        %Grid search
        %Running time of one point: 3 mins
        grid_h = (0.1:0.005:0.13)';
        
        %Grid of objective function
        CV_grid = NaN(size(grid_h,1),1);
        
        %Start grid search
        for h_index = 1:size(grid_h,1)
            h_index
            CV_grid(h_index,1) = CV(n,log_wage,grid_h(h_index,1)) ;
        end
        
        %Find the minimum CV index
        [M, I] = min(CV_grid);
        
        %optimal h based on Cross Validation
        h_cv = grid_h(I);
        fhat_NP_CV = NaN(size(grid_x,1),1);
        for x_index = 1:size(grid_x,1)
            fhat_NP_CV(x_index) = (1/h_cv) *mean(normpdf((grid_x(x_index)-log_wage)/h_cv)); %second order gaussian
        end
    %}
    
    %Plot 1979
    if year == 79
        %plot Parametric Estimator
        %plot(grid_x,fhat_P,'red')
        %hold on

        %Plot Rule of Thumb Estimator
        %plot(grid_x,fhat_NP,'red')
        %hold on 

        %Plot Plug in Estimator
        plot(grid_x,fhat_NP_plugin,'red')
        hold on 

        %Plot Cross Validation Estimator
        %plot(grid_x,fhat_NP_CV,'green')
    end
    
    %Plot 1989
    if year == 89
        %plot Parametric Estimator
        %plot(grid_x,fhat_P,'black')
        %hold on

        %Plot Rule of Thumb Estimator
        %plot(grid_x,fhat_NP,'black')
        %hold on 

        %Plot Plug in Estimator
        plot(grid_x,fhat_NP_plugin,'black')
        %hold on 

        %Plot Cross Validation Estimator
        %plot(grid_x,fhat_NP_CV,'green')
    end
  
    axis([1 4.5 0 1.5])
    title('Rule of Thumb Estimator')
    legend('1979(1.06\sigma n^{-1/5}; h=0.0520)','1989(1.06\sigma n^{-1/5}; h=0.0587)')
    xlabel('Female log wage')
    ylabel('Density')
        
end

function value = CV(n,x,h)

    %double summation without for loop
    %Closed form of CV
    %f1 = @(x,i,j,h) (2 * sqrt(pi)) ^ (-1) .* exp(-1 .* bsxfun(@minus, x(i),x(j)').^2 ./(4*h^2));
    %sum1 = sum(reshape(f1(x, 1:n, 1:n,h), 1, []));
    %Exceed the maximum limit of array
    sum1 = 0;
    tic
    for i = 1:n
        j = 1:n;
        sum1 = sum1 + sum((2 * sqrt(pi)) ^ (-1) .*h .* exp(-1 .* (x(i)-x(j)).^2 ./(4*h^2)));
    end
    toc
    tic
    sum2 = 0;
    for i = 1:n
        j = setdiff(1:n,i);
        sum2 = sum2 + sum(normpdf((x(j,1)-x(i,1))./h));
    end
    toc
    %f2 = @(x,i,j,h)  normpdf(bsxfun(@minus, x(j), x(i)')./h);
    %sum2 = sum(reshape(f2(x, 1:n, 1:n,h), 1, []));
    
    value = n ^(-2) * h ^(-2) * sum1 - (2/(n*(n-1)*h)) * sum2;   
          
end


