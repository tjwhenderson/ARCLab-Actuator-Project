function [mu,sigma] = Gaussian_Mix(classes)
%% Artificial Actuator Data-Driven Model with "Learning from incomplete data" implementation

%dbstop if naninf

% Info:
% Inputs:

% Classes: PZT
%          DEA
%          IPMC
%          SMA
%          SCP
%          SFA
%          TSA
%          EAP
%classes = ["PZT", "DEA", "IPMC", "SMA", "SCP", "SFA", "TSA", "EAP"];

%% Import dataset
%X = GetGoogleSpreadsheet('1dELDmS4YZjsuyiIjc9m4ZyThvj9uYU64wKo6hq8xEJA');
X = GetGoogleSpreadsheet('12O8l2jjX-HykoBuSqWtsS3sa8CICr9UZ5F4zTUMKKG0');
X(1,:) = [];
X(:,1) = [];
Y = X(:,1);
for i = 1:size(X,1)
    class = X(i,1);
    onehotlbl = strcmp(class, classes);
    [~,lbl(i)] = max(onehotlbl);   
end
X = str2double(X);
X(:,1) = [];
%X = [X(:,2) X(:,3)];
X = [X(:,2) X(:,3) X(:,4)];
X = X';
X_org = X;
% Normalize data
%X = log(X);
%X = zscore(X);
data_mean = nanmean(X,2);
data_std = nanstd(X,[],2);
X = bsxfun(@minus,X,nanmean(X,2));
X = bsxfun(@rdivide,X,nanstd(X,[],2));
%X = (X - min(min(X)))./(max(max(X)) - min(min(X)));   
% Set 'm' to the number of data points.
m = size(X,2);
    
% Permute each feature into observed and unobserved
R = ~isnan(X);
[~,I] = sort(R,1,'descend');

for i = 1:size(R,2)
    X_new(:,i) = X(I(:,i),i);
end

% Implement Matrix completion via nuclear norm
X_NN = X;
X_NN(isnan(X_NN)) = 0;
cd NNMatrixCompletion
[CompletedMat, ier] = MatrixCompletion(X_NN.*R,R,3000,'nuclear', 10, 1e-8, 0); 
cd ..


%% Implement Modified EM

k = length(classes);  % The number of clusters.
n = size(X,1);  % Number of features

% Find rows where all elements observed
comp = X_new;
comp(:,any(isnan(comp),1))=[];

% Randomly select k data points to serve as the initial means.
indeces = randperm(size(comp,2));
mu = comp(:,indeces(1:k));

% Use identity matrix as the initial covariance for each submatrix for each cluster.
e = 0.0075*ones(n); % epsilon to be added to the diagonal entries of the covariance matrix at each iter
sigma = [];
for j = 1:k
    sigma{j} = abs(cov(comp'));
    NN_sigma{j} = abs(cov(CompletedMat'));
end

% Assign equal prior probabilities to each cluster.
phi = ones(1,k)*(1/k);

%% Run Modified Expectation Maximization

% Loop until convergence.
for iter = 1:5000

    fprintf('EM Iteration %d\n', iter);

    %% Expectation

    % Find E[z_ij|...] for observed dimensions of x_i

        % For each cluster
        for j = 1:k
            % For each data point
            for i = 1:m
                % Evaluate the Gaussian for the data vector 'i' for cluster 'j'.
                idx = sum(~isnan(X_new(:,i)));
                  
                pdf(i,j) = det(sigma{j}(1:idx,1:idx))^(-1/2) * exp((-1/2)*(X_new(1:idx,i)-mu(1:idx,j))'*inv(sigma{j}(1:idx,1:idx))*(X_new(1:idx,i)-mu(1:idx,j)));
            end
        end

        % Multiply each pdf value by the prior probability for cluster.
        pdf_w = bsxfun(@times, pdf, phi);

        % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
        h = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));

        % Find E[z_ij*x_i^m|...] & E[z_ij*x_i^m*x_ij^mT|...]
        x_m_hat = zeros(size(X_new,2),n,k);
        moment2 = cell(1,size(X_new,2));
        for j = 1:k
            for i = 1:m
                idx = sum(~isnan(X_new(:,i)));
                temp = mu(idx+1:end,j) + sigma{j}(idx+1:end,1:idx)*((sigma{j}(1:idx,1:idx))\(X_new(1:idx,i)-mu(1:idx,j)));
                if isempty(temp)
                    x_m_hat(i,:,j) = 0;
                else
                    x_m_hat(i,n-length(temp)+1:end,j) = temp;
                end
                %moment1(i,j) = h(i,j)*x_m_hat(i,j);
                temp2 = h(i,j)*(sigma{j}(idx+1:end,idx+1:end) - sigma{j}(idx+1:end,1:idx)*(sigma{j}(1:idx,1:idx)\sigma{j}(1:idx,idx+1:end)) + x_m_hat(i,:,j)*x_m_hat(i,:,j)');
                if isempty(temp2)
                    moment2{i}{j} = {};
                else
                    moment2{i}{j} = temp2;
                end
            end    
        end

    %% Maximization

    % Store the previous means.
    prevMu = mu; 
    prevSigma = sigma;

    % For each of the clusters...
    for j = 1:k

        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(h(:,j), 1);

        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        X_mu_temp = X_new;
        X_mu_temp(isnan(X_mu_temp)) = 0;
        X_mu = X_mu_temp + x_m_hat(:,:,j)';
        mu(:,j) = (h(:,j)'*X_mu')./sum(h(:,j));  
            
        % Compare to mean imputation
        X_new_mean = mean(comp,2);
        X_mu_temp = X_new;
        for dim = 1:n
            X_mu_temp(isnan(X_mu_temp(dim,:))) = X_new_mean(dim,1);
        end
        MI_X_mu = X_mu_temp;
        MI_mu(:,j) = (h(:,j)'*MI_X_mu')./sum(h(:,j));
            
        % Compare to Nuclear Norm
        NN_mu(:,j) = (h(:,j)'*CompletedMat')./sum(h(:,j));
            
        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example.
        sigma_k = zeros(n, n);
        MI_sigma_k = zeros(n, n);
        NN_sigma_k = zeros(n,n);
        for i = 1:m
            idx = sum(~isnan(X_new(:,i)));
            if idx == n
                exp_mat = X_new(1:idx,i)*X_new(1:idx,i)';
            else
                exp_mat = [X_new(1:idx,i)*X_new(1:idx,i)', X_new(1:idx,i)*x_m_hat(i,idx+1:n,j);
                           x_m_hat(i,idx+1:n,j)'*X_new(1:idx,i)', moment2{i}{j}];
            end
            sigma_k = sigma_k + h(i,j)*(exp_mat - mu(:,j)*mu(:,j)');
            MI_sigma_k = MI_sigma_k + h(i,j)*(MI_X_mu(:,i) - MI_mu(:,j))*(MI_X_mu(:,i) - MI_mu(:,j))';
            NN_sigma_k = NN_sigma_k + h(i,j)*(CompletedMat(:,i) - NN_mu(:,j))*(CompletedMat(:,i) - NN_mu(:,j))';
        end
        sigma{j} = diag(diag(abs(sigma_k ./ sum(h(:,j))) + e));
        MI_sigma{j} = diag(diag(abs(MI_sigma_k./sum(h(:,j))) + e));
        NN_sigma{j} = diag(diag(abs(NN_sigma_k./sum(h(:,j))) + e));
    end

    % Check for convergence.
    norm(mu-prevMu)
    if norm(mu-prevMu)<0.0001
        break
    end  
end

%% Classification
for j = 1:k
    for i = 1:m
        idx = sum(~isnan(X_new(:,i)));
        liklihood(i,j) = det(sigma{j}(1:idx,1:idx))^(-1/2) * exp((-1/2)*(X_new(1:idx,i)-mu(1:idx,j))'*inv(sigma{j}(1:idx,1:idx))*(X_new(1:idx,i)-mu(1:idx,j)));
    end
end
[~,pred_class] = max(liklihood,[],2);

c = 0;
d = 0;
for l = 1:m
    if pred_class(l) == lbl(l)
        c = c + 1;
    end
end
accuracy = c/m;


% Display a scatter plot of the two distributions.
figure(2);
hold off;
%plot(X(2,:), X(3, :), 'bo');
plot(X(1,:),X(2,:),'o');
hold on;

set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-2, 5, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
for m = 1:length(classes)
    z{m} = gaussianND(gridX, mu(1:2, m)', sigma{m}(1:2,1:2));
end

for n = 1:length(classes)
    NN_z{n} = gaussianND(gridX, NN_mu(1:2, n)', NN_sigma{n}(1:2,1:2));
end

% Reshape the responses back into a 2D grid to be plotted with contour.
for m = 1:length(classes)
    Z{m} = reshape(z{m}, gridSize, gridSize);
end

for n = 1:length(classes)
    NN_Z{n} = reshape(NN_z{n}, gridSize, gridSize);
end

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, real(Z{1}));

hold off;

title('Original Data and Estimated PDFs');
xlabel('strain');
ylabel('stress');

figure(3);
hold off;
plot(X(1, :), X(2, :), 'bo');
hold on;

set(gcf,'color','white') % White background for the figure.

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, real(NN_Z{1}));

hold off;

title('Original Data and Estimated PDFs - NN');
xlabel('strain');
ylabel('stress');

Labels = [lbl' pred_class];



end