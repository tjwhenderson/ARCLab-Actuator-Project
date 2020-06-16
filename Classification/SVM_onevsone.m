clear all
close all
% Use GetGoogleSpreadsheet function to load in data
X = GetGoogleSpreadsheet('1dELDmS4YZjsuyiIjc9m4ZyThvj9uYU64wKo6hq8xEJA');

% Define actuator class names
classes = ["PZT", "DEA", "IPMC", "SMA", "SCP", "SFA", "TSA", "EAP"];

% Remove first column and row from data
X(1,:) = [];
X(:,1) = [];

% Define new first column as labels
Y = X(:,1);

for i = 1:size(X,1)
    class = X(i,1); % extract each class 
    onehotlbl = strcmp(class, classes); % one-hot encode matching class from classes list
    [~,lbl(i)] = max(onehotlbl); % save class label index for each row
end

X = str2double(X); % convery string entries to double format
X(:,1) = []; % remove first column
X = [X(:,2) X(:,3)]; % extract 3nd and 3rd cloumns corresponding to stress and strain
X = X'; % flip rows and columns of data

%Normalize the data
X = log(X);
%X = exp(X);
%data_mean = nanmean(X,2);
%data_std = nanstd(X,[],2);
%X = bsxfun(@minus,X,nanmean(X,2));
%X = bsxfun(@rdivide,X,nanstd(X,[],2));

% Assign class name to each number label (not sure this section is
% necessary...)
PZT = X(:,lbl== 1)';
DEA = X(:,lbl == 2)';
IPMC = X(:,lbl == 3)';
SMA = X(:,lbl==4)';
SCP = X(:,lbl==5)';
SFA = X(:,lbl == 6)';
TSA = X(:,lbl == 7)';
EAP = X(:,lbl ==8)';

% Define Multivariate SVM parameters
t = templateSVM('Standardize',true,'SaveSupportVectors',true,'KernelFunction','rbf');
predictorNames = {'stress','strain'};
responseName = 'ActuatorType';
classNames = {'PZT', 'DEA', 'IPMC', 'SMA', 'SCP', 'SFA', 'TSA', 'EAP'};

% Define Model
Mdl = fitcecoc(X',Y,'Learners',t,'ResponseName',responseName,...
    'PredictorNames',predictorNames,'ClassNames',classNames);
L = size(Mdl.CodingMatrix,2); % length
sv = cell(L,1); % support vectors

% Find support vectors
for j = 1:L
    SVM = Mdl.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end

% Filled in data matrix
complete = X';

% Plot SVM figure for stress vs. strain
markers = {'ko','ro','bo','go','co','yo','mo','wo'};
for j = 1:L
    figure(j)
    hold on
    %gscatter(complete(:,1),complete(:,2),Y);
    lbl1 = find(Mdl.CodingMatrix(:,j)==1);
    lbl2 = find(Mdl.CodingMatrix(:,j)==-1);
    svs = sv{j};
    plot(svs(:,1),svs(:,2),markers{1},...
        'MarkerSize',10 + (3-1)*3);
    type1 = X(:,lbl== lbl1);
    type2 = X(:,lbl==lbl2);
    
    scatter(type1(1,:),type1(2,:),'r');
    scatter(type2(1,:),type2(2,:),'b');
    %plot(X(:,lbl==lbl2)','b*');
    %draw boundaries
    [xgrid,ygrid] = meshgrid(linspace(-4,8,100),linspace(-3,7,100));
    [~,scores] = predict(Mdl.BinaryLearners{j},[xgrid(:),ygrid(:)]);
    
    contour(xgrid,ygrid,reshape(scores(:,2),size(xgrid)),[0,0],'g');
    xlabel('strain')
    ylabel('stress')
    title("SVM for "+classes(lbl1)+" and "+classes(lbl2))
    legend(['support vectors',classes(lbl1),classes(lbl2)],'Location','northwest')
    filename = [num2str(j) '.jpg'];
    saveas(j,filename)
    hold off
end
