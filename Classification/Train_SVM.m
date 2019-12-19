function Mdl = Train_SVM(CompletedMat,Y)


%%%%%multivariate SVM
t = templateSVM('Standardize',true,'SaveSupportVectors',true,'KernelFunction','linearkernel');
predictorNames = {'stress','strain','efficiency'};
responseName = 'ActuatorType';
classNames = {'SMA','PZT','DEA','EAP'};
Mdl = fitcecoc(CompletedMat',Y,'Learners',t,'ResponseName',responseName,...
    'PredictorNames',predictorNames,'ClassNames',classNames);
L = size(Mdl.CodingMatrix,2);
sv = cell(L,1);
for j = 1:L
    SVM = Mdl.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end

complete = CompletedMat';
figure
gscatter(complete(:,1),complete(:,2),Y);
hold on
markers = {'ko','ro','bo','go','co','mo'};
for j = 1:L
    svs = sv{j};
    plot(svs(:,1),svs(:,2),markers{j},...
        'MarkerSize',10 + (j-1)*3);
end
title('SMA,PZT,DEA,EAP')
xlabel('strain')
ylabel('stress')
legend([{'SMA','PZT','DEA','EAP'},{'Support vectors - SVM 1',...
    'Support vectors - SVM 2','Support vectors - SVM 3','Support vectors - SVM 4'...
    'Support vectors - SVM 5','Support vectors - SVM 6'}],...
    'Location','Best')
hold off

end

