function G = testSVM(Mdl,X)
[label,NegLoss,score] = predict(Mdl,X);
%fprintf("Label predicted: %s\n",label);
celldisp(label);
sma_prob = exp(NegLoss(1))/sum(exp(NegLoss));
pzt_prob = exp(NegLoss(2))/sum(exp(NegLoss));
dea_prob = exp(NegLoss(3))/sum(exp(NegLoss));
eap_prob = exp(NegLoss(4))/sum(exp(NegLoss));
fprintf("Probability of SMA: %f\n",sma_prob);
fprintf("Probability of PZT: %f\n",pzt_prob);
fprintf("Probability of DEA: %f\n",dea_prob);
fprintf("Probability of EAP: %f\n",eap_prob);



G = label;



end