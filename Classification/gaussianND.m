function pdf = gaussianND(X,mu,sigma)

n = size(X,2);
meandiff = bsxfun(@minus,X,mu);
pdf = 1/sqrt((2*pi)^n*det(sigma))*exp((-1/2) * sum((meandiff*inv(sigma).*meandiff),2));

end