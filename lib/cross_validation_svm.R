###2017spring_group14
##################Cross Validation for Model Selection##############################
####################################################################################

# Cross validation function for either the GBM or the advanced model

# INPUT: 
#     X = features of input images, matrix in images*features format
#     y = class labels for training images, 0 represent chicken and 1 dogs
#     K = number of folds used in cv
#     paras = list of parameter values, passed on directly to training functions


# OUTPUT: mean test error over all folds

cross_validation_svm = function(x=sift_features, y=label_train, paras=NULL, K=5, model='SVM'){
  
  
  source("../lib/train_SVM.R")
  source("../lib/test.R")
  
  n = as.numeric(length(y))
  n.fold = floor(n/K)
  s = sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error = rep(NA, K)
  
  for (i in 1:K){
    train.data = x[s != i,]
    train.label = y[s != i]
    test.data = x[s == i,]
    test.label = y[s == i]
    
    fit=train_SVM(train.data, train.label, paras)
    
    pred = test(fit, test.data)  
    
    cv.error[i] = mean(pred != test.label) 
    
  }			
  
  return(mean(cv.error))
  
}
