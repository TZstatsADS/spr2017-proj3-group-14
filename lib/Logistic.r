
cv.function <- function(data, label, K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  
  set.seed(0)
  library(caret)
  fold <- createFolds(1:dim(data)[1], K, list=T, returnTrain=F)
  fold <- as.data.frame(fold)
  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    test.data <- data[fold[,i],]
    train.data <- data[-fold[,i],]
    test.label <- label[fold[,i],]
    train.label <- label[-fold[,i],]
    
    #par <- list(depth = d, Ntrees = n, Shrinkage = r)
    fit <- train.logit (train.data, train.label)
    pred <- test.logistic(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}



####### train.logit() ############

train.logit <- function(dat_train, label_train){
  library("sgd")
  fit <- sgd(dat_train, label_train, model='glm', model.control=binomial(link="logit"))
  return(fit)
}

test.logistic <- function(fit, dat_test){
  pred <- predict(fit, dat_test,type = 'response')  
  
  return(as.numeric(pred> 0.5))
}
