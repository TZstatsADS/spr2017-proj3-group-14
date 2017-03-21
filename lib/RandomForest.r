
# the focuse is on RM, since it alrealdy show good result#
##########use cv for RM#####

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
    
    fit <- train(train.data, train.label)
    pred <- test(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}


train<- function(dat_train, label_train){
  library(randomForest)
  df <- as.data.frame(cbind(dat_train, label_train))
  allX <- paste("X",1:ncol(dat_train),sep="")
  names(df) <- c(allX,"label")
  fit <- randomForest(as.factor(label)~.,data = df, importance = TRUE,ntree = 500)
  return(fit)
}

test<- function(fit, dat_test){
  pred <-ifelse(predict(fit,newdata=dat_test)>0.5,1,0)
  #return(as.numeric(pred> 0.5))
  return (pred)
}
