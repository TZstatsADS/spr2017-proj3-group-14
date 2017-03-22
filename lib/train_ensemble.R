# the goal of this r file is to use bagging and random forest / logistic regression / SVM
# the reason to choose: 
#   1.SVM would be Big-O cost(n^2) very inefficient in large scale
#   2.logistic regression perform worse than RM on the sample file
#      (to quick the process, choose sample, only linear model is not enough)
#   3.logistic regression have strong corelation with each other for each weak learner
#   4.Random forest is okay here, theoratical okay & error okay
# ensamble learning to get the good result based on traditional mahine learning
# Traditional machine learning - I mean not deep learning KE HAN
#library
library(caret)
library(sgd)
#load data
features=read.csv(file.choose())
labels=read.csv(file.choose())
features=t(features)

#before we ran into logistic part, 
#we choose features(otherwise, no converge in data)
#for image data, it is not a good idea to use this
# feature selection
# 1.cut off low variance feature
# 2.PCA, using dimension deduction method (tried, but the result is not valid)
# 3.Random forest to choose (implictly choose)

# cut off low variance variables
variance=apply(features,2,var)
summary(variance)
select=index[variance<1.980e-07]
all_data<-data.frame(labels,features)
all_data<-all_data[select,]
dim(all_data)

#training & testing
set.seed(10)
positions <- sample(nrow(all_data),size=floor((nrow(all_data)/4)*3))
training<- all_data[positions,]
testing<- all_data[-positions,]

#logistic regression
#not converge, cannot be used here
lm_fit<-glm(training$V1~.,family=binomial(link='logit'),data=training)
predictions<-ifelse(predict(lm_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)

#random forest
#error=0.21 
library(randomForest)
rf_fit<-randomForest(training$V1 ~.,data=training,ntree=500)
predictions<-ifelse(predict(rf_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)
error

# svm
library(e1071)
svm_fit<-svm(training$V1~.,data=training)
svm_predictions<-ifelse(predict(svm_fit,newdata=testing)>0,1,0)
error<-((sum((testing$V1!=svm_predictions)))/nrow(testing))
error



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

########all into funtion style##########

#############method1: based on variance####

variance_cut_off <- function(data, cutoff){
  # cutoff: the variance cutoff value
  
  variance <-  apply(data, 2, var)
  min(variance)
  max(variance)
  
  # count the number of variables left
  # if variance > 0.5e-6, we keep the feature, then there are 1968(n) features remaining
  n=0
  for(i in 1:5000){
    if(variance[i] >= cutoff){n=n+1}
    else{n=n}
  }
  
  # remove the features with small variance
  cut <- rep(cutoff,5000)
  getcol <-  variance - cut >= 0
  return(data[,getcol])
}

###########method2: random forest###
# since the permutation variable importance is affected by collinearity
# it's necessary to handle collinearity prior to running RF

random_forest <- function(data, label, n){
  # n: number of features to keep
  
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  library(caret)
  install.packages("corrplot")
  library(corrplot)
  library(plyr)
  
  # Give each feature a "name" and Calculate correlation matrix
  feature1 <- data
  colnames(feature1)[1:5000] <- as.character(seq(1,5000,by=1))
  descrCor <- cor(feature1)
  
  # Print correlation matrix and look at max correlation
  summary(descrCor[upper.tri(descrCor)])
  
  # Find attributes that are highly corrected
  highlyCorrelated <- findCorrelation(descrCor, cutoff=0.6)
  highlyCorCol <- colnames(feature1)[highlyCorrelated]
  # Remove highly correlated variables and create a new dataset
  features1 <- feature1[, -which(colnames(feature1) %in% highlyCorCol)]

  ########### 2.Use random forest
  # ensure the results are repeatable
  install.packages("randomForest")
  library(randomForest)
  
  df <- as.data.frame(cbind(features1,label))
  allX <- paste("X",1:ncol(features1),sep="")
  names(df) <- c(allX,"label")
  
  time <- system.time(rf <- randomForest(as.factor(label)~.,data = df, importance = TRUE,ntree = 500))
  imp <- importance(rf, type=1)
  imp <- data.frame(predictors=rownames(imp),imp)
  
  imp.sort <- arrange(imp,desc(MeanDecreaseAccuracy))
  imp.sort$predictors <- factor(imp.sort$predictors,levels=imp.sort$predictors)
  
  # Select the top n predictors
  imp.100=imp.sort[1:n,]
  print(imp.100)
  
  varImpPlot(rf, type=1)
  
  return(df[,c(imp.100$predictors)])
}

########## into funtion SVM ############
trainSVM <- function(dat_train, label_train, par = NULL){
  
  ### load libraries
  library(e1071)
  
  if(is.null(par)){
    ### default parameter values
    gamma <- 0.1
    cost <- 1
    kernel <- 'radial'
  } else {
    gamma <- par$gamma
    cost <- par$cost
    kernel <- par$kernel
  }
  
  fit_svm <- svm(dat_train, label_train, kernel = kernel,gamma = gamma, cost = cost, cross = 10)
  
  return(fit = fit_svm)
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
# choose the best for further analysis, ensemble+

#average for logistic regression
library(foreach)
length_divisor<-6
iterations<-5000
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
  train_pos<-1:nrow(training) %in% training_positions
  lm_fit<-glm(training$V1 ~.,family=binomial(link='logit'),training[train_pos,])
  predict(lm_fit,newdata=testing)
}
predictions<-ifelse(rowMeans(predictions)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)
error


#first ensembel
#bagging into random forest
length_divisor<-6
iterations<-5000
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
  train_pos<-1:nrow(training) %in% training_positions
  lm_fit<-glm(training$V1 ~.,family=binomial(link='logit'),training[train_pos,])
  predict(lm_fit,newdata=testing)
}
predictions<-ifelse(rowMeans(predictions)>0.5,1,0)

library(randomForest)
rf_fit<-randomForest(training$V1 ~.,data=training,ntree=500)
rf_predictions<-predict(rf_fit,newdata=testing)
lm_predictions<-predict(lm_fit,newdata=testing)
predictions<-ifelse((lm_predictions+rf_predictions)/2>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)
error

##########use cross validation #####

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
