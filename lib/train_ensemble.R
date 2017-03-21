# the goal of this r file is to use bagging and random forest / logistic regression / SVM
# the reason to choose: 
#   1.SVM would be Big-O cost(n^2) very inefficient in large scale
#   2.logistic regression perform worse than RM on the sample file(to quick the process, choose sample)
#   3.logistic regression have strong corelation with each other for each weak learner
# ensamble learning to get the good result based on traditional mahine learning
# Traditional machine learning - I mean not deep learning KE HAN

#load data
features=read.csv(file.choose())
labels=read.csv(file.choose())
features=t(features)

#before we ran into logistic part, 
#we choose features(otherwise, no converge in data)
#for image data, it is not a good idea to use this
# terrible result please do not use it!
# logistic part
set.seed(10)
all_data<-data.frame(labels,features)
positions <- sample(nrow(all_data),size=floor((nrow(all_data)/4)*3))
training<- all_data[positions,]
testing<- all_data[-positions,]

lm_fit<-glm(training$V1~.,family=binomial(link='logit'),data=training)
predictions<-ifelse(predict(lm_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)
error

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
library(randomForest)
rf_fit<-randomForest(training$V1 ~.,data=training,ntree=500)
predictions<-ifelse(predict(rf_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)
error


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




# svm
library(e1071)
svm_fit<-svm(training$V1~.,data=training)
svm_predictions<-ifelse(predict(svm_fit,newdata=testing)>0,1,0)
error<-((sum((testing$V1!=svm_predictions)))/nrow(testing))
error
