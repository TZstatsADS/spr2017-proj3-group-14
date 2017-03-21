###################################################################################
#                      Main execution script for experiments                      #
###################################################################################

### Project 3
### Group 14
### ADS Spring 2017


############################ Step 0: Set directotires #############################

experiment_dir <- "/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/"
# Please change to your own directories
img_train_dir <- paste(experiment_dir, "train/", sep="")
img_test_dir <- paste(experiment_dir, "test/", sep="")

#### Load library and source function
library("gbm")
library("data.table")
source('/Users/limengchen/Desktop/spr2017-proj3-group-14-master/lib/cross_validation.R')
source('/Users/limengchen/Desktop/spr2017-proj3-group-14-master/lib/train_GBM.R')

#### Set up controls for evaluation experiments
run.cv=TRUE # run cross-validation on the training set
K <- 5  # number of CV folds
run.test=TRUE # run evaluation on an independent test set


############ Step 1: import training images and class labels ########################

### read labels
mylabels<-read.table(paste(experiment_dir, "labels.csv", sep=""),header=T)
set.seed(2)
### randomly select 1500 images as training data set
### remaining 500 images as testing data set
label_train_num<-sample(1:2000,1500,replace=FALSE) 
label_test_num<- -label_train_num
label_train<-mylabels[label_train_num,]
label_train<-as.matrix(label_train)
label_test<-mylabels[label_test_num,]
#need to be changed based on professor's data
label_test<-as.matrix(label_test)


### read sift features
sift_features_all<-read.csv("/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/sift_features.csv")
sift_features_all<-t(sift_features_all)
sift_features_all<-as.matrix(sift_features_all)
sift_features_train<-sift_features_all[label_train_num,]
sift_features_test<-sift_features_all[label_test_num,]


################## Step 2: Baseline Model: GBM #####################################

### Run Cross Validation on different shrinkage parameters
# we take parameters from 0.1 to 5 step 0.1
shrinks<-seq(0.1, 0.5, 0.1)
# record the test error for GBM with diffrent shrinkage parameters
test_err_GBM = numeric(length(shrinks))
for(i in 1:length(shrinks)){
  cat("GBM model Cross Validation No.", i, "model out of",length(shrinks), "models", "\n")
  paras = list(depth=1,
               shrinkage=shrinks[i],
               n.trees=100)
  test_err_GBM[i] = cross_validation(sift_features_train, label_train,
                                     paras=paras, K=5, model='GBM')
}


### Plot the test error
plot(y=test_err_GBM,x=shrinks,xlab="shrinkages",ylab="CV Test Error",main="GBM CV Test Error",type="n")
points(y=test_err_GBM,x=shrinks, col="blue", pch=16)
lines(y=test_err_GBM,x=shrinks, col="blue")

### Fit best shrinkage parameters into GBM Model
best_shrink = shrinks[which.min(test_err_GBM)]
paste("The best shrinkage parameter is",best_shrink) # The best shrinkage parameter is 0.1
best_paras = list(depth=1, shrinkage=best_shrink, n.trees=100)
best_paras # depth=1, shrinkage=0.1, n.trees=100

### running time
tm_GBM<-NA
tm_GBM <- system.time(GBM_fit<-train_GBM(x=sift_features_train, y=label_train, paras=best_paras))
tm_GBM # 15.159s

### Fit test sift features into trained GBM model and predict the results
source('/Users/limengchen/Desktop/spr2017-proj3-group-14-master/lib/test.R')
pred_GBM<-test(GBM_fit,sift_features_test)

# save prediction in output folder, both in csv and Rdata
write.csv(pred_GBM,file="../output/GBM_prediction.csv")
save(pred_GBM,file="../output/GBM_prediction.Rdata")
pred_accuracy<-mean(pred_GBM==label_test)
pred_accuracy # pred_accuracy=0.73


################## Step 3 (1): Advances Model: SVM #####################################

set.seed(2)
# combine labels with features
all_data<-data.frame(mylabels,sift_features_all)
positions <- sample(nrow(all_data),size=floor((nrow(all_data)/4)*3))
training<- all_data[positions,] # 1500 training data set
testing<- all_data[-positions,] # 500 testing data set

### run cross validation on different cost parameters
???


### Fit SVM model
svm_fit<-svm(training$V1~.,data=training)
svm_predictions<-ifelse(predict(svm_fit,newdata=testing)>0,1,0)
error_svm<-((sum((testing$V1!=svm_predictions)))/nrow(testing))

pred_accuracy_svm=mean(svm_predictions==testing$V1)
pred_accuracy_svm # 0.51， too low!!!

write.csv(svm_predictions,file="../output/svm_predictions.csv")
save(pred_GBM,file="../output/svm_predictions.Rdata")
pred_accuracy<-mean(svm_predictions==label_test)

### running time
??? # seems quick

################## Step 3 (2): Advances Model: Random Forest #####################################

### run cross validation on parameters
???

### Fit Random Forest model
library(randomForest)
rf_fit<-randomForest(training$V1 ~.,data=training,ntree=500)
predictions<-ifelse(predict(rf_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)

pre_accuracy_rf=mean(svm_predictions==testing$V1)
pre_accuracy_rf # 0.726

### running time
???  # seems slow