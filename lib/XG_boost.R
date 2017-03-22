#########################################
###########    XG boost    ##############
#########################################

# Load library
library(xgboost)


# Load sift feature and label
sift <- read.csv('/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/sift_features.csv')
sift <- t(sift)
label <- read.csv('/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/labels.csv')


# Create training and testing data
index <- sample(2000,1600)
sift_train <- sift[index,]
label_train <- label[index,]
label_train <- lapply(t(label_train), as.numeric)
dtrain <-xgb.DMatrix(as.matrix(sift_train),label = label_train)

sift_test <- sift[-index,]
label_test <- label[-index,]
label_test <- lapply(t(label_test), as.numeric)
dtest <- xgb.DMatrix(as.matrix(sift_test),label = label_test)


# CV function to fit parameters
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:30) {
  param <- list(objective = "binary:logistic",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2) 
  )
  cv.nround = 50
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=8, maximize=FALSE)
  
  min_logloss = min(mdcv[, test.error.mean])
  min_logloss_index = which.min(mdcv[ , test.error.mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}


# fit XG boost model based on training data
nround = best_logloss_index
set.seed(best_seednumber)
xg.fit <- xgboost(data=dtrain, params=best_param, nrounds=nround, nthread=6)


# predict using testing data
xg.pred <- predict(xg.fit, dtest)
xg.pred <- as.numeric(xg.pred > mean(xg.pred))

error <- (sum(data.frame(label_test)!= xg.pred)) /ncol(data.frame(label_test)) # 0.42
