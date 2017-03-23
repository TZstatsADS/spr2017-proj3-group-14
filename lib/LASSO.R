
install.packages("glmnet")
install.packages("readr")
library(glmnet)
library(readr)

data = read_csv("/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/sift_features.csv")
data = t(data)

data2 = read.csv("/Users/limengchen/Desktop/spr2017-proj3-group-14-master/data/labels.csv")

x = as.matrix(data)

x.sd = apply(x, 2, function(x) (x - mean(x))/sd(x))




y = data2[,1]

nfolds = 10

cv.fit  = cv.glmnet(x.sd , factor(y) ,nfolds = nfolds, family = "binomial")

plot(cv.fit)


#use predictors selected by optimal lambda
#optimal lambda

cv.fit$lambda.min


coef(cv.fit, lambda = "lambda.min")


predictors = which(coef(cv.fit, lambda = "lambda.min")[,1] < 0.0001)

predictors
dim(data[,predictors])
fea_lasso <- data[,predictors]

write.csv(fea_lasso, file='/Users/limengchen/Desktop/fea_lasso.csv')


#ridge
cv.fit  = cv.glmnet(x , factor(y) ,nfolds = nfolds, family = "binomial", alpha = 0)

plot(cv.fit)


#use predictors selected by optimal lambda
#optimal lambda

cv.fit$lambda.min


coef(cv.fit, lambda = "lambda.min")



predictors = which(coef(cv.fit, lambda = "lambda.min")[,1] != 0)

predictors










