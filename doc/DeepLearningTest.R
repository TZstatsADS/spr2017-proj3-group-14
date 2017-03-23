# Import data
source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage")

# Pre-process of images

library("EBImage")
img_dir <- "../data/raw_images/"
source("../lib/annex.R")

set.seed(1)
# display(getImage(imgs,10,40))  # to show example image from the matrix imgs

### Load images or re-compile them ###

## load data to imgs
load("../data/data1rotAugmentation.RData")

## only if recompile the data from raw images
#n <- 2000
#sizeImg <- 50
#rot <- 1
#n_files <- length(list.files(img_dir))
#rand_sample <- sample(1:n_files, n)
#imgs <- matrixfy_imgs(img_dir, rand_sample, n, sizeImg, sizeImg, rot=rot)
#labels <- read.csv("../data/labels.csv", header = T)[rand_sample,1]
#labels <- rep(labels, each=(1*rot+2))
#save(imgs, labels, sizeImg, rot, n, file="../data/data1rotAugmentation.RData")

data <- test_train_sep_dataaug(imgs, labels, 0.2, rot)
train.x <- data[[1]]
train.y <- data[[2]]
test.x <- data[[3]]
test.y <- data[[4]]
remove(data) # save memory since useless

# Deep Learning phase
require(mxnet)

# The only model that gives really good result.
# Training accuracy of 84.56% and test accuracy 77.42%.
# Model is saved in DNNbest.RData and will be load as "model" if compile line 52 directly.

# To retrain the model:
mx.set.seed(2)
model <- mx.mlp(train.x, train.y, hidden_node=c(500,250,100,10), out_node=2, activation = "sigmoid", out_activation="softmax",
                num.round=750, array.batch.size=150, learning.rate=0.3,
                eval.metric=mx.metric.accuracy, dropout = 0.70, momentum = 0.7, initializer=mx.init.normal(1.5))

accuracy_model(model, test.x,test.y)
save(model, file="../data/DNNbest.RData")

# To predict new images:
load("../data/DNNbest.RData")
preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
print(table(pred.label, test.y))

#################################################################################################################
# 77.41% model saved (but overfit to 99% in training accuracy) -> might try to improve it but delete it after   #
#################################################################################################################
#model <- mx.mlp(train.x, train.y, hidden_node=c(300,200,50), out_node=2, activation = "sigmoid", out_activation="softmax",
#                num.round=700, array.batch.size=150, learning.rate=0.15,
#                eval.metric=mx.metric.accuracy, dropout = 0.7, momentum = 0.7, initializer=mx.init.normal(1.5))


########################################
# Test with LeNet model (Advanced CNN) # -> not concluant, lack of data or computing power to train longer.
########################################

# parameter dropout (proba of drop)
pdrop <- 0.4

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=4)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(2,2), num_filter=8)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
drop1 <- mx.symbol.Dropout(flatten,p=pdrop)
fc1 <- mx.symbol.FullyConnected(data=drop1, num_hidden=250)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="relu")

# second fullc
drop2 <- mx.symbol.Dropout(tanh3, p=pdrop)
fc2 <- mx.symbol.FullyConnected(data=drop2, num_hidden=100)

# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

train.array <- (train.x-apply(train.x,2,mean))/apply(train.x,2,sd)
dim(train.array) <- c(sizeImg, sizeImg, 1, nrow(train.x))
test.array <- (test.x-apply(train.x,2,mean))/apply(train.x,2,sd)
dim(test.array) <- c(sizeImg, sizeImg, 1, nrow(test.x))

device.cpu <- mx.cpu()

mx.set.seed(2)
tic <- proc.time()
modelLeNet <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                          ctx=device.cpu, num.round=3, array.batch.size=200,
                                          learning.rate=0.1, optimizer="adagrad",
                                          eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(1/sizeImg),
                                          epoch.end.callback=mx.callback.log.train.metric(100))# wd=0.00001)


accuracy_model(modelLeNet, test.array,test.y)
graph.viz(modelLeNet$symbol)

##################
##  Simple CNN  ## -> not concluant either
##################

pdrop <- 0.7

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=10)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(2,2), num_filter=15)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
drop1 <- mx.symbol.Dropout(flatten,p=pdrop)
fc1 <- mx.symbol.FullyConnected(data=drop1, num_hidden=200)
relu1 <- mx.symbol.Activation(data=fc1, act_type="relu")

# loss
output <- mx.symbol.SoftmaxOutput(data=relu1)

train.array <- (train.x-apply(train.x,2,mean))/apply(train.x,2,sd)
dim(train.array) <- c(sizeImg, sizeImg, 1, nrow(train.x))
test.array <- (test.x-apply(train.x,2,mean))/apply(train.x,2,sd)
dim(test.array) <- c(sizeImg, sizeImg, 1, nrow(test.x))

device.cpu <- mx.cpu()

mx.set.seed(0)
tic <- proc.time()
modelCNN <- mx.model.FeedForward.create(output, X=train.array, y=train.y,
                                        ctx=device.cpu, num.round=50, array.batch.size=250,
                                        learning.rate=0.5, optimizer="adagrad",
                                        eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(1/sizeImg),
                                        epoch.end.callback=mx.callback.log.train.metric(150))# wd=0.00001)
accuracy_model(modelCNN, test.array,test.y)


# good links to understand CNN:
# CNN: http://web.engr.illinois.edu/~slazebni/spring14/lec24_cnn.pdf
# optimizer in NN: http://sebastianruder.com/optimizing-gradient-descent/
# set up CNN : http://cs231n.github.io/neural-networks-2/#init
# initialization weights CNN: https://github.com/NVIDIA/DIGITS/blob/master/examples/weight-init/README.md
# on LeNet: http://deeplearning.net/tutorial/deeplearning.pdf

############## -> applied directly on image matrices
##  t-SNE  ### -> separation of classes does not occur. The decorrelation does not work.
############## -> maybe because of similarities in the images and other element not relevant (background)

library("Rtsne")

colors = c("red","blue")
names(colors) = unique(train.y)

## Executing the algorithm on curated data for 2D
tsne <- Rtsne(train.x, dims = 2, perplexity=20, initial_dims=30, check_duplicates=F, theta=0.3, verbose=TRUE, max_iter = 500)

## Plotting 2D
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train.y, col=colors[train.y+1])

## Executing the algorithm on curated data for 3D
tsne <- Rtsne(train.x, dims = 3, perplexity=20, initial_dims=30, check_duplicates=F, theta=0.3, verbose=TRUE, max_iter = 500)

## Plotting 3D
library("scatterplot3d")
scatterplot3d(tsne$Y[,1],tsne$Y[,2],tsne$Y[,3],color=colors[train.y+1])

###################
## Sift features ## -> DNN does not work, diverges
###################

siftF <- t(apply(read.csv("../data/sift_features.csv", header=F),2,as.numeric))
#siftF <- t(read.csv("../data/sift_features.csv", header = T))
labels <- read.csv("../data/labels.csv", header = T)[,1]

dataSift <- test_train_sep(siftF, labels, 0.33333) 
trainS.x <- dataSift[[1]]
trainS.y <- dataSift[[2]]
testS.x <- dataSift[[3]]
testS.y <- dataSift[[4]]

mx.set.seed(2)
modelS <- mx.mlp(trainS.x, trainS.y, hidden_node=c(300,500,200,100), out_node=2, activation = "sigmoid", out_activation="softmax",
                 num.round=50, array.batch.size=50, learning.rate=0.1,
                 eval.metric=mx.metric.accuracy, dropout = 0.6, momentum = 0.7, initializer=mx.init.normal(1.5))

accuracy_model(modelS, testS.x,testS.y)
graph.viz(modelS$symbol)

############## -> applied directly on sift features
##  t-SNE  ### -> separation of classes does not occur. The decorrelation does not work.
############## 

colors = c("red","blue")
names(colors) = unique(train.y)

## Executing the algorithm on curated data
tsne <- Rtsne(trainS.x, dims = 2, perplexity=20, initial_dims=50, check_duplicates=F, theta=0.1, verbose=TRUE, max_iter = 700)
#exeTimeTsne<- system.time(Rtsne(train.x, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500))

## Plotting 2D
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=trainS.y, col=colors[trainS.y+1])

#####################
## Lasso features  ## -> DNN diverges to 50-50% classification
#####################

lasso <- apply(read.csv("../data/fea_lasso.csv", header=F),2,as.numeric)

dataLasso <- test_train_sep(lasso, labels, 0.33333) 
trainL.x <- dataLasso[[1]]
trainL.y <- dataLasso[[2]]
testL.x <- dataLasso[[3]]
testL.y <- dataLasso[[4]]

mx.set.seed(2)
modelL <- mx.mlp(trainL.x, trainL.y, hidden_node=c(1000,200,200,50), out_node=2, activation = "relu", out_activation="softmax",
                 num.round=100, array.batch.size=100, learning.rate=0.05,
                 eval.metric=mx.metric.accuracy, dropout = 0.4, momentum=0.7, initializer=mx.init.normal(0.05))
accuracy_model(modelL, testL.x,testL.y)

############## -> applied directly on lasso features
##  t-SNE  ### -> separation of classes does not occur. The decorrelation does not work.
############## 

colors = c("red","blue")
names(colors) = unique(train.y)

## Executing the algorithm on curated data
tsne <- Rtsne(trainL.x, dims = 2, perplexity=20, initial_dims=50, check_duplicates=F, theta=0.1, verbose=TRUE, max_iter = 700)
#exeTimeTsne<- system.time(Rtsne(train.x, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500))

## Plotting 2D
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=trainL.y, col=colors[trainL.y+1])