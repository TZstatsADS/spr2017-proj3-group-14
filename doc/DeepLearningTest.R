# Import data
source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage")

# Pre-process of images

library("EBImage")
img_dir <- "../data/raw_images/"
source("../lib/annex.R")
n <- 2000
sizeImg <- 50
n_files <- length(list.files(img_dir))
rand_sample <- sample(1:n_files, n)
imgs <- matrixfy_imgs(img_dir, rand_sample, n, sizeImg, sizeImg)
labels <- read.csv("../data/labels.csv", header = T)[rand_sample,1]

data <- test_train_sep(imgs, labels, 0.33333) 
train.x <- data[[1]]
train.y <- data[[2]]
test.x <- data[[3]]
test.y <- data[[4]]

# Deep Learning phase
require(mxnet)

mx.set.seed(2)
model <- mx.mlp(train.x, train.y, hidden_node=c(100,50,10), out_node=2, activation = "sigmoid", out_activation="softmax",
                num.round=500, array.batch.size=50, learning.rate=0.07,
                eval.metric=mx.metric.accuracy, dropout = 0.4, momentum = 0.7, initializer=mx.init.normal(1.5))
graph.viz(model$symbol)

accuracy_model(model, test.x,test.y)
#########################
# Test with LeNet model #
#########################

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=10)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="sigmoid")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(3,3), num_filter=20)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="sigmoid")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=200)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=50)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

train.array <- train.x
dim(train.array) <- c(sizeImg, sizeImg, 1, nrow(train.x))
test.array <- test.x
dim(test.array) <- c(sizeImg, sizeImg, 1, nrow(test.x))

device.cpu <- mx.cpu()

mx.set.seed(2)
tic <- proc.time()
modelLeNet <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=15, array.batch.size=10,
                                     learning.rate=0.05, optimizer="sgd",
                                     eval.metric=mx.metric.accuracy, initializer=mx.init.normal(0.2),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
                                     # wd=0.00001
graph.viz(modelLeNet$symbol)
accuracy_model(modelLeNet, test.array,test.y)