crop_image_ratio11 <- function(img){
  w <- dim(img)[1]
  h <- dim(img)[2]
  m <- min(w,h)
  if (m==h) {
    medm <- floor(w/2-h/2)
    medp <- floor(w/2+h/2)
    return (img[(medm:medp),])}
  else {
    medm <- floor(h/2-w/2)
    medp <- floor(h/2+w/2)
    return (img[,(medm:medp)])
  }
}


matrixfy_imgs <- function(img_dir, rand_sample, n=NA, width=100 ,height=100){
  
  ### Convert images to matrices for training/testing images

  ### Input: a directory that contains images ready for processing
  ### Output: a matrix where each row i represent the flatten image i reshaped to (width,height)
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(img_dir))
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n, width*height)
  cpt<-1
  for(i in rand_sample){
    # name of image 1: image_0001
    nb <- paste(c(rep("0",3-floor(log(i,10))),i),collapse="")
    img <- readImage(paste0(img_dir,  "image", "_", nb,".jpg"))
    dat[cpt,] <- as.vector(resize(crop_image_ratio11(img), width, height))
    cpt<-cpt+1
  }
  return(dat)
}

getImage <- function(imgs, index, width){
  img <- imgs[index,]
  dim(img) <- c(width,width)
  return(img)
}

test_train_sep <- function(data, labels, test_perc){
  # return a list with train,test sets
  n <- nrow(data)
  n_th <- floor(n*test_perc)
  rand_sample <- sample(1:n,n_th)
  return(list(imgs[-rand_sample,], labels[-rand_sample],imgs[rand_sample,], labels[rand_sample]))
}

accuracy_model <- function(model, test.x, test.y){
  preds = predict(model, test.x)
  pred.label = max.col(t(preds))-1
  print(mean(pred.label))
  print(table(pred.label, test.y))
  sprintf("Accuracy of model: %s%%",round(mean(pred.label==test.y),3)) 
}
