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


matrixfy_imgs <- function(img_dir, rand_sample, n=NA, width=100 ,height=100, rot=0){
  
  ### Convert images to matrices for training/testing images

  ### Input: a directory that contains images ready for processing
  ### Output: a matrix where each row i represent the flatten image i reshaped to (width,height)
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(img_dir))
  if (rot==0) {
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
  }
  else {
    dat <- matrix(NA, (1*rot+2)*n, width*height)
    cpt<-1
    for(i in rand_sample){
      # name of image 1: image_0001
      nb <- paste(c(rep("0",3-floor(log(i,10))),i),collapse="")
      img <- readImage(paste0(img_dir,  "image", "_", nb,".jpg"))
      img_aug <- data_aug.rotation(img, rot)
      for (j in 1:(1*rot+2)) {
        dat[cpt,] <- as.vector(resize(crop_image_ratio11(img_aug[[j]]), width, height))
        cpt<-cpt+1
      }
    }
  }
  return(dat)
}

data_aug.rotation <- function(img, times) { # implement also symetry
  angles <- seq(0,5,length.out = times+1)[1:times+1]
  res <- vector("list", times+2)
  res[[1]] <- img
  res[[2]] <- flop(img)
  counter <- 3
  for (angle in angles) {
    res[[counter]] <- rotate(img, angle)
    counter <- counter+1
  }
  return(res)
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
  return(list(data[-rand_sample,], labels[-rand_sample],data[rand_sample,], labels[rand_sample]))
}

test_train_sep_dataaug <- function(data, labels, test_perc, rot){
  # return a list with train,test sets
  n <- nrow(data)/(1*rot+2)
  n_th <- floor(n*test_perc)
  rand_sample <- sample(1:n,n_th)
  s <- rep((rand_sample-1)*(1*rot+2),each=1*rot+2)+(1:(1*rot+2))
  return(list(data[-s,], labels[-s],data[s,], labels[s]))
}

accuracy_model <- function(model, test.x, test.y){
  preds = predict(model, test.x)
  pred.label = max.col(t(preds))-1
  print(table(pred.label, test.y))
  sprintf("Accuracy of model: %s%%",round(mean(pred.label==test.y)*100,3)) 
}
