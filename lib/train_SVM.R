train_SVM = function(x, y, paras=NULL){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     X =  matrix in images*features format
  #     y = class labels for training images
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  
  library('e1071')
  
  if(is.null(par)){
    degree=3
    tolerance=0.001
    nu=0.5
    cost=1
    coef0=0
  } 
  else {
    eval(parse(text = paste(names(paras), paras, sep='=', collapse = ';')))
  }
  
  svm = svm(x=x, y=y,
                    kernel="radial",
                    degree=degree,
                    coef0=coef0, 
                    tolerance=tolerance,
                    nu=nu,
                    cost=cost)
  
  return(svm)
}
