###2017spring_group14
##################The Base Model: Gradient Boosting Machine##############################
#########################################################################################

train_GBM = function(x, y, paras=NULL){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     X =  matrix in images*features format
  #     y = class labels for training images
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  
  library('gbm')

  if(is.null(par)){
    depth = 1
    shrinkage = 0.1
    n.trees = 100
  } 
  else {
    eval(parse(text = paste(names(paras), paras, sep='=', collapse = ';')))
  }
  
  gbm_fit = gbm.fit(x=x, y=y,
                    distribution = "bernoulli",
                    n.trees = n.trees,
                    interaction.depth = depth, 
                    shrinkage = shrinkage,
                    bag.fraction = 0.5,
                    verbose=FALSE)
  
  return(gbm_fit)
}
