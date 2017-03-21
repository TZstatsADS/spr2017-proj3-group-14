###2017spring_group14
######################################################
### Fit the classification model with testing data ###
######################################################


test_svm = function(fit_train, dat_test){
  # Fit the classfication model with testing data
  # INPUT: 
  #     fit_train = trained model object, either gbm or xgb.Booster
  #     dat_test = processed features from testing images 
  #
  # OUTPUT: training model specification
  
  library('e1071')
  
  pred=predict(fit_train,
               newdata = dat_test,
               tolerance = fit_train$tolerance)
  
  
  return(as.numeric(pred> 0.5))
}