########################################################
####                 Texture feature               #####
########################################################
Get.Main<-function(Img,NP = 32){
  library(radiomics)
  library(EBImage)
  #---------Get Main Object Areas------#
  Test = Img@.Data
  nrow = dim(Test)[1]
  ncol = dim(Test)[2]
  H = hist(Test)
  index = which(H$density>1)
  if(length(index)==1){
    Temp1 = as.vector(Test)
    Layer = which(Temp1>=H$breaks[index] & Temp1<=H$breaks[index+1])
    Temp2 = Temp1[-Layer]
    H = hist(Temp2)
    index = which(H$density>1)
  }
  MainColors = H$breaks[c(min(index),max(index))]
  Temp = as.vector(Test)
  Temp[Temp<MainColors[1] | Temp>MainColors[2]] = 0
  Temp[Temp!=0] = 1
  #------Select a submatrix for analyze--#
  Test1 = matrix(Temp,nrow,ncol)
  col.index = which.max(apply(Test1,2,sum)) 
  row.index = which.max(apply(Test1,1,sum)) 
  
  GetSubMatrix<-function(col.index,row.index,ncol,nrow,NP){
    col.min = max(1,(col.index-NP)) 
    col.max = min(ncol,(col.index+NP-1)) 
    row.min = max(1,(row.index-NP)) 
    row.max = min(nrow,(row.index+NP-1)) 
    col = col.min:col.max
    row = row.min:row.max
    if(col.min == 1){ col = 1:(NP*2)}
    if(col.max == ncol){ col = (ncol-2*NP+1):ncol}
    if(row.min == 1){ row = 1:(NP*2)}
    if(row.max == nrow){ row = (nrow-2*NP+1):nrow}
    return(list(row=row,col=col))
  }
  Sub  = GetSubMatrix(col.index,row.index,ncol,nrow,NP)
  if(max(dim(Test)<64)){ Main = Test}else{
    Main = Test[Sub$row,Sub$col]
  }
  while(length(unique(as.numeric(Main)))<=32){
    NP = NP+10
    Sub = GetSubMatrix(col.index,row.index,ncol,nrow,NP)
    Main = Test[Sub$row,Sub$col]
  }
  
  #---Try another part of Main---# #The alternative part works! This is weird!
  #Main = Test[1:10,2:15]
  #----Get Texture Features ----#
  Texture = tryCatch(glcm(Main),error=function(e){return(glcm(Test))}) #Can't solve the error part 
  TF = calc_features(Texture)
  Texture.Features = as.vector(TF)
  return(Texture.Features)
}




Get.Texture.Features<-function(num,Name){
  library(EBImage)
  Features = NULL
  for(i in num){
    print(i)
    if(i %in% 1:9) file.name = paste0(Name,"_000",i,".jpg")
    if(i %in% 10:99) file.name = paste0(Name,"_00",i,".jpg")
    if(i %in% 100:999) file.name = paste0(Name,"_0",i,".jpg")
    if(i %in% 1000:9999) file.name = paste0(Name,"_",i,".jpg")
    Img = channel(readImage(file.name),"gray")
    #---Analyze Texture Features---#
    Temp = Get.Main(Img,NP=32)
    Features = cbind(Features,t(Temp))
  }
  colnames(Features) = c(rep(Name,length(num)))
  return(Features)
}

#---Extract Features---#

Chicken.TF = Get.Texture.Features(1:1000,'/Users/limengchen/Desktop/spr2017-proj3-group-14-master 111/data/training_data/raw_images/image')
Dog.TF = Get.Texture.Features(1001:2000,"/Users/limengchen/Desktop/spr2017-proj3-group-14-master 111/data/training_data/raw_images/image")
save(Dog.TF,file="/Users/limengchen/Desktop/Dog.TF.RData")
save(Chicken.TF,file = "/Users/limengchen/Desktop/Chicken.TF.RData")

Chicken.TF <- t(Chicken.TF)
Dog.TF <- t(Dog.TF)
feature_Tex <- rbind(Chicken.TF, Dog.TF)
rownames(feature_Tex) <- paste('image',1:2000)
save(feature_Tex,file="/Users/limengchen/Desktop/feature_Tex.RData")




}



