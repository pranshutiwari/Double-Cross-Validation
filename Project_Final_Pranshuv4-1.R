## step 1 loading libraries
## Step  2 taking the data 
## Step 3 Categorizing/treating the data correctly as factor or continous and imputing values
# Step 4 Single fold CV -Kappa
# Step 5 Double CV Kappa based for Model assessment only but no prediction as I dont have probability to select
# Step 6 Since data set is imbalanced use highly senstive objective function to find optimum probability on best double cv model
# Step 7 run the best model on single CV on optimum probability 
# Step 8 Now lets creating clinical data set based on 3 parameters and run step 5,6.7 again to see that at clinical level can algorithm work
# Step 9 charts
## Step 1


library(dplyr)
library(tree)
library(glmnet)
library(gbm)
library(caret)
library(VIM)
library(MASS)
library(ggformula)
library(dplyr)
library(ggformula)
library(pROC)
library(nnet)
library(NeuralNetTools)
library(neuralnet)
library(corrplot)
library(yardstick)

# Step 2  & Step 3 
df11=read.csv('healthcare-dataset-stroke-data (1).csv')
df11$gender=as.factor(df11$gender)
df11$heart_disease=as.factor(df11$heart_disease)
df11$ever_married=as.factor(df11$ever_married)
df11$work_type=as.factor(df11$work_type)
df11$Residence_type=as.factor(df11$Residence_type)
df11$stroke=as.factor(df11$stroke)
df11$hypertension=as.factor(df11$hypertension)
df11$smoking_status=as.factor(df11$smoking_status)
summary(df11)
head(df11)
na1=which((df11$bmi=='N/A') & (df11$heart_disease==1))
class(df11$bmi[2])
na2=which((df11$bmi=='N/A') & (df11$heart_disease==0))
na1;na2
length(na1)
length(na2)
df11$bmi=as.numeric(as.character(df11$bmi))
summary(df11)
df11[sapply(df11, is.character)] <- lapply(df11[sapply(df11, is.character)], 
                                       as.factor)
summary(df11)
#remove id
df11=df11[,-1]#remove id
summary(df11)
index1=which(df11$heart_disease==1)
index2=which(df11$heart_disease==0)
mean1=mean(df11$bmi[index1],na.rm=TRUE)
mean2=mean(df11$bmi[index2],na.rm=TRUE)
mean2;mean1
summary(df11)
df11$bmi[na1]=mean1
df11$bmi[na2]=mean2
summary(df11)
index_other=which(df11$gender=='Other')
df11=df11[-index_other,]
df11$gender = factor(df11$gender, levels = c("Female", "Male"))
summary(df11)
###################
# Step 4 
# now lets proceed with single cross validation 
lambdalist = exp((-5000:-2000)/100)
plot(lambdalist)
n.lambda=length(lambdalist)
alphalist = c(.9,1); n.alpha = length(alphalist)
plot(alphalist)
training = trainControl(method = "cv", number = 5)
#convering data to matrix as some packages use matrix
xy.out=df11
penalized_model= train(stroke~ . ,
                    data = xy.out,
                    method = "glmnet",
                    trControl = training,
                    tuneGrid = expand.grid(alpha=alphalist,lambda=lambdalist),family='binomial',preProc=c('scale','center'))

head(predict(penalized_model,newdata = xy.out,type='prob'))

max(penalized_model$results$Kappa)
plot(penalized_model$results$Kappa,ylim=c(2.5,3.5))
penalized_model$bestTune
library(dplyr)
library(tidyr)
penalized_model$results
enet_mat <- penalized_model$results %>%
  dplyr::select(alpha,lambda,Kappa) %>%
  pivot_wider(names_from = alpha, values_from = Kappa) 
enet_mat
enet_mat = as.matrix(enet_mat)
enet_mat
row.names(enet_mat) = round(enet_mat[ ,1],3)
enet_mat
enet_mat = enet_mat[ , -1]
enet_mat
library(gplots)
#heatmap.2(enet_mat, Rowv = FALSE, Colv = FALSE, 
          #dendrogram = "none", trace = "none", 
          #xlab = "alpha", ylab = "lambda")
#finallambda = fit_caret_penalized$bestTune$lambda
#finalalpha = fit_caret_penalized$bestTune$alpha
ann_model=train(stroke ~ . ,
                   data = xy.out,
                   method = "nnet",
                   trControl = training,metric='Kappa',linout=FALSE,tunegrid=expand.grid(size=c(1,2,3),decay=c(0.0001,0.001,.001,0.01,0.1,0.2)),preProc=c('scale','center'))
ann_model$bestTune

all_best_Types = c("Penalized_Logistic","ANN")
all_best_Pars = list(penalized_model$bestTune,ann_model$bestTune)
all_best_Pars
all_best_Models = list(penalized_model$finalModel,
                       ann_model$finalModel)
all_best_Models
all_best_Kappa = c(max(penalized_model$results$Kappa),max(ann_model$results$Kappa))
all_best_Kappa
one_best_Type = all_best_Types[which.max(all_best_Kappa)]
one_best_Pars = all_best_Pars[which.max(all_best_Kappa)]
one_best_Pars
one_best_Model = all_best_Models[[which.max(all_best_Kappa)]]
one_best_Model
class(one_best_Pars)
############# one best model (LASSO) fit to full data #############
ee_net_lambda = one_best_Pars[[1]]$lambda
ee_net_alpha=one_best_Pars[[1]]$alpha
ee_net_alpha
coef <- coef(penalized_model$finalModel,s=ee_net_lambda)
coef

allLasso_log=length(penalized_model$results$Kappa)
allLasso_Log
allANN=length(ann_model$results$Kappa);
############# compare all models - visual understanding #############
# model counts and types
mann = allANN; meenet =allLasso_log
meenet
mmodels = mann+meenet;mmodels
modelMethod = c(rep("ANN",mann),rep("Penalized_Logistic",meenet))
all_caret_Kappa = c(ann_model$results$Kappa,
                   penalized_model$results$Kappa)
coloptions = rainbow(4)
colused = coloptions[as.numeric(factor(modelMethod))+1]
charused = 5*(as.numeric(factor(modelMethod)))
charused
all_caret_Kappa
max(all_caret_Kappa)
max(all_caret_Kappa)
plot(1:mmodels,all_caret_Kappa,col=colused,pch=charused,
     xlab = "Model label",ylab = "Kappa")
order.max = c(which.max(penalized_model$results$Kappa),
              which.max(ann_model$results$Kappa))
abline(v=order.max,lwd=2)
print(order.max)
pro=predict(ann_model,newdata=df11,type='prob')
###
pro
##sample
p=table(df11$stroke,(pro[,2]>0.1))
###lets create a cost function 
#cost of not identifying a stroke person but having stroke is 10 times higher than identifying  a healthy patient witk stroke for prevention
c1=100
c2=10
r=p[2,2]/(p[2,2]+p[2,1])
r
s=p[1,1]/(p[1,1]+p[1,2])
ct=c(c1,c2)
v=c(r,s)
v
Cost=t(ct)%*% v
Cost
## sample ends - now use sample to create function
## ########cost function  for steop 6 
cost_function<-function(p){
  c1=100# Cost of Person with Stroke not diagnosed as stroke
  c2=10# cost of Healthy person misdiagnosed as stroke
  if ((dim(p)[2])>1){
   r=(p[2,2])/(p[2,2]+p[2,1])
   s=p[1,1]/(p[1,1]+p[1,2])
   ct=c(c1,c2)
   v=c(r,s)
   Cost=t(ct)%*% v# Transpose Ct*v
   } else {
     r=0
     s=1
     ct=c(c1,c2)
     v=c(r,s)
     Cost=t(ct)%*% v
   }
  return (Cost)
}
h=cost_function(p)
h[1,1]
dim(p)
# Now lets do this over validation run:
ncv=5
n=dim(df11)[1]
best_alpha=1
best_lambda=0.006
groups=rep(1:ncv,length=n)
size_valid=one_best_Pars[1]
decay_valid=one_best_Pars[2]
p_range=c(0.1,0.15,0.2,0.25,0.3)
## STEP 5
## Lets use double cross validation where we change the training and validation set to see how good these models perform consistently
## PLEASE NOTE THAT SINCE DATA IS NOT BALANCED AND I NEED TO FIND PROBABILITY LEVEL OF RIGHT THRESHOLD
## THIS DOUBLE CROSS VALIDATION IS ONLY FOR CONSISTENT MODEL ASSESSMENT AND NOT HONEST PREDICTIONS

ncv=10
set.seed(10)
n=dim(df11)[1]
#lets use the entire data and divide it into training set and validation set
groups=rep(1:ncv,length=n)
#we have created different folds and now sample it
cvgroups=sample(groups,n)#sampling it
# creating storage facility to store data for each fold results
allpredictedCV = rep(NA,n)#storing prediction value
# set up storage to see what models are "best" on the inner loops
allbestTypes = rep(NA,ncv)#each fold will have best value 
allbestPars = vector("list",ncv)# each best model will have a list of parameters 
all_best_Kappa_cont=rep(0,ncv)
for (fold in 1:ncv){
  #lets create a matrix as glmnet function works as matrix format
  #in case we use matrix in glmnet function
  cond1=(cvgroups==fold)#LIST OF TRUE AND FALSE 
  valid_index=which(cond1==TRUE)
  train_index=which(cond1==FALSE)
  validdata = df11[valid_index,]
  traindata=df11[train_index,]
  #For each fold of train_data #lets tune the parameter and then choose the best parameter for the fold
  dataused=traindata# for ease of understanding 
  training=trainControl(number = 5,method="cv")# setting up cv for caret for inner cross validation 
  ann_model=train(stroke~.,data =  dataused,trControl=training,tuneGrid=expand.grid(size=c(1,2,3),decay=c(0.0001,0.001,0.002,0.01,0.1)),method='nnet',linout=FALSE,metric='Kappa',preProc=c('center','scale'))
  penalized_model=train(stroke~.,data = dataused,trControl=training,metric='Kappa',method='glmnet',tuneGrid=expand.grid(alpha=c(1),lambda=c(0.0001,0.001,0.01,0.1,0.2,0.3,0.9)))
  all_best_Types = c("ANN","Penalized_Logistic")
  all_best_Pars = list(ann_model$bestTune,penalized_model$bestTune)
  all_best_Model = list(ann_model$finalModel,penalized_model$finalModel)
  all_best_Kappa = c(max(ann_model$results$Kappa),max(penalized_model$results$Kappa))
  one_best_Type = all_best_Types[which.max(all_best_Kappa)]# Higher value of Kappa
  one_best_Pars = all_best_Pars[which.max(all_best_Kappa)]
  allbestTypes[fold] = one_best_Type
  allbestPars[[fold]] = one_best_Pars
  if (one_best_Type == "ANN") {  # then best is one of linear models
    ann_size = one_best_Pars[[1]]$size
    ann_decay=one_best_Pars[[1]]$decay
    print(one_best_Type)
    #allpredictedCV[valid_index] = predict(one_best_Model,newdata = validdata)
  } else if (one_best_Type == "Penalized_Logistic") {   # then best is one of LASSO models
    penalized_lambda=one_best_Pars[[1]]$lambda
    #allpredictedCV[valid_index]  = predict(one_best_Model,newdata = validdata)
    print(one_best_Type)
  }
}
#I have hashed prediction here since the by default probability level is 5 and we need to find the best probability level
# this right probability level is found in next step using single CV by creating objective function
validdata
allbestTypes
#make this example reproducible
vote_ann=0
vote_pen_log=0
for (fold in 1:ncv) {
  writemodel = paste("The best model at loop", fold, 
                     "is of type", allbestTypes[fold],
                     "with parameter(s)",allbestPars[fold])
  print(writemodel, quote = FALSE)
}
# The total number of votes are more for ANN. Within ANN maximum votes went to size-3, decay =0.1 followed by decay 0.01,size=3
#Penalized Logistic Regression is the second best model and alpha =1,Lambda=0.0001 /0.001 are best model
# Although not the best model it will be good to show the ROC curve forp >0.1

#STEP 5B VOTING#######
vote_ann=0
vote_pen_logistic=0
for (j in 1:length(allbestTypes)){
  if (allbestTypes[j]=='ANN'){
    vote_ann=vote_ann+1
  }else{
    vote_pen_logistic=vote_pen_logistic+1
  }
}
vote_pen_logistic
vote_ann
vote_ann
# STEP 6 USING OBJECTIVE FUNCTION CREATED
#Since ANN IS THE BEST MODEL  LETS LOOK AT COST MATRIX TO FIND THE OPTIMUM PROBABILITY 
# FINDING OPTIMUM PROBABILITY
ncv=10
best_size=3# Since there are 2 items 
best_decay=c(0.01,0.1)#finetune further
groups=rep(1:ncv,length=n)
cvgroups=sample(groups,n)
allpredictedCV=rep(0,n)
cost_mat=matrix(data=NA,nrow=ncv,ncol = length(p_range))
for (k in 1:length(p_range)){
 for (fold in (1:ncv)){
    condition=(cvgroups==fold)#list of TRUE and FALSE
    train_1=df11[condition==FALSE,] #TRAIN
    valid=df11[condition==TRUE,]#VALID
    dataused=train_1#TRAIN
    training2=trainControl(number=5,method = 'cv') #inner for decay variation
    model_ann_voted=train(stroke~.,data =  dataused,trControl=training2,tuneGrid=expand.grid(size=c(3),decay=best_decay),method='nnet',linout=FALSE,metric='Kappa',preProc=c('center','scale'))
    probability=predict(model_ann_voted,newdata = valid ,type='prob')
    yhat=ifelse(probability[,2]>p_range[k],1,0)
    p=table(valid$stroke,yhat)
    h=cost_function(p)
    cost_mat[fold,k]=h[1,1]
 } 
}
cost_mat
cost_mat
senstivity_p=colMeans(cost_mat)
senstivity_p
r=which.max(senstivity_p)
pmax=p_range[r]
pmax

#####WE HAVE NOW FOUND CONSISTENT BEST MODEL FROM DOUBLE CROSS VALIDAITON AND FOUND OPTIMUM PROBABILITY
## NOW LETS MAKE HONEST PREDICTION
# Hence Probability >0.1 will be identified as stroke:
#Now Running Single Fold Cross Validation on best model size=1,decay for full predictor data 
ncv=10
best_decay=0.1
best_size=3
allpredictedCV=rep(0,dim(df11)[1])
predictor_prob3=rep(0,dim(df11)[1])
for (fold in (1:ncv)){
  condition=(cvgroups==fold)#list of TRUE and FALSE
  train_1=df11[condition==FALSE,] #TRAIN
  valid=df11[condition==TRUE,]#VALID
  dataused=train_1#TRAIN
  model_ann_voted=train(stroke~.,data =  dataused,tuneGrid=expand.grid(size=best_size,decay=best_decay),method='nnet',linout=FALSE,metric='Kappa',preProc=c('center','scale'))
  probability=predict(model_ann_voted,newdata = valid ,type='prob')
  predictor_prob3=probability[,2]
  yhat=ifelse(probability[,2]>0.1,1,0)
  allpredictedCV[condition==TRUE]=yhat
}
summary(model_ann_voted)
plotnet(model_ann_voted$finalModel)
predictor_prob3
Final_Confusion_matrix=table(df11$stroke,allpredictedCV)
#Final_Confusion_matrix=data.frame(prediction=allpredictedCV,truth=df11$stroke)
#Lets create a dataframe of different metrics
Accuracy=sum(diag(Final_Confusion_matrix))/sum(Final_Confusion_matrix)
Senstivity=Final_Confusion_matrix[2,2]/(Final_Confusion_matrix[2,2]+Final_Confusion_matrix[2,1])
Specificity=Final_Confusion_matrix[1,1]/(Final_Confusion_matrix[1,1]+Final_Confusion_matrix[1,2])
Final_Confusion_matrix

cm <- yardstick::conf_mat(Final_Confusion_matrix, truth, prediction)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")

# Please note this is not THE best model but plotting to show how good logistic regression was to compare ROC curve only
# #Penalized Logistic Regression is the second best model and alpha =1,Lambda=0.0001 /0.001 are best model
# Although not the best model it will be good to show the ROC curve forp >0.1
allpredictedCV_log=rep(0,dim(df11)[1])
predictor_prob=rep(0,dim(df11)[1])
for (fold in (1:ncv)){
  condition=(cvgroups==fold)#list of TRUE and FALSE
  train_1=df11[condition==FALSE,] #TRAIN
  valid=df11[condition==TRUE,]#VALID
  dataused=train_1#TRAIN
  model_log_2nd_best=train(stroke~.,data =  dataused,tuneGrid=expand.grid(alpha=c(1),lambda=c(0.002)),method='glmnet',metric='Kappa',preProc=c('center','scale'),family='binomial')
  probability=predict(model_log_2nd_best,newdata = valid ,type='prob')
  predictor_prob[condition==TRUE]=probability[,2]
  yhat=ifelse(probability[,2]>0.1,1,0)
  allpredictedCV_log[condition==TRUE]=yhat
}
r=table(df11$stroke,allpredictedCV_log)
roc_rf_prob = roc(response = df11$stroke, predictor = predictor_prob)
plot.roc(roc_rf_prob)
Accuracy_log=sum(diag(r))/(sum(r))
Senstivity_Log=(r[2,2])/(r[2,1]+r[2,2])
Specificity_Log=(r[1,1])/(r[1,1]+r[1,2])
## Summary Sheet###
df_measures=data.frame("Type"=c("ANN-All","Penalized Logistic_All"),"Accuracy"=c(Accuracy,Accuracy_log),"Senstivity"=c(Senstivity,Senstivity_Log),"Specificity"=c(Specificity,Specificity_Log))
knitr::kable(head(df_measures[, 1:4]), "pipe")



###
table2=coef(model_log_2nd_best$finalModel, model_log_2nd_best$bestTune$lambda)
knitr::kable(round(table2[,],2), "simple")

pdf("plot_2v2.pdf", width = 11, height = 8.5)
par(mar = c(.1, .1, .1, .1))
library(pROC)
myroc = roc(response=df11$stroke, predictor=predictor_prob)
myroc2 = roc(response=df11$stroke, predictor=predictor_prob3)
plot.roc(myroc)
plot.roc(myroc2, add=T, col="red", lty=2)
legend("bottomright", legend=c("Pena_Log:Lamb=0.002","NNET_Best"),lty=c(1,2), col=c("black","red"),cex = 1.0)
#
dev.off()
auc(myroc)# AUC value of Penalized logistic:ALL variables
auc(myroc2)#AUC value of Neural Network based models:All variables

pdf("plot_nnet5v3.pdf", width = 22, height = 12)
par(mar = c(.1, .1, .1, .1))
plotnet(model_ann_voted$finalModel)
dev.off()

#VARIMP PLOT
#Final Summary of The best Model of NNET:
summary(model_ann_voted)
varImp(model_ann_voted)

q=ggplot(data=GA,aes(x=reorder(Var_Name,Overall),y=Overall))+geom_bar(stat='identity')
q+theme(axis.text.x = element_text(face = "bold", angle = 90))

par(mar = c(.1, .1, .1, .1))
pdf("plot_VAR_barv3.pdf", width = 18, height = 9)
par(mar = c(.1, .1, .1, .1))
GA=varImp(model_ann_voted$finalModel)
GA$Var_Name=rownames(GA)
q=ggplot(data=GA,aes(x=reorder(Var_Name,Overall),y=Overall))+geom_bar(stat='identity')
q+theme(axis.text.x = element_text(face = "bold", angle = 90))+coord_flip()
dev.off()
GA

GA=varImp(model_ann_voted$finalModel)
GA$Var_Name=rownames(GA)
q=ggplot(data=GA,aes(x=reorder(Var_Name,Overall),y=Overall))+geom_bar(stat='identity')
q+theme(axis.text.x = element_text(face = "bold", angle = 90))+coord_flip()


#Since Lek Profile uses how response varies I am considering 3 top variables: Age, BMI, Smoking Status and Hypertension as key varuables
#sample Lek profile

####################################################################################
###############EXTRA ANALYSIS FOR COMPARING WITH CLINICAL DATA WITH ONLY 3 VARIABLES######
##This is just extra analysis because in certain places of world all data is not available but age,BMI AND BLood gluscose is easily availalbe
colnames(df11)
df_top=df11[,c(2,8,9,11)]#clinical conditions
set.seed(100)
summary(df_top)
ctrl = trainControl(method = "cv", number = 5)
best_fit_ann = train(stroke ~ .,
                  data = df_top,
                  method = "nnet",
                  tuneGrid = expand.grid(size = 3, 
                                         decay = c(0.001,0.01)),
                  preProc = c("center", "scale"),
                  linout = FALSE,
                  maxit = 100,
                  trace = FALSE,
                  trControl = ctrl)

lekprofile(best_fit_ann)
#Comparison of Clincial conditions vs Non Clinical Conditions
#Clinical conditions are the one where we have numerical values like BMI, Age & Glucose
# This suggest that size=1 and decaly=0.1 is the best model in these clinical conditions
# Lets compare the model performance of clinical conditions Vs all conditions 
#Computing Probability >0.1 
ncv=10
set.seed(10)
n=dim(df_top)[1]
#lets use the entire data and divide it into training set and validation set
groups=rep(1:ncv,length=n)
#we have created different folds and now sample it
cvgroups=sample(groups,n)#sampling it
# creating storage facility to store data for each fold results
allpredictedCV_d = rep(NA,n)#storing prediction value
# set up storage to see what models are "best" on the inner loops
allbestTypes = rep(NA,ncv)#each fold will have best value 
allbestPars = vector("list",ncv)# each best model will have a list of parameters 
all_best_Kappa_cont=rep(0,ncv)
for (fold in 1:ncv){
  #lets create a matrix as glmnet function works as matrix format
  #in case we use matrix in glmnet function
  cond1=(cvgroups==fold)#LIST OF TRUE AND FALSE 
  valid_index=which(cond1==TRUE)
  train_index=which(cond1==FALSE)
  validdata = df_top[valid_index,]
  traindata=df_top[train_index,]
  #For each fold of train_data #lets tune the parameter and then choose the best parameter for the fold
  dataused=traindata# for ease of understanding 
  training=trainControl(number = 5,method="cv")# setting up cv for caret for inner cross validation 
  ann_model=train(stroke~.,data =  dataused,trControl=training,tuneGrid=expand.grid(size=c(1,2,3),decay=c(0.0001,0.001,0.002,0.01,0.1)),method='nnet',linout=FALSE,metric='Kappa',preProc=c('center','scale'))
  penalized_model=train(stroke~.,data = dataused,trControl=training,metric='Kappa',method='glmnet',tuneGrid=expand.grid(alpha=c(1),lambda=c(0.0001,0.001,0.01,0.1,0.2,0.3,0.9)))
  all_best_Types = c("ANN","Penalized_Logistic")
  all_best_Pars = list(ann_model$bestTune,penalized_model$bestTune)
  all_best_Model = list(ann_model$finalModel,penalized_model$finalModel)
  all_best_Kappa = c(max(ann_model$results$Kappa),max(penalized_model$results$Kappa))
  one_best_Type = all_best_Types[which.max(all_best_Kappa)]
  one_best_Pars = all_best_Pars[which.max(all_best_Kappa)]
  allbestTypes[fold] = one_best_Type
  allbestPars[[fold]] = one_best_Pars
  #all_best_RMSE_cont[fold]=all_best_RMSE
  if (one_best_Type == "ANN") {  # then best is one of linear models
    ann_size = one_best_Pars[[1]]$size
    ann_decay=one_best_Pars[[1]]$decay
    print(one_best_Type)
    #allpredictedCV[valid_index] = predict(one_best_Model,newdata = validdata)
  } else if (one_best_Type == "Penalized_Logistic") {   # then best is one of LASSO models
    penalized_lambda=one_best_Pars[[1]]$lambda
    #allpredictedCV[valid_index]  = predict(one_best_Model,newdata = validdata,s=penalized_lambda)
    print(one_best_Type)
  }
}
validdata
#make this example reproducible
vote_ann=0
vote_pen_log=0
for (fold in 1:ncv) {
  writemodel = paste("The best model at loop", fold, 
                     "is of type", allbestTypes[fold],
                     "with parameter(s)",allbestPars[fold])
  print(writemodel, quote = FALSE)
}
###
ncv=10
best_decay=0.1
best_size=1
allpredictedCV_top=rep(0,dim(df_top)[1])
predictor_prob4=rep(0,dim(df_top)[1])
for (fold in (1:ncv)){
  condition=(cvgroups==fold)#list of TRUE and FALSE
  train_1=df_top[condition==FALSE,] #TRAIN
  valid=df_top[condition==TRUE,]#VALID
  dataused=train_1#TRAIN
  model_ann_voted_top=train(stroke~.,data =  dataused,tuneGrid=expand.grid(size=best_size,decay=best_decay),method='nnet',linout=FALSE,metric='Kappa',preProc=c('center','scale'))
  probability=predict(model_ann_voted_top,newdata = valid ,type='prob')
  predictor_prob4[condition==TRUE]=probability[,2]
  yhat=ifelse(probability[,2]>0.1,1,0)
  allpredictedCV_top[condition==TRUE]=yhat
}
summary(model_ann_voted_top)
plotnet(model_ann_voted_top$finalModel)
predictor_prob4
clinical=table(df_top$stroke,allpredictedCV_top)
clinical
accuracy_clinical=sum(diag(clinical))/sum(clinical)
senstivity_clinical=clinical[2,2]/(clinical[2,2]+clinical[2,1])# TP/TP+FN
specificity_clinical=clinical[1,1]/(clinical[1,1]+clinical[1,2])# True Negative/(TN+FP)
cost_function(clinical)
y_hat_top=ifelse(predictor_prob4>0.3,1,0)

pdf("plot_ROC_Differnet_Models3v4.pdf", width = 11, height = 8.5)
par(mar = c(.1, .1, .1, .1))
library(pROC)
myroc = roc(response=df11$stroke, predictor=predictor_prob)
myroc2 = roc(response=df11$stroke, predictor=predictor_prob3)
myroc3=roc(response=df_top$stroke,predictor=predictor_prob4)
plot.roc(myroc)
plot.roc(myroc2, add=T, col="red", lty=2)
plot.roc(myroc3,add=T,col="green",lty=3)
legend("bottomright", legend=c("Pena_Log-All:Lamb=0.002","NNET_Best:ALL","ANN_Clinical_Only: Age+BMI+Glucose"),lty=c(1,2), col=c("black","red","green"),cex = 1.0)
#
dev.off()
#summary sheet for All variables +3 clinical variables##
## Summary Sheet including additional 3 clinical variaBLES##
df_measures_final=data.frame("Model Type "=c("ANN-All","Penalized Logistic_All","3-Var:ANN "),"Accuracy"=c(Accuracy,Accuracy_log,accuracy_clinical),"Senstivity"=c(Senstivity,Senstivity_Log,senstivity_clinical),"Specificity"=c(Specificity,Specificity_Log,specificity_clinical))
knitr::kable(head(df_measures_final[, 1:4]), "pipe")
df_measures_final2 <- head(df_measures_final)
knitr::kable(df_measures_final2, col.names = gsub("[.]", " ", names(df_measures_final)))
# Response variable behaviour for full data set vs parameters

df11 <- df11 %>%
  mutate(example_preds = allpredictedCV)#prediction of ANN based model on All predictor dataset
allpredictedCV
df11$hypertension=as.factor(df11$hypertension)
df11$gender=as.factor(df11$gender)
df11$example_preds=as.factor(df11$example_preds)
df11%>%
  gf_point(example_preds ~ age, col =~ hypertension)+ylab("Predicted Stroke Response")
