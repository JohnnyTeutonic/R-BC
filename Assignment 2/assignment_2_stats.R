# Required packages:

needed_packages<-c("glmnet", "rmarkdown", "FNN", "klaR", "MASS", "readxl", "plot3D",
                   "fpc", "factoextra", "rgl", "Rtsne", "caret")
lapply(needed_packages, install.packages, character.only = TRUE)
lapply(needed_packages, library, character.only = TRUE)

#n.b. need r version 3.4 on linux to use 'ggbiplot' package:
#install.packages("ggbiplot")

# For reproducibility purposes.
set.seed(1)

## TASK 1

data<-read.csv(file = 'assignment_2_report_files/train.csv')
test_final<-read.csv(file='assignment_2_report_files/test.csv')

# use a holdout strategy to make predictions from the trainin data

holdout<-function(splitsize, dataset){
  firstcut<-floor(splitsize*nrow(dataset))
  secondcut<-ceiling((1-splitsize)*nrow(dataset))
  train_ind <- sample(seq_len(nrow(dataset)), size = firstcut)
  train<-dataset[train_ind, ]
  test<-dataset[-train_ind, ]
  list<-list(train, test)
  return(list)
}


choose_k<-function(tr, ts, Y_tr, Y_ts, reg=TRUE){
  "Function to determine the optimal k and minimum loss for either knn regression or knn classification.
  'k' is calculated from a minimum choice of '2' to a maximum consisting of the square root of the number of samples.
  'tr' is the training data. 
  'ts' is the testing data. 
  'Y_tr' is the true labels of the training data.
  'Y_ts' is the true labels of the testing data -> used to calculate the error.
  'reg' is a boolean that sets whether to perform 'knn-regression' (set to TRUE by default)
  or 'knn classification.'
  
  returns: list containing four objec4ts->
  list[[1]]->minimum_loss correspondonding to lowest loss (mse for regression, 0-to-1 for classification) 
  list[[2]]->choice of best k, 
  list[[3]->the best model,
  list[[4]->nested list containing the losses for each k.
  list[[5]]->nested list containing the actual number of each k used in the loop.
  "
  
  best_k = 0
  best_mod = 0
  min_loss = 1000
  if ((ceiling(sqrt(nrow(ts))) %% 2)==0){
    max_k <-ceiling(sqrt(nrow(ts)))+1
  } else{
    max_k <-ceiling(sqrt(nrow(ts)))
  }
  k.choice<-seq(from=3, to=max_k, by=2)
  k_list <- rep(1, length(k.choice-1))
  knn_list<-rep(1, length(k.choice-1))
  var_list <- list(
    var1 = k.choice,
    var2 = k_list 
  )
  # only choose odd-valued 'k' to prevent ties:
  for (i in 1:length(var_list$var2))
  {
    if (reg){
      knn_preds<-knn.reg(tr,  ts, Y_tr, var_list$var1[i])
      knn_loss<-mean((Y_ts - knn_preds$pred) ^ 2)
      k_list[i] = knn_loss
      knn_list[i] = var_list$var1[i]
    }
    else {
      # return majority vote class as 'prob'.
      knn_preds<-knn(tr,  ts, factor(Y_tr), var_list$var1[i], prob=TRUE)
      knn_loss<-(1/length(Y_ts))*length(which(Y_ts!=knn_preds))
      k_list[i] = knn_loss
      knn_list[i] = var_list$var1[i]
    }
    
    if (knn_loss<min_loss){
      min_loss=knn_loss
      best_k=var_list$var1[i]
      best_mod=knn_preds
    }
  }
  return(list(min_loss, best_k, best_mod, k_list, knn_list))
}

a<-holdout(0.7, data)
train<-data.frame(a[1])
dev<-data.frame(a[2])
Y1_tr<-train[, 1]
Y2_tr<-train[, 2]
Y3_tr<-train[, 3]
Y1_ts<-dev[, 1]
Y2_ts<-dev[, 2]
Y3_ts<-dev[, 3]
Xtrain<-train[,-(1:3)]
Xtest<-dev[, -(1:3)]


## Y1
model1<-glm(Y1_tr~., data=Xtrain)

mse_tr_y1 = mean(model1$residuals^2)

mse_ts_y1<-mean((Y1_ts - predict.lm(model1, Xtest)) ^ 2)

lambdaCV1 <- cv.glmnet(as.matrix(Xtrain),Y1_tr)$lambda.min
model_cv1 <- glmnet(as.matrix(Xtrain), Y1_tr, lambda = lambdaCV1)
sum(abs(model_cv1$beta>0))
#[1] 17
# 17 predictors significant after lasso regularisation for Y1.

cv_probs_y1<-predict(model_cv1, newx = as.matrix(Xtest), type='response')
mse_cv_y1<-mean((Y1_ts - (cv_probs_y1 ^ 2)))


#knn_params_1<-knn.cv(Xtrain, k=5, cl = Y1_tr)
opt_k_1<-choose_k(Xtrain, Xtest, Y1_tr, Y1_ts)

opt_k_1[[2]]
# [1] 7
# best k is '7' for Y1 on dev data
knn_preds_y1<-knn.reg(Xtrain, Xtest, y = Y1_tr, k = opt_k_1[[2]])
mse_knn_y1<-mean((Y1_ts - knn_preds_y1$pred) ^ 2)

y1_potential<-c(mse_ts_y1, mse_knn_y1, mse_cv_y1)

which.min(y1_potential)
#[1] 1
# we stick with linear model for y1 as it has the lowest MSE on the dev data out of all three models.

## Y2
mse_ts_y2<-mean((Y2_ts - predict.lm(model1, Xtest)) ^ 2)

opt_k_2<-choose_k(Xtrain, Xtest, Y2_tr, Y2_ts)

opt_k_2[[2]]
# [1] 9
# choice of k equal to '9' optimal for Y2 knn-regression.

knn_preds_y2<-knn.reg(Xtrain, Xtest, y = Y2_tr, k=opt_k_2[[2]])
mse_knn_y2<-mean((Y2_ts - knn_preds_y2$pred) ^ 2)
y2_potential<-c(mse_ts_y2, mse_knn_y2)

which.min(y2_potential)
# [1] 2
# knn model for 'y2' has the smallest MSE on the dev data, so this will be our final model 
# along with the optimal k chosen (opt_k_2[[2]])


## Y3

# do KNN classification on Y3, find the best k:
opt_k_3<-choose_k(Xtrain, Xtest, factor(Y3_tr), Y3_ts, reg=FALSE)

opt_k_3[[2]]
#[2] 39
# from above, best choice of k is 39 on dev data

# minimal loss on dev data for knn-classification:
knn_loss_3<-opt_k_3[[1]]
# [1] 0.176


# lasso regression to find significant X predictors:
lambdaCV3 <- cv.glmnet(as.matrix(Xtrain), factor(Y3_tr), family = "binomial", type.measure = 'auc')$lambda.min
model_cv3 <- glmnet(as.matrix(Xtrain), factor(Y3_tr), family = "binomial", lambda = lambdaCV3)
sum(abs(model_cv3$beta>0))
# [1] 26
# 26 out of 50 variables are non-zero
cv_probs_y3<-predict(model_cv3, newx = as.matrix(Xtest), type='response')
cv_preds_y3 <- ifelse(cv_probs_y3 > 0.5, 1, 0)

# Use the 0-1 loss function: 
log_loss_3<-(1/length(Y3_ts))*length(which(Y3_ts!=cv_preds_y3))
# [1] 0.008666667


# use an LDA model on all the predictors:
model_lda<-lda(Y3_tr~., data=Xtrain)
YPredLDA <- predict(model_lda, Xtest)$class
log_loss_lda<-(1/length(Y3_ts))*length(which(YPredLDA!=Y3_ts))

# use an QDA model on all the predictors:

model_qda<-qda(Y3_tr~., data=Xtrain)
YPredQDA <- predict(model_qda, Xtest)$class
log_loss_qda<-(1/length(Y3_ts))*length(which(YPredQDA!=Y3_ts))

# create a model using only the significant lasso regression predictors
mod_3_selected<-paste("X", which(model_cv3$beta>0), sep = '', collapse = '+')
mod3_formula<-as.formula(paste('Y3_tr ~', mod_3_selected))

model_lda_selected_3<-lda(mod3_formula, data=Xtrain)
YPredLDA_selected_3 <- predict(model_lda_selected_3, Xtest)$class
log_loss_lda_selected_3<-(1/length(Y3_ts))*length(which(YPredLDA_selected_3!=Y3_ts))


# create a vector of all the log-losses for each model:
y3_potential<-c(log_loss_3, knn_loss_3, log_loss_lda, log_loss_lda_selected_3, log_loss_qda)

which.min(y3_potential)
# [1] 1
# logistic_cv model has much smaller MSE than other models on dev data, so we use this model for y3 test predictions


## final predictions for each Y response variable on the test data:

preds1_final<-predict(model1, test_final, type='response')
preds2_final<-knn.reg(Xtrain, test=test_final, y=Y2_tr, k=opt_k_2[[2]])
cv_probs_final<-predict(model_cv3, newx = as.matrix(test_final), type='response')
preds3_final<- ifelse(cv_probs_final > 0.5, 1, 0)

y.pred<-as.matrix(cbind(preds1_final, preds2_final$pred, preds3_final))

write.csv(y.pred, file = "pred_REICH_REAMES.csv", row.names = FALSE)



## TASK 2

dataset<-read_excel('assignment 2 task 2/BreastTissue.xls', sheet='Data')

# treat carcinoma as one group, fad-mas-gla as another group (glandular, mastopathy, fibro-adenoma), 
# then ac refers to adipose and connective tissue.

new_group_names<-factor(c(rep("car", 21), rep("fad-mas-gla", 49), rep("ac", 36)))
cancer_group_factors<-factor(c(rep("cancer", 54), rep("benign", 52)))
new_data_frame<-data.frame(class=new_group_names, dataset[, -c(1,2)])
cancer_variables<-new_data_frame[, -1]
actual_class<-new_data_frame[, 1]

# also create a dataframe for cancer vs non-cancerous tissue:
binary_data_frame<-data.frame(class=cancer_group_factors, dataset[, -c(1,2)])

# holdout strategy of 90% train, 10% test
partitions<-holdout(0.9, new_data_frame)
train<-partitions[[1]]
test<-partitions[[2]]
mod_lda_can<-lda(class~., data=train)
YPredLDA_can <- predict(mod_lda_can, test[, -1])$class
log_loss_lda_can<-(1/nrow(test))*length(which(YPredLDA_can!=test$class))
# only .272 or 27% wrong -> pretty low error.

# holdout strategy of 70% train, 30% test
partitions_2<-holdout(0.7, new_data_frame)
train_2<-partitions_2[[1]]
test_2<-partitions_2[[2]]
mod_lda_can_2<-lda(class~., data=train_2)
YPredLDA_can_2 <- predict(mod_lda_can, test_2[, -1])$class
log_loss_lda_can_2<-(1/nrow(test_2))*length(which(YPredLDA_can_2!=test_2$class))
# not too bad -> only .156 log_loss.

mod_qda_can<-lda(class~., data=train_2)
YPredQDA_can <- predict(mod_qda_can, test_2[, -1])$class
log_loss_qda_can<-(1/nrow(test_2))*length(which(YPredQDA_can!=test_2$class))
# 25% error rate; not as low error as lda.

partimat(class~I0+DA, data=train_2, method='lda')

x.new <- matrix(seq(min(train$DA), max(train$DA), length = 100), ncol = 1) 

experiment_reg<- ksmooth(train[,2], y=train[, 1], kernel = "normal", bandwidth = 150,
        x.points = x.new)


# holdout strategy of 70% train, 10% test on cancer vs non-cancerous tissue
partitions_bin<-holdout(0.7, binary_data_frame)
train_bin<-partitions_bin[[1]]
test_bin<-partitions_bin[[2]]
mod_lda_bin<-lda(class~., data=train_bin)
YPredLDA_bin <- predict(mod_lda_bin, test_bin[, -1])$class
log_loss_lda_bin<-(1/nrow(test_bin))*length(which(YPredLDA_bin!=test_bin$class))
# 12.5% wrong -> pretty low error for cancer vs non-cancer grouping.

mod_qda_bin<-lda(class~., data=train_bin)
YPredQDA_bin <- predict(mod_lda_bin, test_bin[, -1])$class
log_loss_qda_bin<-(1/nrow(test_bin))*length(which(YPredQDA_bin!=test_bin$class))
# same loss as lda...

# perform PCA dimensionality reduction:

dataset.pca<-prcomp(dataset[,c(3:11)], center = TRUE, scale=TRUE, retx=TRUE)
sum.pca<-summary(dataset.pca)

# second option for pca:

pca.alt<-princomp(dataset[, c(3:11)])
scores.v2<-pca.alt$scores[, 1:2]

##
sorted_loadings_1<-sort(dataset.pca$rotation[, 1], decreasing = TRUE)[1:3]
sorted_loadings_2<-sort(dataset.pca$rotation[, 2], decreasing = TRUE)[1:3]

pca1<-sorted_loadings_1[1]*dataset$PA500+sorted_loadings_1[2]*dataset$`Max IP`+sorted_loadings_1[3]*dataset$P
pca2<-sorted_loadings_2[1]*dataset$PA500+sorted_loadings_2[2]*dataset$HFS+sorted_loadings_2[3]*dataset$`A/DA`

reduced.dataset<-data.frame(class=dataset$Class, cbind(pca1, pca2))
partitions_red<-holdout(0.7, reduced.dataset)
train_red<-partitions_red[[1]]
test_red<-partitions_red[[2]]
mod_lda_red<-lda(class~., data=train_red)
YPredLDA_red <- predict(mod_lda_red, test_red[, -1])$class
log_loss_lda_red<-(1/nrow(test_red))*length(which(YPredLDA_red!=test_red$class))
# .46% loss; much worse than non-reduced dataset.

reduced.dataset_bin<-data.frame(class=cancer_group_factors, cbind(pca1, pca2))
partitions_red_bin<-holdout(0.7, reduced.dataset_bin)
train_red_bin<-partitions_red_bin[[1]]
test_red_bin<-partitions_red_bin[[2]]
mod_lda_red_bin<-lda(class~., data=train_red_bin)
YPredLDA_red_bin <- predict(mod_lda_red_bin, test_red_bin[, -1])$class
log_loss_lda_red_bin<-(1/nrow(test_red_bin))*length(which(YPredLDA_red_bin!=test_red_bin$class))
# only 9.375% loss...better than non-reduced data

mod_qda_red<-qda(class~., data=train_red_bin)
YPredQDA_red_bin <- predict(mod_qda_red, test_red_bin[, -1])$class
log_loss_qda_red<-(1/nrow(test_red_bin))*length(which(YPredQDA_red_bin!=test_red_bin$class))
# 12.5% loss

# can we predict carcinoma vs non-carcinoma? do some log-reg to find out...
carcinoma_count<-sum(dataset$Class=='car')
non_carcinoma_count<-sum(dataset$Class!='car')

carcinoma_group_factors<-factor(c(rep("carcinoma", carcinoma_count), rep("non-carcinoma", non_carcinoma_count)))
carc_data_frame<-data.frame(class=carcinoma_group_factors, dataset[, -c(1,2)])

part_car<-holdout(0.7, carc_data_frame)

train_car<-part_car[[1]]
test_car<-part_car[[2]]

log_reg_carc<-glm(class~., data=train_car, family=binomial)
preds_carc_log<-predict(log_reg_carc, test_car[, -1], type='response')
preds_carc_log <- ifelse(preds_carc_log > 0.5, 'non-carcinoma', 'carcinoma')

log_loss_carc<-(1/length(preds_carc_log))*length(which(factor(preds_carc_log)!=test_car$class))

# approx 31% log-loss errror for simply additive logistic regression model on carcinoma vs non carcinoma for
# reduced dataset.

# try and do knn classification:

opt_k_reduced_bin<-choose_k(train_car[, -1], test_car[,-1], train_car[, 1], test_car[, 1], reg=FALSE)

opt_k_reduced_bin[[1]]
## [1] 0.03125

# only 3.1% error; better than the paper!

opt_k_reduced_bin[[2]]
## [1] 3
# optimal 'k' is 3...


dataset.pca<-prcomp(dataset[,c(3:11)], center=TRUE, scale=TRUE)
summary(dataset.pca)

#ggbiplot(dataset.pca,ellipse=TRUE,groups=group_factors)
#ggbiplot(dataset.pca,ellipse=TRUE,groups=new_group_factors)
#ggbiplot(dataset.pca,ellipse=TRUE,groups=cancer_group_factors)

#ggbiplot(dataset.pca,choices=2:3, ellipse=TRUE, groups=group_factors)
#ggbiplot(dataset.pca,choices=2:3, ellipse=TRUE,groups=new_group_factors)
#ggbiplot(dataset.pca,choices=2:3, ellipse=TRUE,groups=cancer_group_factors)

#ggbiplot(dataset.pca,choices=c(1,3), ellipse=TRUE, groups=group_factors)
#ggbiplot(dataset.pca,choices=c(1,3),ellipse=TRUE,groups=new_group_factors)
#ggbiplot(dataset.pca,choices=c(1,3),ellipse=TRUE,groups=cancer_group_factors)


db_out<-dbscan(dataset[,3:11],eps=.15,MinPts=5)
fviz_cluster(db_out, dataset[,3:11], geom = "text")


mat_dataset<-unique(as.matrix(dataset[,c(3:11)]))
set.seed(23)
tsne_out<-Rtsne(mat_dataset,pca=TRUE,dims=3, normalize=TRUE,pca_center=TRUE, pca_scale=TRUE,initial_dims=3,perplexity=20,theta=0.0)


colors = rainbow(length(unique(dataset$Class)))
names(colors) = unique(dataset$Class)
par(mgp=c(2.5,1,0))
plot(tsne_out$Y, t='n', main="tSNE", xlab="tSNE dimension 1", ylab="tSNE dimension 2", "cex.main"=2, "cex.lab"=1.5)
text(tsne_out$Y, labels=dataset$Class, col=colors[dataset$Class])
plot(tsne_out$Y[,2:3], t='n', main="tSNE", xlab="tSNE dimension 2", ylab="tSNE dimension 3", "cex.main"=2, "cex.lab"=1.5)
text(tsne_out$Y[,2:3], labels=dataset$Class, col=colors[dataset$Class])


plot3d(tsne_out$Y,col=as.integer(group_factors),labels=group_factors)
legend3d("topright", legend =c('adi','car','conn','fad','gla','mas'), pch = 16)


# naive bayes without kde (using gaussian feature assumption)
m<-NaiveBayes(train_2[,-1],train_2[, 1])
preds_m<-predict(m, test_2[, -1], type='response')
# Use the 0-1 loss function: 
log_loss_m<-(1/nrow(test_2))*length(which(test_2[,1]!=preds_m$class))
#[1] 0.0625

# naive bayes with kde (setting 'usekernel'=TRUE)
m_k<-NaiveBayes(train_2[,-1],train_2[, 1], usekernel = TRUE)
preds_mk<-predict(m_k, test_2[, -1], type='response')
# Use the 0-1 loss function: 
log_loss_mk<-(1/nrow(test_2))*length(which(test_2[,1]!=preds_mk$class))
#[1] 0.03125

f<-data.frame('KNN loss'=opt_k_reduced_bin[[1]], 'LDA loss'=log_loss_lda_can_2, 'QDA loss'=log_loss_qda_can, 'GNB loss'=log_loss_m, 'KDENB loss'=log_loss_mk)

# function to predict classsifier accuracy against is 'nbFuncs' available from the caret library.
control <- rfeControl(functions=nbFuncs, method="cv", number=10)

results <- rfe(train_2[, -1], train_2[,1], sizes=c(1:9), rfeControl=control)

results$bestSubset
#[1] 5
results$optVariables
# [1] "I0"    "P"     "PA500" "DA"    "Area" 

plot(results$results$Variables, results$results$Accuracy)


x<-tsne_out$Y[,1]
y<-tsne_out$Y[,1]
z<-tsne_out$Y[,1]
scatter3D(x, y, z, bty = "f", pch = 18, 
          col.var = dataset$Class, 
          pch = 18, ticktype = "detailed",
          colkey = FALSE, side = 1, 
          addlines = TRUE, length = 0.5, width = 0.5,
          labels =dataset$Class, phi=0, theta=0)
