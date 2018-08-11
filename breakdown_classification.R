## Author: Gautam Kumar, PGDBA, IIM Calcutta, IIT Kharagpur and ISI Kolkata.
library(caret)
library(randomForest)


oven_data_1 <- read.csv("Reflow_Oven_Analytics\\reflow_oven_data\\Line_B1_Oven.csv",header = TRUE);
ncol(oven_data_1)

oven_data_1$Time[1]
oven_data_1$Time[nrow(oven_data_1)-1]


total_observation_no_1 <- nrow(oven_data_1)
na_presewnt <- is.na(oven_data_1)
any(na_presewnt)

## Remove the firt 3 columns, ans they does not look important for the modeling.
oven_data_1 <- oven_data_1[,-c(1:3)]

# Create a new variable which indicates valid failure based on numPcbs and Red Light.
oven_data_1$valid_failure <- (oven_data_1$numPcbs > 0) & (oven_data_1$Red_light == 1)
nrow(oven_data_1[oven_data_1$valid_failure==1,])

## Remove the yellow and green light columns as I will be modelling based on Red light column.
oven_data_1 <- oven_data_1[ , -which(names(oven_data_1) %in% c("Yellow_light","Red_light","Green_light"))]
ncol(oven_data_1)



failure_indices <- oven_data_1$valid_failure==1
valid_failure_df_1 <- oven_data_1[failure_indices,]

## Over Sampling of the Failure data.
failure_overSample_df <- data.frame()
for(i in 1:5000) {
  x1 <- sample(1112,1)
  failure_overSample_df <- rbind(failure_overSample_df,valid_failure_df_1[x1,])
}

valid_failure_df_1 <- rbind(valid_failure_df_1,failure_overSample_df)

normal_oven_1_df  <- oven_data_1[-failure_indices,]
normal_sample_indices <- sample(nrow(normal_oven_1_df),6200)

normal_samplle_oven_1_df <- normal_oven_1_df[normal_sample_indices,]

first_oven1_sample <- rbind(valid_failure_df_1,normal_samplle_oven_1_df)
train_indices <- sample(nrow(first_oven1_sample), ceiling(nrow(first_oven1_sample)*0.70))
train_oven_1_df <- first_oven1_sample[train_indices,]
test_oven_1_df <- first_oven1_sample[-train_indices,]


train_oven_1_df$valid_failure <- as.factor(train_oven_1_df$valid_failure)
first_oven1_sample$valid_failure <- as.factor(first_oven1_sample$valid_failure)


## Since Valid failure in being calculated based on numPcbs,,they will be correlated
## so I am removing the numPcbs column.
train_oven_1_df <- train_oven_1_df[ ,-which(names(train_oven_1_df) %in% c("numPcbs"))]

## Do the full model using random forest for featture selection.
bag.oven <- randomForest(valid_failure~., data=train_oven_1_df, importance=TRUE,
                                   mtry=7,proximity=TRUE)

importance(bag.oven)
## Plot the variable importance graph.
varImpPlot(bag.oven)

## Over Samlping Actual Modeling.

## Random Forest.
oven_overSample_rf <- randomForest(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length, data=train_oven_1_df, importance=TRUE,
                        mtry=7,proximity=TRUE)

library(e1071)
## Support vector Machine.
oven_overSample_svm <- svm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,train_oven_1_df)

## Decision Tree.
library(tree)
oven_overSample_dt <- tree(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,data = train_oven_1_df)

## Plot the Decision tree.
plot(oven_overSample_dt )
text(oven_overSample_dt  ,pretty =0)

## Do the cross validation decision tree

cv.oven_overSample_dt  =cv.tree(oven_overSample_dt ,FUN=prune.misclass )
plot(cv.oven_overSample_dt )

par(mfrow =c(1,2))

prune.oven =prune.misclass (oven_overSample_dt ,best =7)
plot(prune.oven )
text(prune.oven ,pretty =0)

## Logistic regression.
library(glm)
fullmod = glm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,data = train_oven_1_df,family=binomial(link = "logit"))
summary(fullmod)




## Prepare the test set by selecting exact variables which has been used in creating
## the model.
test_oven_1_df_p <- test_oven_1_df[ ,which(names(test_oven_1_df) %in% c("Boards_Produced","O2_PPM","Exit_sensor","Air_value","Z8Tact","Program","Z9Tact","Z8Bact","Z1Bact","Z2Bact","Z6Bact","Z7Bact","Z6Tact","N2_2","Z7Tact","Z6Tact","Z5Bact","Entrance_sensor","Cvr_width","Z3Bact","Z2Tact","Z5Tact","Pcb_length"))]

## Do the prediction on the test set, using models made.
oven_predict_rf = predict (oven_overSample_rf ,newdata =test_oven_1_df_p)
oven_predict_svm = predict (oven_overSample_svm ,newdata =test_oven_1_df_p)
oven_predict_tree = predict (oven_overSample_dt,newdata =test_oven_1_df_p,type = "class")


test_oven_1_df_p$valid_failure <- test_oven_1_df$valid_failure

test_oven_1_df_p$oven_predict_rf <- oven_predict_rf
test_oven_1_df_p$oven_predict_svm <- oven_predict_svm
test_oven_1_df_p$oven_predict_tree <- oven_predict_tree

## Create the confusion matrix one by one by using above variables one at a time.
confusionMatrix(test_oven_1_df_p$oven_predict_tree , test_oven_1_df_p$valid_failure)






#Undersampling method for sampling for skewed data set.

valid_failure_df_1 <- oven_data_1[failure_indices,]

normal_oven_1_df  <- oven_data_1[-failure_indices,]
normal_sample_indices <- sample(nrow(normal_oven_1_df),1200)

normal_samplle_oven_1_df <- normal_oven_1_df[normal_sample_indices,]

first_oven1_sample <- rbind(valid_failure_df_1,normal_samplle_oven_1_df)
train_indices <- sample(nrow(first_oven1_sample), ceiling(nrow(first_oven1_sample)*0.70))
train_oven_1_df <- first_oven1_sample[train_indices,]
test_oven_1_df <- first_oven1_sample[-train_indices,]



train_oven_1_df$valid_failure <- as.factor(train_oven_1_df$valid_failure)
first_oven1_sample$valid_failure <- as.factor(first_oven1_sample$valid_failure)


ncol(first_oven1_sample)
train_oven_1_df <- train_oven_1_df[ ,-which(names(train_oven_1_df) %in% c("numPcbs"))]


bag.oven <- randomForest(valid_failure ~ ., data=train_oven_1_df, importance=TRUE,
                         mtry=8,proximity=TRUE)

library(e1071)


importance(bag.oven)

varImpPlot(bag.oven)


##Actual Modelling.

oven_underSample_rf <- randomForest(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length, data=train_oven_1_df, importance=TRUE,
                                   mtry=7,proximity=TRUE)

library(e1071)
oven_underSample_svm <- svm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,train_oven_1_df)

library(tree)
oven_underSample_dt <- tree(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,data = train_oven_1_df)

## Logistic regression.
install.packages("glm2")
library(glm2)
oven_underSample_logit = glm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact + Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact + Pcb_length,data = train_oven_1_df,family=binomial);
summary(oven_underSample_logit)

## Plot the Decision tree.
plot(oven_underSample_dt )
text(oven_underSample_dt  ,pretty =0)

## Do the cross validation decision tree

cv.oven_underSample_dt  =cv.tree(oven_underSample_dt ,FUN=prune.misclass )
plot(cv.oven_underSample_dt )

##par(mfrow =c(1,2))

prune.oven =prune.misclass (oven_overSample_dt ,best =9)
plot(prune.oven )
text(prune.oven ,pretty =0)




test_oven_1_df_p <- test_oven_1_df[ ,which(names(test_oven_1_df) %in% c("Boards_Produced","O2_PPM","Exit_sensor","Air_value","Z8Tact","Program","Z9Tact","Z8Bact","Z1Bact","Z2Bact","Z6Bact","Z7Bact","Z6Tact","N2_2","Z7Tact","Z6Tact","Z5Bact","Entrance_sensor","Cvr_width","Z3Bact","Z2Tact","Z5Tact","Pcb_length"))]

oven_predict_rf = predict (oven_underSample_rf ,newdata =test_oven_1_df_p)
oven_predict_svm = predict (oven_underSample_svm ,newdata =test_oven_1_df_p)
oven_predict_tree = predict (oven_underSample_dt,newdata =test_oven_1_df_p,type = "class")
oven_predict_logit = predict.lm(oven_underSample_logit,newdata =test_oven_1_df_p,type = "response")



test_oven_1_df_p$valid_failure <- test_oven_1_df$valid_failure

test_oven_1_df_p$oven_predict_rf <- oven_predict_rf
test_oven_1_df_p$oven_predict_svm <- oven_predict_svm
test_oven_1_df_p$oven_predict_tree <- oven_predict_tree
test_oven_1_df_p$oven_predict_logit <- (oven_predict_logit > 0.6)


confusionMatrix(test_oven_1_df_p$oven_predict_tree , test_oven_1_df_p$valid_failure)






## SMOTE.
library(caret)
splitIndex <- createDataPartition(oven_data_1$valid_failure, p = .50,
                                  list = FALSE,
                                  times = 1)
trainSplit <- oven_data_1[ splitIndex,]
testSplit <- oven_data_1[-splitIndex,]

prop.table(table(trainSplit$valid_failure))

library(DMwR)
trainSplit$valid_failure <- as.factor(trainSplit$valid_failure)
trainSplit <- SMOTE(valid_failure ~ ., trainSplit, perc.over = 100, perc.under=200)
trainSplit$valid_failure <- as.factor(trainSplit$valid_failure)

prop.table(table(trainSplit$valid_failure))

library(randomForest)
rf.oven <- randomForest(valid_failure ~ ., data=trainSplit, importance=TRUE,
                         mtry=8,proximity=TRUE)



importance(bag.oven)

varImpPlot(bag.oven)



##Actual Modelling.

##Actual Modelling.

oven_smote_rf <- randomForest(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor 
                              + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact
                              + Z2Bact + Z6Bact + Z7Bact + N2_2+N2_1 + Z7Tact +
                                Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact 
                              + Z2Tact + Z5Tact + Pcb_length+Z1Tact, data=train_oven_1_df, 
                              importance=TRUE,
                                    mtry=7,proximity=TRUE)

library(e1071)
oven_smote_svm <- svm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor 
                      + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact
                      + Z2Bact + Z6Bact + Z7Bact + N2_2+N2_1 + Z7Tact +
                        Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact 
                      + Z2Tact + Z5Tact + Pcb_length+Z1Tact,train_oven_1_df)

library(tree)
oven_smote_dt <- tree(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor +
                        Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact +
                        Z2Bact + Z6Bact + Z7Bact + Z6Tact + N2_2 + Z7Tact + Z6Tact +
                        Z5Bact + Entrance_sensor + Cvr_width + Z3Bact + Z2Tact + Z5Tact +
                        Pcb_length,data = train_oven_1_df)


## Logistic regression.
library(glm)
oven_smote_logit = glm(valid_failure ~ Boards_Produced + O2_PPM + Exit_sensor 
              + Air_value + Z8Tact +Program + Z9Tact + Z8Bact + Z1Bact
              + Z2Bact + Z6Bact + Z7Bact + N2_2+N2_1 + Z7Tact +
                Z6Tact + Z5Bact + Entrance_sensor + Cvr_width + Z3Bact 
              + Z2Tact + Z5Tact + Pcb_length+Z1Tact,data = train_oven_1_df,family=binomial(link = "logit"))
summary(oven_smote_logit)


test_oven_1_df_p <- test_oven_1_df[
  ,which(names(test_oven_1_df) %in% c("Boards_Produced","O2_PPM","Exit_sensor",
                                      "Air_value","Z8Tact","Program","Z9Tact",
                                      "Z8Bact","Z1Bact","Z2Bact","Z6Bact","Z7Bact",
                                      "N2_2","N2_1","Z7Tact","Z6Tact","Z5Bact",
                                      "Entrance_sensor","Cvr_width","Z3Bact","Z2Tact",
                                      "Z5Tact","Pcb_length","Z1Tact"))]


## Plot the Decision tree.
plot(oven_smote_dt )
text(oven_smote_dt  ,pretty =0)

## Do the cross validation decision tree

cv.oven_smote_dt  =cv.tree(oven_smote_dt ,FUN=prune.misclass )
plot(cv.oven_smote_dt )

##par(mfrow =c(1,2))

prune.oven =prune.misclass (oven_smote_dt ,best =7)
plot(prune.oven )
text(prune.oven ,pretty =0)





oven_predict_rf = predict (oven_smote_rf ,newdata =test_oven_1_df_p)
oven_predict_svm = predict (oven_smote_svm ,newdata =test_oven_1_df_p)
oven_predict_tree = predict (oven_smote_dt,test_oven_1_df_p,type = "class")
oven_predict_logit = predict.lm(oven_smote_logit,newdata =test_oven_1_df_p,type = "response")



test_oven_1_df_p$valid_failure <- test_oven_1_df$valid_failure

test_oven_1_df_p$oven_predict_rf <- oven_predict_rf
test_oven_1_df_p$oven_predict_svm <- oven_predict_svm
test_oven_1_df_p$oven_predict_tree <- oven_predict_tree



test_oven_1_df_p$oven_predict_logit <- (as.numeric(oven_predict_logit)  > 0.6)


confusionMatrix(test_oven_1_df_p$oven_predict_logit , test_oven_1_df_p$valid_failure)












## See the distribution of worktypes.
reflow_word_order_df <- read.csv("Reflow_Oven_Analytics\\reflow_oven_data\\Reflow_Oven_Work_Order_Data_Dump_1.csv",header = TRUE)
worktype_df <- as.data.frame(table(reflow_word_order_df$Worktype))
worktype_df <- worktype_df[-1,]
worktype_df$perc <- (worktype_df$Freq / sum(worktype_df$Freq))*100
write.csv(worktype_df,file = "worktype.csv")

worktype_ppm_df = reflow_word_order_df[reflow_word_order_df$Worktype == "PPM",]




nzv <- nearZeroVar(oven_data_1, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]



valid_Failures_df_1 <- oven_data_1[(oven_data_1$numPcbs > 0) & (oven_data_1$Red_light == 1),]
nrow(valid_Failures_df_1)

valid_Failures_percentage = nrow(valid_Failures_df_1)/total_observation_no_1





valid_failure_df_1 <- pos_pcb_df_1[(pos_pcb_df_1$Red_light == 1),]

nrow(valid_failure_df_1)

nrow(pos_pcb_df_2[pos_pcb_df_2$Red_light == 1 ,])

red_light_percentage <- red_light_no / total_observation_no

## Total p


oven_data_1$Time[1]
## Time Duration of oven 1 data.
as.POSIXct(as.character(oven_data_1$Time[1]),tz = "UTC")

