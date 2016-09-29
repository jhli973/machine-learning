##Note from Coursera Practical Machine Learning - JHU

##Week 1
SPAM Example
```
plot(density(spam$your[spam$type=="nonspam"]),
        col = "blue", main="", xlab="Frequency of 'your'")
lines(density(spam$your[spam$type=="spam"]), col="red")
abline(v=0.5, col="black")
#pred
pred <-ifelse(spam$your > 0.5, "spam", "nonspam")
# accuracy
table(pred, spam$type)/length(spam$type)
```

**Prediction is about accuracy tradeoffs**
+ Interpretability versus accuracy (Interpretability matters)
+ Speed versus accuracy
+ Simplicity versus accuracy
+ Scalability versus accuracy (Scalability matters)

**In sample versus out of sample Error**
In sample Error - resubstitution error
Out of sample Error - generalization error (overfitting)

```
library(kernlab); data(spam); ser.seed(333)

#sample 10 samples
smallSpam <- spam[sample(dim(spam)[1], size =10),]
spamLabel <-(smallSpam$type=="spam") * 1 + 1

#plot the capitalAve versus index
plot(smallSpam$capitalAve, col=spamLabel)

#build a algorithm
rule1<- function(x){
	prediction <- rep(NA, length(x))
    prediction[x > 2.7] <- "spam"
    prediction[x < 2.7] <- "nonspam"
    prediction[(x >= 2.40 & x <=2.45)] <- "spam"
    prediction[(x > 2.45 & x <= 2.70)] <- "nonspam"
    return(prediction)
}
table(rule1(smallSpam$capitalAve), smallSpam$type)
mean(rule1(smallSpam$capitalAve)==smallSpam$type)
```

**Prediction study design**
+ Define your error rate
+ Split data into training, testing and validation(optional)
+ Use cross-validation pick features on the training set
+ Use cross-validation pikc prediction function on the training set
+ If no validation, apply the best model we have to test set one time
+ Avoid small sample sizes

**Some principles to remember**
+ Set the test/validation set aside and don't look at it.
+ In general randomly sample training and test
+ If predictions evolve with time split train/test in time chuncks (called backtesting in finance)

**Types of error**
In general, positive = Identified and negative = rejected
Positive: true positive and false positive
Negative: true negative and false negative

+ Key quantities:
 + Sensitivity (recall): Pr(positive test | disease) = TP/(TP + FN)
 + Specificity: Pr(negative test | no disease) = TN/(FP + TN)
 + Positive Predictive Value (precision): Pr(disease | positive test) = TP / (TP + FP)
 + Negative Predictive Value: Pr(no disease | negative test) = TN / (FN + TN)
 + Accuracy: Pr(correct outcome) = (TP + TN)/(TP + FP + TN + FN)
 + False positive rate (alpha) = type I error = 1 - specificity
 + False negative rate (beta) = type II error =  1 - sensitivity
 + Power = sensitivity = 1- beta

**Common error measures**
1. Mean squared error (MSE) or Root mean squared error (RMSE)
  + for continuous data, sensitive to outliers
2. Median absolute deviation
  + Continuous data, often more robust
3. Sensitivity (recall) and specificity
  + Often used in medical tests and widely used if you care about one type error more than the other type of error
4. Accuracy
  + Weights false positives and false negative equally - discrepancy
5. Concordance
  + For multiclass cases - kappa

**ROC Curves** - A graphical plot that illustrates the performance of a binary classifier system
x-axis: False positive rate (1 - specificity)
y-axis: True positive rate (also called sensitivity or recall)
AUC (area under the curve)
+ AUC = 0.5: random guessing
+ AUC = 1: perfect classifier
+ In general, above 0.8 consider as "good"

**Cross-validation**
Approach:
+ Split data into training and testing sets
+ Sub-split training set into sub-training/test sets
+ Build a model on the sub-training set
+ Evaluate the model on the sub-test set
+ Repeat and average the estimated errors

Used for:
+ Picking variables(features) to include a model
+ Picking the type of prediction function to use
+ Picking the parameters in the prediction function
+ Comparing different predictors

Methods(on training set only, for time series data, data must be used in "chuncks"):
+ Random subsampling
  + With replacement
  + Without replacement (bootstrap): Underestimate the error. Can be corrected by 0.632 bootstrap but it is complicated
+ K-fold
  + Larger K: less bias, more variance
  + Smaller K: more bias, less variance
+ Leave one out

##Week 2
**Caret functionality**
+ Preprocessing (cleaning)
  + preProcess
+ Data splitting
  + createDataPartition
  + createResample
  + createTimeSlices
+ Training/testing functions
  + train
  + predict
+ Model comparision
  + confusionMatrix

**SPAM Example**
```
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(train)

set.seed(23343)
modelFit<-train(type~.,data=training, methos="glm")
modelFit
modelFit$finalModel
pred<- predict(modelFit, newdata=testing)
pred
confusionMatrix(pred, testing$type)
```

**Data Slicing**
```
set.seed(32323)

#return training set
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
sapply(folds, length)

#return test set
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=FALSE)
folds[[1]][1:10]

#resampling with replacement
samples <- createResample(y=spam$type, times=10, list=TRUE)

#time slices
tme <- 1:1000
folds <- ceateTimeSlices(y=tme, initialWindow=20, horizon=10)
names(folds)

folds$train[[1]]
fold$test[[1]]
```
**Training options**
```
modelFit <- train(type~., data=training, method='glm')

args(train.default)

function(x, y, method='rf', preProcess=NULL, ..., weights=NULL,
metric=ifelse(is.factor(y), "Accuracy", "RMSE"),
maximize=ifelse(metric="RMSE", FALSE, TRUE),
trControl=trainControl(), tuneGrid=NULL, tuneLength=3)
```
**Metric options**
+ Continuous outcomes:
  + RMSE = Root mean squared error
  + RSquared = R^2 from regression models

+ Categorical outcomes:
  + Accuracy = Fraction correct
  + Kappa = A measure of concordance

**trainControl**
```
args(trainControl)
function (method="boot", number = ifelse(method %in% c("cv", "repeatedcv"), 10, 25),
repeats = ifelse(method %in% c("cv", "repeatedcv"), 1, number), p = 0.75, initialWindow = NULL,
horizon = 1, fixedWindow = TRUE, verboseIter = FALSE, returnData =TRUE,
returnResample = "final", savePredictions = FALSE, classProbs = FALSE,
summaryFunction = defaultSummary, selectionFunction = "best",
custom = NULL, preProcOptions = list(thresh = 0.95, ICAcomp = 3, k = 5),
index = NULL, indexOut = NULL, timingSamps = 0,
predictionBounds = rep(FALSE, 2), seed = NA, allowParallel = TRUE)
```
**trainControl resampling**
+ method
  + boot = bootstrapping
  + boot632 = bootstrapping with adjustment
  + cv = cross validation
  + repeatedcv = repeated cross validation
  + LOOCV = leave one out cross validation

+ number
  + For boot/cross validation
  + Number of subsamples to take

+ repeats
  + Number of times to repeat subsampling
  + If big this can slow things down

**Plotting predictors**
```
inTrain <- createDataPartition(y=Wage&wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

#Feature plot(caret package)
featurePlot(x=training[, c("wage", "education", "jobclass")], 
y=training$wage, plot="pairs")

# qplot (ggplot2 package)
q <- qplot(age, wage, colour=jobclass,  data=training)
q + geom_smooth(method='lm', formula=y~x)
```

**Cut2, making factors(Hmisc package)**
```
cutWage <- cut2(training$wage, g=3)
table(cutWage)

qplot(cutWage, age, data=training, fill=cutWage,
geom=c("boxplot"))
```
**Boxplot with points overlayed**
```
q1 <- qplot(cutWage, age, data=training, fill=cutWage,
geom=c("boxplot"))

q2 <- qplot(cutWage, age, data=training, fill=cutWage,
geom=c("boxplot", "jitter"))

grid.arrange(p1, p2, ncol=2)
```
**Density plots**
```
qplot(wage, colour=education, data=training, geom="density")
```
**Things you should be looking for**
+ Imbalance in outcomes/predictors
+ Outliers
+ Groups of points not explained by a predictor
+ Skewed varaibles

**Preprocess**
```
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve, main="", xlab="ave. capital run length")

mean(training$capitalAve)
sd(training$capitalAve)
```
**Standardizing variable** - reduces the variability
```
trainCapAve <- training$capitalAve
trainCapAves <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(trainCapAveS)  # equals to zero
sd(trainCapAveS)   # equals to one

#Apply the same standardizing to test set
testCapAve <- testing$capitalAve
testCapAves <- (testCapAve - mean(trainCapAve))/sd(trainCapAve)
```
**Standardizing - use preProcess function** - same result
```
preObj <- preProcess(training[,-58], method=c("center", "scale"))
trainCapAves <- predict(preObj, training[,-58])$capitalAve

testCapAves <- predict(preObj, testing[,-58])$capitalAve

#directly use in the train function
modelFit <- train(type~., data=training,
         preProcess=c("center", "scale"), method="glm")
```
**Standardizing - Box-Cox transforms**
```
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2))
hist(trainCapAveS); qqnorm(trainCapAveS)
```
**Standardizing - Imputing data**
```
# Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA] <- NA

# Impute and standardize use k-nearest neighbor imputation
preObj <- preProcess(training[,-58], method="knnImpute")
capAve <- predict(preObj, training[,-58])$capAve

# Standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth - mean(capAveTruth))/sd(capAveTruth)
```

**Notes and further reading**
+ Training and test must be processed in the same way
+ Test transformations will likely be imperfect
  + Especially ifthe test/training sets collected at different times
+ Careful when transforming factor variables


##Covariate (predictor/feature) creation

**Two levels of covariate creation**
Level 1: From raw data to covariate
Level 2: Transforming tidy covariates

**1. Convert factor variables to indicator variables**
```
dummies <- dummyVars(wage ~ jobclass, data=training)
head(predict(dummies, newdata=training))
```
**2. Removing zero covariates**
```
# Use nearZeroVar function to identify those variables
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv
```
**Splines basis** - allow curve fittinh
```
library(splines)
bsBasis <- bs(training$wage, df=3)
bsBasis

lm1 <- lm(wage ~ bsBasis, data=training)
plot(training$age, training$wage, pch=19, cex=0.5)
points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)

# Splines on the test set
predict(bsBasis, age=testing$age)
```

##Preprocessing with principle components analysis
```
library(caret); library(kernlab); data(spam)
inTrain <- createdataPartition(y=spam$type, p=0.75, list=FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

m <- abs(cor(training[,-58]))
diag(m) <- 0
which(m > 0.8, arr.ind=T)
```
**Correlated predictors**
```
names(spam)[c(34, 32)]

plot(spam[,34], spam[,32])
```
**Basic PCA idea**
+ We might not need every predictor
+ A weighted combination of predictors might be better
+ Pick a combination to capture the "most information" possible
+ Benefits
  + Reduced number of predictors
  + Reduced noise due to averaging

**Rotate the plot to see which way captures the most information**
x = 0.71 x num415 + 0.71 x num857
y = 0.71 x num415 - 0.71 x num857
```
X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training857
plot(X,Y)
```

**Principle**
+ Find a new set of multivariate varaibles that are uncorrelated and explain as much variance as possible
+ If you put all the variables together in one matrix, find the best matrix created with fewer variables (lower rank) that explains the original data

The first goal is statistical and the second goal is data compression.

**Related solutions - PCA/SVD**
SVD (singular value decomposition)
if X is a matrix with each variable in a column and each observation in a row the SVD is a "matrix decomposition"

X = UDV^T^

Where the columns of U are orthogonal

PCA
Equals to the right singular values if you first scale (subtract the mean, divide by the standard deviation) the variables.

**Principle components in R -prcomp** - allow you deal with more than two variables
```
smallSpam <- spam[, c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$X[,1], prComp$X[,2])

prComp$rotation

typeColor <- ((spam$type=="spam")*1 + 1)
prComp <- prcomp(log10(spam[,-58] + 1))
plot(prComp$X[,1], prComp$X[,2], col=typeColor, xlab="PC1", ylab="PC2")

#PCA with caret
preProc <- preProcess(log10(spam[,-58]+1), method="pca",pcaComp=2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col=typeColor)

preProc <- preProcess(log10(training[,-58]+1), method="pca",pcaComp=2)
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(training$type ~ ., method="glm", data=trainPC)

testPC <- predict(preProc, log10(tesing[,-58]+1))
confusionMatrix(testing$type, predict(modelFit, testPC))

# Alternative
modelFit <- train(training$type ~ ., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing))
```

**Final thoughts on PCs**
+ Most useful for linear-type models
+ Can make it harder to interpret predictors
+ Watch out for outliers
  + Transform first(with logs/Box Cox)
  + Plot predictors to identify problems


##Preprocessing with principle components analysis
```
library(caret); library(kernlab); data(spam)
inTrain <- createdataPartition(y=spam$type, p=0.75, list=FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

m <- abs(cor(training[,-58]))
diag(m) <- 0
which(m > 0.8, arr.ind=T)
```
**Correlated predictors**
```
names(spam)[c(34, 32)]

plot(spam[,34], spam[,32])
```
**Basic PCA idea**
+ We might not need every predictor
+ A weighted combination of predictors might be better
+ Pick a combination to capture the "most information" possible
+ Benefits
  + Reduced number of predictors
  + Reduced noise due to averaging

**Rotate the plot to see which way captures the most information**
x = 0.71 x num415 + 0.71 x num857
y = 0.71 x num415 - 0.71 x num857
```
X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training857
plot(X,Y)
```

**Principle**
+ Find a new set of multivariate varaibles that are uncorrelated and explain as much variance as possible
+ If you put all the variables together in one matrix, find the best matrix created with fewer variables (lower rank) that explains the original data

The first goal is statistical and the second goal is data compression.

**Related solutions - PCA/SVD**
SVD (singular value decomposition)
if X is a matrix with each variable in a column and each observation in a row the SVD is a "matrix decomposition"

X = UDV^T^

Where the columns of U are orthogonal

PCA
Equals to the right singular values if you first scale (subtract the mean, divide by the standard deviation) the variables.

**Principle components in R -prcomp** - allow you deal with more than two variables
```
smallSpam <- spam[, c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$X[,1], prComp$X[,2])

prComp$rotation

typeColor <- ((spam$type=="spam")*1 + 1)
prComp <- prcomp(log10(spam[,-58] + 1))
plot(prComp$X[,1], prComp$X[,2], col=typeColor, xlab="PC1", ylab="PC2")

#PCA with caret
preProc <- preProcess(log10(spam[,-58]+1), method="pca",pcaComp=2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col=typeColor)

preProc <- preProcess(log10(training[,-58]+1), method="pca",pcaComp=2)
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(training$type ~ ., method="glm", data=trainPC)

testPC <- predict(preProc, log10(tesing[,-58]+1))
confusionMatrix(testing$type, predict(modelFit, testPC))

# Alternative
modelFit <- train(training$type ~ ., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing))
```

**Final thoughts on PCs**
+ Most useful for linear-type models
+ Can make it harder to interpret predictors
+ Watch out for outliers
  + Transform first(with logs/Box Cox)
  + Plot predictors to identify problems


##Predicting with regression
Pros:
 + Easy to implement and intepret
Cons:
 + Often poor performance in nonlinear settings

**Example: Old faithful eruptions**
```
library(caret); data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p=0.5, list=FALSE)

trainFaith <- faithfull[inTrain, ]
testFaith <- faithful[-inTrain, ]
head(trainFaith)
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")

# Fit a linear model
lm1 <- lm(eruptions ~ waiting, data=trainFaith)
summary(lm1)

par(mfrow=c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
lines(trainFaith$waiting, lm1$fitted, lwd=3)
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
ines(trainFaith$waiting, predict(lm1, newdata=testFaith), lwd=3)

coef(lm1)[1] + coef(lm1)[2]*80

newdata <- data.frame(waiting=80)
predict(lm1, newdata)

# Calculate RMSE on training
sqrt(sum(lm1$fitted-trainFaith$eruptions)^2))

# Calculate RMSE on test
sqrt(sum(predict(lm1, newdata=testFaith)-testFaith$eruptions)^2))

# Add prediction intervals
pred1 <- predict(lm1, newdata=testFaith, interval="prediction")
ord <- order(testFaith$waiting)
plot(testFaithwaiting, testFaith$eruptions, pch=19, col="blue")
matlines(testFaith$waiting[ord], pred1[ord,], type="l",
col=c(1,2,2), lty=c(1,1,1), lwd=3)

# Predict with caret
modFit <- train(eruptions ~ waiting, data=trainFaith, metho="lm")
summary(modFit$finalModel)
```
##Predicting with regression Multiple Covariates

Example: Wage data
```
library(ISLR); library(ggplot2); library(caret)
data(Wage)
Wage <- subset(Wage, select=-c(logwage))
sumary(Wage)

inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
```
**Feature plot**
```
featurePlot(x=training[,c("age", "education", "jobclass")],
            y = training$wage, plot="pairs")

qplot(age, wage, data=training)
qplot(age, wage, colour=jobclass,data=training)
```
**Fit a linear model**
```
modFit <- train(wage ~ age + jobclass + education, method="lm", data=training)
finMod <- modFit$finalModel
print(modFit)
```
**Diagnostics**
```
plot(finMod, 1, pch=19, cex=0.5, col="#00000010")

# Color by variables not used in the model
qplot(finMod$fitted, finMod$residuals, colour=race, data=training)

# Plot by index - missing data
plot(finMod$residuals, pch=19)

# plot predicted value vs actual value
pred <- predict(modFit, testing)
qplot(wage, pred, colour=year, data=testing)

# Predict with all covariates
modFitAll <- train(wage ~ ., data=training, method="lm")
pred <- predict(modFitAll, testing)
qplot(wage, pred, data=testing)
```


##Quiz 2
```
install.packages("AppliedPredictiveModeling")
install.packages("caret")
install.packages("Hmisc")
library(AppliedPredictiveModeling)
library(Hmisc)
library(caret)
data(concrete)

set.seed(1000)

inTrain=createDataPartition(mixtures$CompressiveStrength,p=3/4)[[1]]
head(inTrain)

training=mixtures[inTrain,]
testing=mixtures[-inTrain,]
summary(training)

z=cut2(mixtures$Age, g=4)
plot(mixtures$CompressiveStrength,pch=19, col=z)

hist(mixtures$Superplasticizer)

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[inTrain,]
testing = adData[-inTrain,]


preProc <-preProcess(training[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3","IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )], method='pca', thresh=0.90)
pcaPC = predict(preProc, training[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3","IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )])
names(pcaPC)

#predict with PCA
preProc <-preProcess(training[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3","IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )], method='pca', thresh=0.80)
trainPC <- predict(preProc, training[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3","IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )])

modFitPC <- train(training$diagnosis ~ ., method="glm", data=trainPC)
testPC <- predict(preProc, testing[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3"                             ,"IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )])
confusionMatrix(testing$diagnosis, predict(modFitPC, testPC))

modFit <- train(training$diagnosis ~ IL_11+IL_13+IL_16+IL_17E+IL_1alpha+IL_3+IL_4+IL_5+IL_6+IL_6_Receptor+IL_7+IL_8, method="glm", data=training)
testPred <- predict(modFit, testing[,c( "IL_11","IL_13","IL_16","IL_17E","IL_1alpha","IL_3"                             ,"IL_4","IL_5","IL_6","IL_6_Receptor","IL_7","IL_8"  )])
confusionMatrix(testing$diagnosis,testPred)
```

##Predicting with trees

**Basic algorithm**
+ Start with all variables in one group
+ Find the variables/split that best seperates the outcomes
+ Divide the data into two groups ("leaves") on that split ("Node")
+ Within each split, find the best variables/split that separates the outcomes
+ Continue until the groups are too small or sufficiently "pure"

**Measures of impurity**
+ Misclassification Error:   1 - Phatmk(m)
  0 -- perfect purity ; 0.5 -- no purity

+ Gini index: 
  0 -- perfect purity ; 0.5 -- no purity

+ Deviance/information gain:
  0 -- perfect purity ; 1 -- no purity

![](C:\CourseNote\machineLearning2.PNG)

**Example: Iris data**
```
data(iris); library(ggplot2)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

qplot(Petal.Width, Sepal.Width, colout=Species, data=training)

library(caret)
modFit <- train(Species ~., method="rpart", data=training)
print(modFit$finalModel)

# Plot tree
plot(modFit$finalModel, uniform=TRUE, main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# Prettier plots
library(rattle)
fancyRpartPlot(modFit$finalModel)

predict(modFit, newdata=testing)
```

**Notes**
+ Classification trees are non-linear models
  + They use interactions between variables
  + Data transformations may be less important (monotone transformations)
  + Trees can also be used for regression problems
  + R packages for tree building
    + in caret package: party, rpart
    + tree

## Bagging - short for bootstrap aggregating
**Basic idea:**
1. Resample cases and recalculate predictions
2. Average or majority vote

**Notes:**
+ Get similar bias
+ Reduced variance
+ More useful for non-linear functions

```
library(ElemStatLearn); data(ozone, package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)
```

**Bagged loess**
```
ll <- matrix(NA, nrow=10, ncol=155)
for (i in 1:10){
 ss <- sample(1:dim(ozone)[1], replace=TRUE)
 ozone0 <- ozone[ss,]
 ozone0 <- ozone0[order(ozone0$ozone0),]
 loess0 <- loess(temperature ~ ozone, data = ozone0, span=0.2)
 ll[i,] <- predict(loess, newdata = data.frame(ozone=1:155))
 }

# Plot
plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){lines(1:155, ll[i,], col="grey",lwd=2)}
lines(1:155, apply(ll,2,mean),col="red",lwd=2)
```
**Bagging in caret**
+ Some models perform bagging for you, in train function consider method options
  + bagEarth
  + treebag
  + bagFDA

+ Alternatively you can bag any model you choose using the bag function

```
predictors <- data.frame(ozone=ozone$ozone)
temperature <- ozone$temperature
treebag <- bag(predictors, temperature, B=10,
       bagControl = bagControl(fit=ctreeBag$fit,
                               predict=ctreeBag$pred,
                               aggregate = ctreeBag$aggregate))

plot(ozone$ozone, temperature, col="lightgrey", pch=19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), pch=19, col='red')
points(ozone$ozone, predict(treebag, predictors), pch=19, col='blue')
```

**Notes and further resources**
+ Bagging is most useful for nonlinear models
+ Often used with trees - an extension is random forests
+ Several models use bagging in caret's train function

## Random forests
1. Bootstrap samples
2. At each split, bootstrap variables
3. Grow multiple trees and vote

**Pros:**
+ Accuracy

**Cons:**
+ Speed
+ Interpretability
+ Overfitting

**Example of random forests**
```
data(iris); library(ggplots)
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species ~ ., data = training, method = "rf", prox = TRUE)
modFit

# Get a specific tree ( 2nd)
getTree(modFit$finalModel, k=2)
```

**Class "center"**
```
irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP);
irisP$Species <- rownames(irisP)

p <- qplot(Petal.Width, Petal.Length, col=Species, data=training)
p + geom_point(aes(x=Petal.Width, y=Petal.Length, col=Species), size=5, shape=4, data=irisP)
```

**Predicting new values**
```
pred <- predict(modFit, testing);
testing$predRight <- pred==testing$Species
table(pred, testing$Species)

qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main="newData predictions")
```

**Notes**
+ Random forests are usually one of the two top performing algorithms along with boosting in prediction contests.
+ Random forests are difficult to interpret but often accurate
+ Care should be taken to avoid overfitting (see rfcv function)

##Boosting
**Basic idea**
1. Take lots of (possibly) weak predictors
2. Weight them and add them up
3. Get a stronger predictor

**Basice steps behind boosting**
1. Sart with a set of classifiers h1, ..., hk
  Examples: all possible trees, all possible regression models, all possible cutoffs.

2. Create a classifier that combines classification functions: f(x) = sgn(sum(aihi(x)))
  + Goals is to minimize error (on training set)
  + Iterative, select one h at each step
  + Calculate weight based on errors
  + Upweight missed classifications and select next h

check Adaboost on wikipedia

**Boosting in R**
+ Boosting can be used with any subset of classifiers
+ One large subclass is gradient boosting
+ R has multiple boosting libraries
  + gbm -- boosting with tree
  + mboost -- model based boosting
  + ada -- statistical boosting based on additive logistic regression
  + gamBoost -- for boosting generalized additive models
+ Most of these are available in the caret package

**Wage Example**
```
library(ISLR); data(Wage); library(ggplot2); library(caret)
Wage <- subset(Wage, select=-c(logwage)
inTrain <- createDataPaitition(y=Wage$wage, p=0.7, list=FALSE)

training <- Wage[inTrain,];
testing <- Wage[-inTrain,]

modFit <- train(wage ~., method="gbm", data=training, verbose=FALSE)
print(modFit)

qplot(predict(modFit, testing), wage, data=testing)
```

##Model based prediction
**Basic idea**
1. Assume the data follow a probabilistic model
2. Use Bayes' theorem to identify optimal classifiers

Pros:
+ can take advantage of structure of the data
+ may be computationally convienient
+ are reasonably accurate on real problems

Cons:
+ make additional assumptions about the data
+ when the model is incorrect you may get reduced accuracy

Linear Discriminant Analysis
Quadratic Discriminant Analysis
Dscidion boundery
Naive Bayes
```
data(iris); library(ggplot2); library(caret)
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

dim(taining); dim(testing)

modlda = train(Species ~ ., data=training, method="lda")
modnb = train(Species ~ ., data=training, method="nb")

plda = predict(modlda, testing)
pnb = predict(modnb, testing)

table(plda, pnb)

equalPredictions = (plda==pnb)
qplot(Petal.Width, Sepal.Width, colour=equalPredictions, data=testing)
```

**Regularization**

**Lasso**

**Approaches for combining classifiers**

1. Bagging, boosting and random forests
  + Usually combine similar classifiers
2. Combining different classifiers
  + Model stacking
  + Model ensembling

**Example of combining classifiers with wage data**
```
library(ISLR); data(Wage); library(ggplot2); library(caret)
Wage <- subset(Wage, select=-c(logwage))

# Create a building data set and validation set
inBuild <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
validation <- Wage[-inBuild,]; buildData <- Wage[inBuild,]

inTrain <- createDataPartition(y=buildData$wage, p=0.7, list=FALSE)
training <- buildData[inTrain,]; testing <- buildData[-inTrain,]

# Build two different models
mod1 <- train(wage ~., method="glm", data=training)
mod2 <- train(wage ~., method="rf", data=training, trControl = trainControl(method="CV"), number=3)

pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
qplot(pred1, pred2, colour=wage, data=testing)

# Fit a model to the predictions
predDF <- data.frame(pred1, pred2, wage=testing$wage)
combModFit <- train(wage ~., method="gam", data=predDF)
combPred <- predict(combModFit, predDF)

# Check testing error
sqrt(sum(pred1 - testing$wage)^2))
sqrt(sum(pred2 - testing$wage)^2))
sqrt(sum(combPred - testing$wage)^2))

# Evaluate on validation
```
































































































































































































































































































