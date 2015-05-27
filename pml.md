

```r
library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
options(warn=-1)
```


Setting the seed:


```r
set.seed(1234)
```

Let us load the data and get rid of errorneous data points:


```r
trainData <- read.csv("~/Downloads/pml-training.csv", na.strings=c("#DIV/0!") )
evalData  <- read.csv("~/Downloads/pml-testing.csv", na.strings=c("#DIV/0!") )
```

Better transform columns to numeric:


```r
    for(i in 8:(ncol(trainData)-1)) {
        trainData[,i] = as.numeric(as.character(trainData[,i]))
    }

for(i in 8:(ncol(evalData)-1)) {
    evalData[,i] = as.numeric(as.character(evalData[,i]))
}
```

What is the feature data:


```r
featureData <- colnames(trainData[colSums(is.na(trainData)) == 0])[-(1:7)]
modelData   <- trainData[featureData]

myIndex  <- createDataPartition(y=modelData$classe, p=0.75, list=FALSE)
training <- modelData[myIndex,]
testing  <- modelData[-myIndex,]
```
Let us build random forests with 100 trees each. doParallel helps to do parallel computing instead of running simple loops:


```r
depend   <- training[-NCOL(training)]
independ <- training$classe

registerDoParallel()
myRandomForest <- foreach(noTrees=rep(100, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
    randomForest(depend, independ, ntree=noTrees) 
}
```
Here is the error report:


```r
predictTraining <- predict(myRandomForest, newdata=training)
confusionMatrix(predictTraining,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictTesting <- predict(myRandomForest, newdata=testing)
confusionMatrix(predictTesting,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    3    0    0    0
##          B    0  944    8    0    0
##          C    0    2  846    6    0
##          D    0    0    1  798    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9959          
##                  95% CI : (0.9937, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9948          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9895   0.9925   1.0000
## Specificity            0.9991   0.9980   0.9980   0.9998   1.0000
## Pos Pred Value         0.9979   0.9916   0.9906   0.9987   1.0000
## Neg Pred Value         1.0000   0.9987   0.9978   0.9985   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1925   0.1725   0.1627   0.1837
## Detection Prevalence   0.2851   0.1941   0.1741   0.1629   0.1837
## Balanced Accuracy      0.9996   0.9964   0.9937   0.9961   1.0000
```

The test data has very high accuracy. It is quite satisfactory

Here we generate the submission files as provided by coursera:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

x <- evalData
x <- x[featureData[featureData!='classe']]
answers <- predict(myRandomForest, newdata=x)

pml_write_files(answers)
```
