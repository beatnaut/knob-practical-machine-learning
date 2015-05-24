

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
Let us build random forests with 120 trees each. doParallel helps to do parallel computing instead of running simple loops:


```r
depend   <- training[-NCOL(training)]
independ <- training$classe

registerDoParallel()
myRandomForest <- foreach(noTrees=rep(4, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
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
##          A 1392    5    0    0    1
##          B    1  940    8    0    0
##          C    0    4  845    6    0
##          D    2    0    2  798    0
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9905   0.9883   0.9925   0.9989
## Specificity            0.9983   0.9977   0.9975   0.9990   1.0000
## Pos Pred Value         0.9957   0.9905   0.9883   0.9950   1.0000
## Neg Pred Value         0.9991   0.9977   0.9975   0.9985   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2838   0.1917   0.1723   0.1627   0.1835
## Detection Prevalence   0.2851   0.1935   0.1743   0.1635   0.1835
## Balanced Accuracy      0.9981   0.9941   0.9929   0.9958   0.9994
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
