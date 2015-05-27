

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: lattice
## Loading required package: survival
## Loading required package: splines
## Loading required package: Formula
## Loading required package: ggplot2
## 
## Attaching package: 'Hmisc'
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(caret)
```

```
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

```r
library(foreach)
```

```
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
```

```r
library(doParallel)
```

```
## Loading required package: iterators
## Loading required package: parallel
```

```r
options(warn=-1)
```


Setting the seed:


```r
set.seed(1234)
```

Let us load the data and get rid of errorneous data points:


```r
trainData <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
evalData  <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
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
##          B    0  946   10    0    0
##          C    0    0  844    9    0
##          D    0    0    1  795    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9953        
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9941        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9871   0.9888   1.0000
## Specificity            0.9991   0.9975   0.9978   0.9998   1.0000
## Pos Pred Value         0.9979   0.9895   0.9894   0.9987   1.0000
## Neg Pred Value         1.0000   0.9992   0.9973   0.9978   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1721   0.1621   0.1837
## Detection Prevalence   0.2851   0.1949   0.1739   0.1623   0.1837
## Balanced Accuracy      0.9996   0.9972   0.9925   0.9943   1.0000
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
