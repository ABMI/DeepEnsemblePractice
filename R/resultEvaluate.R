library(dplyr)
sigmaTiles = 10
#prediction <- data.frame(value=muMean, outcomeCount=y_test[,2], sigmas=sigmaResult)
prediction <- prediction %>% dplyr::mutate(quantile = ntile(sigmas,sigmaTiles))
#prediction <- prediction %>% dplyr::mutate(quantile = ntile(entropy,sigmaTiles))

aggregate(prediction$sigmas,by=list(prediction$quantile),mean)


#function for Brier Score
brierScore <- function(prediction){
    
    brier <- sum((prediction$outcomeCount -prediction$value)^2)/nrow(prediction)
    brierMax <- mean(prediction$value)*(1-mean(prediction$value))
    brierScaled <- 1-brier/brierMax
    return(list(brier=brier,brierScaled=brierScaled))
}

#Overall evaluation
rocResult<-pROC::roc(prediction$outcomeCount,prediction$value)
optimalCoords = pROC::coords(roc=rocResult, x = "best", best.method="youden")
if (length(dim(optimalCoords))==0){
    optThreshold<-optimalCoords["threshold"]
    
}else{
    optThreshold<-optimalCoords["threshold","best"]
}



#Add optimal threshold to the prediction
prediction$optimalThreshold <- optThreshold

##evalation after stratification by sigmas
predictionSplit<-split(prediction, prediction$quantile)

aucBySigmaList<-lapply(predictionSplit, FUN = function(x){
    rocResult<-pROC::roc(x$outcomeCount,x$value)
    optimalCoords <-pROC::coords(roc=rocResult, x = x$optimalThreshold[1])
    stratumOptimalCoords<-pROC::coords(roc=rocResult, x = "best", best.method="youden")
    
    positive <- x$value[x$outcomeCount == 1]
    negative <- x$value[x$outcomeCount == 0]
    pr <- PRROC::pr.curve(scores.class0 = positive, scores.class1 = negative)
    
    allValue<-pROC::coords(rocResult, x=x$optimalThreshold[1],ret=c("threshold", "specificity", "sensitivity", "accuracy",
                                                           "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                           "1-sensitivity", "1-npv", "1-ppv"))
    
    resultDf <- data.frame(positiveProp = sum(x$outcomeCount == 1) / length(x$outcomeCount),
                           # precision = TP,
                           # recall = TN,
                           # FP = FP,
                           # FN = FN, 
                           auroc = pROC::auc(rocResult), 
                           sigmaMean = mean(x$sigmas),
                           #aurocCi = pROC::ci.auc(rocResult),
                           auprc = pr$auc.integral,
                           accuracy = as.data.frame(t(allValue))$accuracy,
                           sensitivity = optimalCoords["specificity"],
                           specificity = optimalCoords["sensitivity"],
                           stratumThreshold = stratumOptimalCoords["threshold"],
                           stratumSpecificity = stratumOptimalCoords["specificity"],
                           stratumSensitivity = stratumOptimalCoords["sensitivity"],
                           brier = brierScore(x)$brier)
    return(resultDf)
})

aucBySigma = do.call("rbind",aucBySigmaList) 

plot(aucBySigma$sigmaMean)
plot(aucBySigma$positiveProp)
plot(aucBySigma$auroc)
plot(aucBySigma$auprc)
plot(aucBySigma$accuracy)
plot(aucBySigma$sensitivity)
plot(aucBySigma$specificity)
plot(aucBySigma$stratumThreshold)
plot(aucBySigma$brier)

############################
##evaluate single DLs
i=4
mu<-as.numeric(muMatrix[i,])
sigma <- as.numeric(sigmaMatrix[i,])

prediction <- data.frame(value=mu, outcomeCount=y_test[,2], sigmas=sigma)
pROC::roc(y_test[,2],mu )

prediction <- prediction %>% dplyr::mutate(quantile = ntile(sigmas,sigmaTiles))

#Overall evaluation
rocResult<-pROC::roc(prediction$outcomeCount,prediction$value)
optimalCoords = pROC::coords(roc=rocResult, x = "best", best.method="youden")
if (length(dim(optimalCoords))==0){
    optThreshold<-optimalCoords["threshold"]
    
}else{
    optThreshold<-optimalCoords["threshold","best"]
}

#Add optimal threshold to the prediction
prediction$optimalThreshold <- optThreshold

##evalation after stratification by sigmas
predictionSplit<-split(prediction, prediction$quantile)

aucBySigmaList<-lapply(predictionSplit, FUN = function(x){
    rocResult<-pROC::roc(x$outcomeCount,x$value)
    optimalCoords <-pROC::coords(roc=rocResult, x = x$optimalThreshold[1])
    stratumOptimalCoords<-pROC::coords(roc=rocResult, x = "best", best.method="youden")
    
    positive <- x$value[x$outcomeCount == 1]
    negative <- x$value[x$outcomeCount == 0]
    pr <- PRROC::pr.curve(scores.class0 = positive, scores.class1 = negative)
    
    resultDf <- data.frame(positiveProp = sum(x$outcomeCount == 1) / length(x$outcomeCount),
                           # precision = TP,
                           # recall = TN,
                           # FP = FP,
                           # FN = FN, 
                           auroc = pROC::auc(rocResult), 
                           #aurocCi = pROC::ci.auc(rocResult),
                           auprc = pr$auc.integral,
                           sensitivity = optimalCoords["specificity"],
                           specificity = optimalCoords["sensitivity"],
                           stratumThreshold = stratumOptimalCoords["threshold"],
                           stratumSpecificity = stratumOptimalCoords["specificity"],
                           stratumSensitivity = stratumOptimalCoords["sensitivity"])
    return(resultDf)
})
aucBySigma = do.call("rbind",aucBySigmaList) 

plot(aucBySigma$positiveProp)
plot(aucBySigma$auroc)
plot(aucBySigma$auprc)
plot(aucBySigma$sensitivity)
plot(aucBySigma$specificity)
plot(aucBySigma$stratumThreshold)


