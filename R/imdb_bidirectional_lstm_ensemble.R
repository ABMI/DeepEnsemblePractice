library(keras)
use_implementation("tensorflow")
library(tensorflow)

max_features <- 20000#20000
batch_size <- 32
epochs <- 50
outputFolder <- ""
outcomeWeight = 2
earlyStoppingPatience = c(5)
earlyStoppingMinDelta = c(1e-4)
EnsembleNum = 5

# Cut texts after this number of words (among top max_features most common words)
maxlen <- 80  


###Deep Ensemble###
custom_loss <- function(sigma){
    gaussian_loss <- function(y_true,y_pred){
        tf$reduce_mean(0.5*tf$log(sigma) + 0.5*tf$div(tf$square(y_true - y_pred), sigma)) + 1e-6
    }
    return(gaussian_loss)
}



GaussianLayer <- R6::R6Class("GaussianLayer",
                             inherit = KerasLayer,
                             
                             public = list(
                                 output_dim = NULL,
                                 kernel_1 = NULL,
                                 kernel_2 = NULL,
                                 bias_1 = NULL,
                                 bias_2 = NULL,
                                 
                                 initialize = function(output_dim){
                                     self$output_dim <- output_dim
                                 },
                                 build = function(input_shape){
                                     super$build(input_shape)
                                     
                                     self$kernel_1 = self$add_weight(name = 'kernel_1',
                                                                     shape = list(as.integer(input_shape[[2]]), self$output_dim), #list(30, self$output_dim),#shape = keras::shape(30, self$output_dim),
                                                                     initializer = keras::initializer_glorot_normal(),
                                                                     trainable = TRUE)
                                     self$kernel_2 = self$add_weight(name = 'kernel_2',
                                                                     shape = list(as.integer(input_shape[[2]]), self$output_dim),#list(30, self$output_dim),  #shape = keras::shape(30, self$output_dim),
                                                                     initializer = keras::initializer_glorot_normal(),
                                                                     trainable = TRUE)
                                     self$bias_1 = self$add_weight(name = 'bias_1',
                                                                   shape = list(self$output_dim),  #shape = keras::shape(self$output_dim),
                                                                   initializer = keras::initializer_glorot_normal(),
                                                                   trainable = TRUE)
                                     self$bias_2 = self$add_weight(name = 'bias_2',
                                                                   shape = list(self$output_dim), #shape = keras::shape(self$output_dim),
                                                                   initializer = keras::initializer_glorot_normal(),
                                                                   trainable = TRUE)
                                 },
                                 
                                 call = function(x, mask = NULL){
                                     output_mu = keras::k_dot(x, self$kernel_1) + self$bias_1
                                     output_sig = keras::k_dot(x, self$kernel_2) + self$bias_2
                                     output_sig_pos = keras::k_log(1 + k_exp(output_sig)) + 1e-06
                                     return (list(output_mu, output_sig_pos))
                                 },
                                 
                                 
                                 compute_output_shape = function(input_shape){
                                     return (list (
                                         list(input_shape[[1]], self$output_dim), 
                                         list(input_shape[[1]], self$output_dim) )
                                     )
                                 } 
                             )
)

# define layer wrapper function
layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
    create_layer(GaussianLayer, object, list(
        output_dim = as.integer(output_dim),
        name = name,
        trainable = trainable
    ))
}

create_trained_network <- function(x_train, y_train, input_dim, maxlen= maxlen, epochs, batch_size, outputFolder, 
                                   earlyStoppingPatience, earlyStoppingMinDelta, outcomeWeight= 1){
    
    
    input <- keras::layer_input(shape = c(maxlen))
    predictions <- input %>% 
        keras::layer_embedding(input_dim = input_dim, output_dim = 128) %>%
        keras::layer_lstm(units = 64, dropout = 0.4,
                   kernel_initializer = keras::initializer_orthogonal(),
                   recurrent_dropout = 0.4) %>%  #for better performance in ensemble
        keras::layer_dense(units = 1, 
                           kernel_initializer = keras::initializer_orthogonal(), #for better performance in ensemble
                           activation = 'relu'#'softmax' #'relu' #'sigmoid
                           )
    
    c(mu,sigma) %<-% layer_custom(predictions,1,name = 'main_output')
    model <- keras_model(inputs = input, outputs = mu)
    
    model %>% keras::compile(
        optimizer = optimizer_adam(),
        loss = custom_loss(sigma)
    )
    progbarLogger=keras::callback_progbar_logger(count_mode = "samples",
                                                 stateful_metrics = NULL)
    
    csvLogger=keras::callback_csv_logger(filename=file.path(outputFolder,"DL_logger.csv"), separator = ",", append = TRUE)
    earlyStopping=keras::callback_early_stopping(monitor = "val_loss", patience=earlyStoppingPatience,
                                                 mode="auto",min_delta = earlyStoppingMinDelta)
    terminateOnNan = keras::callback_terminate_on_naan()
    reduceLr=keras::callback_reduce_lr_on_plateau(monitor="val_loss", factor =0.1, 
                                                  patience = 5,mode = "auto", min_delta = 1e-5, cooldown = 0, min_lr = 0)
    ##for class weight
    freq<-table(y_train)
    class_weights=setNames(as.list(c(sum(freq)/freq),2), names(freq))
    sample_weights = rep(1, dim(x_train)[1])
    for (i in seq(length(class_weights))){
        sample_weights[y_train==as.numeric(names(class_weights)[i])]<-class_weights[[i]]
    }
    
    history <- model %>% fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split= 0.2,
        shuffle=TRUE, #for better performance in ensemble
        sample_weights = sample_weights,
        callbacks = list(progbarLogger,csvLogger,earlyStopping,reduceLr,terminateOnNan)
    )
    
    
    
    layer_name = 'main_output'
    get_intermediate = k_function(inputs=list(model$input),
                                  outputs=model$get_layer(layer_name)$output)
    
    return(list(intermediate = get_intermediate, history = history))
    
}


#######################

cat('Loading data...\n')
imdb <- dataset_imdb(num_words = max_features)
x_train <- imdb$train$x
y_train <- imdb$train$y

x_test <- imdb$test$x
y_test <- imdb$test$y

cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

cat('Pad sequences (samples x time)\n')
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
##small sample size
x_train <- x_train[1:8000,]
y_train <- y_train[1:8000]

#one-hot encoding
y_train <- keras::to_categorical(y_train, 2)#[,2] #population$outcomeCount
y_test <- keras::to_categorical(y_test, 2)#[,2] #population$outcomeCount

cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

cat('Build model...\n')

# Try using different optimizers and different optimizer configs

logFileName <- file.path(outputFolder,"log.txt")
logger <- ParallelLogger::createLogger(name = "keras",
                                       threshold = "TRACE",
                                       appenders = list(ParallelLogger::createFileAppender(layout = ParallelLogger::layoutTimestamp,
                                                                                           fileName = logFileName)))
ParallelLogger::registerLogger(logger)
ParallelLogger::logTrace("Executed this line")
ParallelLogger::logDebug("There are ", length(ParallelLogger::getLoggers()), " loggers")
predList <- list()
historyList <- list()
for (i in seq(EnsembleNum)){
    #print(i)
    ParallelLogger::logInfo(sprintf("%d th deep learning", i))
    pred_fn<-create_trained_network(x_train,y_train[,2],
                                    input_dim = max_features, maxlen = maxlen, epochs=epochs, batch_size=batch_size, 
                                    outputFolder=outputFolder,
                                    earlyStoppingPatience=earlyStoppingPatience, 
                                    earlyStoppingMinDelta=earlyStoppingMinDelta, outcomeWeight= 1)
    predList<-append(predList,pred_fn$intermediate)
    historyList <- append(historyList, pred_fn$history)
}
#i=1
for (i in seq(EnsembleNum)){
    if(i==1){
        muMatrix <- data.frame()
        sigmaMatrix <-data.frame()
    }
    c(mu,sigma) %<-% predList[[i]](inputs=list(x_test))
    muMatrix<-rbind(muMatrix,t(as.data.frame(mu)))
    sigmaMatrix<-rbind(sigmaMatrix,t(as.data.frame(sigma)))
}

# preds<-c()
# sigmas<-c()
# for (i in seq(length(muMatrix))){
#     out_mu <- mean(muMatrix[,i])
#     out_sigma <-sqrt(mean(sigmaMatrix[,i]+sqrt(muMatrix[,i])) - sqrt(out_mu) )
#     preds<-append(preds,out_mu)
#     sigmas<-append(sigmas,out_sigma)
# }
# pROC::roc(y_test[,2],preds )

muMean <- apply(muMatrix,2,mean)
muSq <- muMatrix^2
sigmaSq <- sigmaMatrix^2
sigmaMean <- apply(sigmaMatrix,2,mean)
sigmaResult=apply(muSq+sigmaSq,2, mean)- muMean^2

pROC::roc(y_test[,2],muMean )
prediction <- data.frame(value=muMean, outcomeCount=y_test[,2], sigmas=sigmaResult)
saveRDS(prediction,file.path(outputFolder, "prediction.rds"))
save(list=ls(),file=file.path(outputFolder, "results.Rdata"))
##AUROC #0.8649 -> 0.8521 -> 0.8663 ->0.8469 ->0.8553 
#prediction<-readRDS(file.path(outputFolder, "prediction.rds"))
ParallelLogger::unlink(file.path(outputFolder,"log.txt"))


hist(as.numeric(t(muMatrix)))
hist(as.numeric(t(sigmaMatrix)))

muVector<-as.numeric(t(muMatrix))
sigmaVector<-as.numeric(t(sigmaMatrix))

sum(muVector>0.5)
sum(muVector<0.5)

muMean<-as.numeric(muMatrix[1,])
class(muMean)
