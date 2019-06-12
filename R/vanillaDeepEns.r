dyn.load('/usr/local/cuda/lib64/libcublas.so.10.0')
dyn.load('/usr/local/cuda/lib64/libcusolver.so.10.0')
dyn.load('/usr/local/cuda/lib64/libcudart.so.10.0')
dyn.load('/usr/local/cuda/lib64/libcudnn.so.7')

#https://medium.com/@albertoarrigoni/paper-review-code-deep-ensembles-nips-2017-c5859070b8ce
#https://blogs.rstudio.com/tensorflow/posts/2018-11-12-uncertainty_estimates_dropout/
#https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb
#https://keras.rstudio.com/articles/guide_keras.html
library(keras)
use_implementation("tensorflow")
library(tensorflow)
library(ggplot2)

####DATA PREPRATION####
# sample size (training data)
n_train <- 1000
# sample size (validation data)
n_val <- 1000
# prior length-scale
l <- 1e-4
# initial value for weight regularizer 
wd <- l^2/n_train
# initial value for dropout regularizer
dd <- 2/n_train

gen_data_1d <- function(n) {
  sigma1 <- 1
  sigma2 <- 4
  X <- matrix(rnorm(n))
  w <- 2
  b <- 8
  Y <- matrix(ifelse(X>0,X %*% w + b + sigma1 * rnorm(n), X %*% w + b + sigma2 * rnorm(n)))
  list(X, Y)
}

c(X, Y) %<-% gen_data_1d(n_train + n_val)

c(X_train, Y_train) %<-% list(X[1:n_train], Y[1:n_train])
c(X_val, Y_val) %<-% list(X[(n_train + 1):(n_train + n_val)], 
                          Y[(n_train + 1):(n_train + n_val)])
############################################



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
input_dim = 1L
input <- keras::layer_input(shape = input_dim)

predictions <- input %>% 
  layer_dense(units = 10L, activation = 'relu') %>%
  layer_dense(units = 6L, activation = 'relu') %>%
  layer_dense(units = 30L, activation = 'relu') 

c(mu,sigma) %<-% layer_custom(predictions,1,name = 'main_output')

##############################
######Train NN Ensemble#######
##############################
create_trained_network <- function(train_x, train_y){
  input <- keras::layer_input(shape = input_dim)
  
  predictions <- input %>% 
    layer_dense(units = 10L, activation = 'relu') %>%
    layer_dense(units = 6L, activation = 'relu') %>%
    layer_dense(units = 30L, activation = 'relu') 
  
  c(mu,sigma) %<-% layer_custom(predictions,1,name = 'main_output')
  model <- keras_model(inputs = input, outputs = mu)
  
  model %>% keras::compile(
    optimizer = optimizer_adam(),
    loss = custom_loss(sigma)
  )
  
  history <- model %>% fit(
    X_train,
    Y_train,
    epochs = 200,
    batch_size = 50
  )
  
  layer_name = 'main_output'
  get_intermediate = k_function(inputs=list(model$input),
                                outputs=model$get_layer(layer_name)$output)
  
  return(get_intermediate)
  
}
pred_fns <- list()
for (i in seq(10)){
  #print(i)
  pred_fn<-create_trained_network(X_train,Y_train)
  pred_fns<-append(pred_fns,pred_fn)
}

muMatrix <- data.frame()
sigmaMatrix <-data.frame()
for ( i in seq(10)){
  #print(i)
  c(mus,sigmas) %<-% pred_fns[[i]](inputs=list(as.matrix(X_val)))
  muMatrix<-rbind(muMatrix,t(as.data.frame(mus)))
  sigmaMatrix<-rbind(sigmaMatrix,t(as.data.frame(sigmas)))
}

preds<-c()
sigmas<-c()

for (i in seq(length(muMatrix))){
  out_mu <- mean(muMatrix[,i])
  out_sigma <-sqrt(mean(sigmaMatrix[,i]+sqrt(muMatrix[,i])) - sqrt(out_mu) )
  preds<-append(preds,out_mu)
  sigmas<-append(sigmas,out_sigma)
}


##visualize the outcome
df <- data.frame(
  x = X_val,
  y_pred = preds,
  upper = preds + sqrt(sigmas),
  lower = preds - sqrt(sigmas)
)


ggplot(df, aes(x, y_pred)) + 
  geom_point() + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3)
