library(keras)

max_features <- 2000 #20000
batch_size <- 32
epochs = 5

# Cut texts after this number of words (among top max_features most common words)
maxlen <- 80  

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
#x_train <- x_train[1:5000,]
#y_train <- y_train[1:5000]

cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

cat('Build model...\n')
model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = max_features, output_dim = 128) %>% 
    layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
)

cat('Train...\n')
model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
)

y_pred<-predict(model, x_test)
pROC::roc(y_test,y_pred )


scores <- model %>% evaluate(
    x_test, y_test,
    batch_size = batch_size
)

cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])
