### Chapter 3

#imdb example (binary classification problem to predict sentiment):

library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data,test_labels)) %<-% imdb

#Train and test data are list of word *indices* in the range [1,10000]

#Decode back to English (see page 61):
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decode_review <- sapply(train_data[[1]], function(index){
  word <- if(index >= 3) reverse_word_index[[as.character(index-3)]]
  if(!is.null(word)) word else "?"
})

#Encode the integer vector sequences into binary matrix
vectorize_sequences <- function(sequences, dimension=10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences)){
    results[i,sequences[[i]]] <- 1
  }
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

#Use two intermediate layers with 16 hidden units with relu activation
#Output scalar prediction using sigmoid function.
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation='relu', input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#Define a validation set for training
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val,y_val)
)

history
str(history)
plot(history) #install ggplot2, otherwise it will use base graphics.

#Retrain model
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#choose 4 epochs (because 20 was way too many... overfit)
model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
#Test on test data:
results <- model %>% evaluate(x_test,y_test)

#Note that this was a naive approach that gives acc of ~85%. With 
#  state of the art techniques, should be able to attain ~95% accuracy.

#Now use model to predict:
model %>% predict(x_test[1:10,])


## Section 3.5 Multiclass (single label) classification problem (data: newswires)

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data,train_labels),c(test_data,test_labels)) %<-% reuters
length(train_data)
length(test_data)
train_data[[1]] #See page 71 for how to decode back to text.
train_labels[[1]]

#Encode data uas binary word vectors
#  See vectorize_sequences function defined above.

#Vectorize data:
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

#One hot encode labels:
to_one_hot <- function(labels,dimension = 46){
  results <- matrix(0,nrow = length(labels), ncol = dimension)
  for(i in 1:length(labels)){
    results[i, labels[[i]] + 1] <- 1
  }
  results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

#Note there is a built in way to one-not encode in keras using the to_categorical function.
# Ex: 
# one_hot_train_labels <- to_categorical(train_labels)

#Use more hidden units (64) b/c number of classes is greater (compared to binary classification)
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy", #TODO: read more about this loss
  metrics = c("accuracy")
)

#Define a validation set:
val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[-val_indices,]

#Train using 20 epochs
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val,y_val)
)

#Train using 6 epochs (to avoid overfitting)

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy", #TODO: read more about this loss
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 6,
  batch_size = 512,
  validation_data = list(x_val,y_val)
  
)

results <- model %>% evaluate(x_test, one_hot_test_labels)
results

#Predictions:
predictions <- model %>% predict(x_test)

#Note: If using integer values for classes (instead of one-hot encoding),
#  use sparse categorical cross entropy loss.


## Section 3.6 Regression problem (predicting housing prices)

dataset <- dataset_boston_housing()
c(c(train_data,train_targets),c(test_data,test_targets)) %<-% dataset

mean <- apply(train_data,2,mean)
std <- apply(train_data,2,sd)
train_data <- scale(train_data, center=mean, scale = std)
test_data <- scale(test_data,center=mean, scale=std) #note the train mean and std are used.

#Because so few data samples are avail, use very small network with 2 hidden layers.

build_model <- function(){
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation="relu",input_shape = dim(train_data[[2]])) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)

  model %>% compile(
    optimizer = "rmsprop",
    loss="mse",
    metrics = c("mae")
  )
}

#Use k-fold cross validation

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 100
all_scores <- c()
for(i in 1:k){
  cat("processing fold ", i, "\n")
  
  val_indices <- which(folds==i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose=0)
  
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores  <- c(all_scores, results$mean_absolute_error)
}


num_epochs <- 500
all_mae_histories <- NULL
for(i in 1:k){
  cat("processing fold ", i, "\n")
  
  val_indices <- which(folds==i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(partial_train_data, partial_train_targets,
                           validation_data = list(val_data,val_targets),
                           epochs = num_epochs, batch_size = 1, verbose=0
                           )
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)

}

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories,2,mean)
)

ggplot(average_mae_history, aes(x=epoch, y = validation_mae)) + geom_line()

#smooth:
ggplot(average_mae_history, aes(x=epoch, y = validation_mae)) + geom_smooth()

#Train final model on all training data:
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 50, batch_size = 16, verbose=0)
results <- model %>% evaluate(test_data,test_targets)
results


