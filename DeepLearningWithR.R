library(keras)
install_keras()
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

str(train_images)

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = "softmax")

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

train_images <- array_reshape(train_images, c(60000,28*28))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000,28*28))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network %>% fit(train_images, train_labels, epochs=5, batch_size=128)

(metrics <- network %>% evaluate(test_images, test_labels))

network %>% predict_classes(test_images[1:10,])


######################################################################
# Chapter 3

library(keras)

imdb <- dataset_imdb(num_words = 10000) #ibdb is a list of 2 lists (each of length 2)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb # %<-% is the multi-assignment operator from the zeallot package.
str(train_data[[1]])

#Need to turn list of numbers into tensors. We'll use a binary vector: 1=word appears in review, 0 otherwise
vectorize_sequence <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)){
    results[i,sequences[[i]]] <- 1
  results  
  }
}

x_train <- vectorize_sequence(train_data)
x_test <- vectorize_sequence(test_data)

str(x_train[1,])

#Convert labels from integer to numeric:
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

#Define the model:
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Compile the model:
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#Validation:
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

#Train model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val,y_val)
)

#See results:
str(history)
plot(history) #uses ggplot2 if it's available, otherwise base plotting.
#We can see here that after the 4th epoc, we seem to have overfitting.

#Retrain with only 4 epochs:
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 4,
  batch_size = 512
)
results <- model %>% evaluate(x_test,y_test)
results

#This is a naive approach that obtains ~85% accuracy. With state-of-the art approaches, 95% accuracy should be attainable.

#Use trained network to generate predictions on new data:
model %>% predict(x_test[1:10,])

#TODO: spin up a gpu machine on AWS to run above code. 
 



