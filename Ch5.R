# Deep learning for computer vision

## Listing 5.1 A small convnet

library(keras)

#Initiate convnet
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation="relu",
                input_shape=c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

#add a classifier on top of the convnet
model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

#train the convnet on the MNIST images
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images/255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images/255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epocks = 5, batch_size = 64
)

#Evaluate model:
results <- model %>% evaluate(test_images, test_labels)


####### 5.2 Dogs vs Cats example

original_dataset_dir <- "~/DeepLearningWithR/data/catsVsDogsData"

base_dir <- "~/DeepLearningWithR/data/cats_and_dogs_small"
dir.create(base_dir)

train_dir <- file.path(base_dir,"train")
dir.create(train_dir)
validation_dir <- file.path(base_dir,"validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_cats_dir <- file.path(train_dir,"cats")
dir.create(train_cats_dir)

train_dogs_dir <- file.path(train_dir,"dogs")
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir,"cats")
dir.create(validation_cats_dir)

validation_dogs_dir <- file.path(validation_dir,"dogs")
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test_dir,"cats")
dir.create(test_cats_dir)

test_dogs_dir <- file.path(test_dir,"dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_cats_dir))

fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir))

fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir))

fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir))

fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir))

#TODO: unzip data in catsVsDogsData folder. Run commands above to move files.
#book page 123