library(keras3)
library(tensorflow)
library(tidyverse)
library(tidymodels)
library(caret)
library(reticulate)
# use_condaenv("r-tensorflow", required = TRUE)
install_tensorflow()
# load raw data
# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')
view(claims_raw)
str(claims_raw)
# preprocess (will take a minute or two)
claims_clean_RData <- claims_raw %>%
  parse_data()

# export
save(claims_clean_RData, file = 'data/claims-clean-example.RData')

# load cleaned data
load('./data/claims-clean-example.RData')

# partition
set.seed(110122)

claims_clean_ <- nlp_fn(claims_clean_RData)

# check the data 
str(claims_clean)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  select(-c(.id, bclass))
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

x_train <- train_text

y_train <- train_labels

test_text <- testing(partitions) %>%
  select(-c(.id, bclass))
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

x_test <- test_text
y_test <- test_labels
str(test_labels)

max_unique_word <- 33063
max_review_len <- 100
str(x_train)  # Check structure of the data
x_train <- as.integer(x_train)
str(x_train)
summary(x_train)  # Summarize to ensure valid integer values

rnn_model <- keras_model_sequential()
rnn_model %>%
  layer_embedding(input_dim = max_unique_word, output_dim = 128) %>% 
  layer_simple_rnn(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Creating our rnn model 
rnn_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam", 
  metrics = c('accuracy')
)


# Variables for our model 
batch_size = 128 
epochs = 5
validation_split = 0.2
# fraud <- claims_data$
str(x_train)
str(y_train)

# applying model 
rnn_history <- rnn_model %>%  fit(
  x_train, y_train, 
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2 
  
)
plot(rnn_history)
# evaluate model 
rnn_model %>% 
  evaluate(x_test, y_test)

