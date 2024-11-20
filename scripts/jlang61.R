# install.packages('keras3')
# install.packages("caret")
# install.packages("tensorflow")
# install.packages("keras")


# install.packages("keras")
# library(keras)
# install_keras(method = "auto")
# library(keras)
# library(tensorflow)
# 
# tensorflow::tf_version()
# keras::k_version()
# library(reticulate)
# 
# library(tidyverse)
# library(tidymodels)
# library(caret)
# library(reticulate)
# library(keras)
# install_keras()


# load raw data
# can comment entire section out if no changes to preprocessing.R
# source('scripts/preprocessing.R')

# # load raw data
# load('data/claims-raw.RData')
# # preprocess (will take a minute or two)
# claims_clean <- claims_raw %>%
#   parse_data()
# str(claims_clean)
# # export
# save(claims_clean, file = 'data/claims-clean-example.RData')

# load cleaned data
load('./data/claims-clean-example.RData')

# setting seed
set.seed(110122)

claims_clean <- nlp_fn(claims_clean)

# splitting data 
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

# setting data 
train_text <- training(partitions) %>%
  select(-c(.id, bclass))
# setting labels 
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# setting training data and labels
x_train <- train_text
y_train <- train_labels

# setting testing data
test_text <- testing(partitions) %>%
  select(-c(.id, bclass))
# setting testing labels
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# setting testing data and labels
x_test <- test_text
y_test <- test_labels

# Extract text data from the document-term matrix
train_text <- training(partitions) %>% 
  unite("text_clean", everything(), sep = " ", remove = FALSE) %>% 
  pull(text_clean)
test_text <- testing(partitions) %>% 
  unite("text_clean", everything(), sep = " ", remove = FALSE) %>% 
  pull(text_clean)

# Tokenize the text data
tokenizer <- text_tokenizer(num_words = 20000)
tokenizer %>% fit_text_tokenizer(train_text)

# Convert the text to sequences of integers
x_train <- texts_to_sequences(tokenizer, train_text)
x_test <- texts_to_sequences(tokenizer, test_text)

# Pad the sequences to ensure they are all the same length
maxlen <- 100
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

# Convert labels to numeric
y_train <- training(partitions) %>% pull(bclass) %>% as.numeric() - 1
y_test <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

# Define the RNN model architecture
rnn_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 20000, output_dim = 128) %>% 
  layer_gru(128, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(64, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(32) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

summary(rnn_model)

# Compile the model with appropriate loss function and optimizer
rnn_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001), 
  metrics = c('accuracy')
)

# Variables for our model 
batch_size = 128 
epochs = 10
validation_split = 0.2

# Train the model
rnn_history <- rnn_model %>% fit(
  x_train, y_train, 
  batch_size = batch_size,
  epochs = epochs,
  validation_split = validation_split
)

# Plot training history
plot(rnn_history)

# Evaluate the model
rnn_model %>% evaluate(x_test, y_test)

