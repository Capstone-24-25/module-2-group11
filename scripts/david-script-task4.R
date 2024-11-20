# packages
library(tidyverse)
library(tidymodels)
library(tidytext)
library(keras3)
library(tensorflow)
library(tokenizers)
library(stopwords)
library(e1071)
library(textstem)

setwd('~/Desktop/PSTAT197/module-2-group11/scripts')
source('preprocessing.R')

setwd('~/Desktop/PSTAT197/module-2-group11/data')
load('claims-test.RData')
load('claims-raw.RData')
load('claims-clean-example.RData')


######### binary classification


# create document term matrix
claims_dtm <- nlp_fn(claims_clean)

# partition
set.seed(641)
partitions <- claims_dtm %>%
  initial_split(prop=0.8)

train_dtm <- training(partitions)
test_dtm <- testing(partitions)

# train_dtm <- training(partitions) %>%
#   unnest_tokens(output = 'token', input = text_clean) %>%
#   anti_join(rename(stop_words, token = word), by = 'token') %>% #filter out stopwords
#   group_by(.id, bclass) %>%
#   count(token) %>%
#   bind_tf_idf(term = token,
#               document = .id,
#               n = n) %>%
#   pivot_wider(id_cols = c(.id, bclass),
#               names_from = token,
#               values_from = tf_idf,
#               values_fill = 0) %>%
#   ungroup()
# 
# train_dtm
# 
# test_dtm <- testing(partitions) %>%
#   unnest_tokens(output = 'token', input = text_clean) %>%
#   anti_join(rename(stop_words, token = word), by = 'token') %>% #filter out stopwords
#   group_by(.id, bclass) %>%
#   count(token) %>%
#   bind_tf_idf(term = token,
#               document = .id,
#               n = n) %>%
#   pivot_wider(id_cols = c(.id, bclass),
#               names_from = token,
#               values_from = tf_idf,
#               values_fill = 0) %>%
#   ungroup()


# choose tokens with highest idf values across all rows
# try top 1000
top_idf_cols <- colSums(train_dtm[, -1:-2], na.rm=TRUE) %>%
  sort(decreasing=TRUE) %>%
  head(1000) %>%
  names()


# create training set
x_train <- train_dtm %>%
  ungroup() %>%
  select(-.id, -bclass) %>%
  select(all_of(top_idf_cols)) %>%
  as.matrix()

y_train <- train_dtm %>%
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1


# create testing set
x_test <- test_dtm %>%
  ungroup() %>%
  select(-.id, -bclass) %>%
  select(all_of(top_idf_cols)) %>%
  as.matrix()

y_test <- test_dtm %>%
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1

# neural network architecture
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  # layer_dense(512, activation = 'relu') %>%
  # layer_dense(256, activation = 'relu') %>%
  layer_dense(128, activation = 'relu') %>%
  # layer_dense(64, activation = 'relu') %>%
  layer_dense(8, activation = 'relu') %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')
summary(model)

# compile model
model %>% compile(loss = 'binary_crossentropy',
                  optimizer = 'adamax',
                  metrics = 'binary_accuracy')

# train model
history <- model %>%
  fit(x = x_train,
      y = y_train,
      validation_split = 0.2,
      epochs = 20)

evaluate(model, x_test, y_test)

# save model
setwd('~/Desktop/PSTAT197/module-2-group11/results')
save_model(model, 'bclass_model.keras')


# validation accuracy gets stuck around 0.8 for traditional CNN

# try RNN
# architecture
model_rnn <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_embedding(input_dim = ncol(x_train), output_dim = 64) %>%
  layer_gru(128, return_sequences = TRUE) %>%
  layer_lstm(8) %>%
  layer_dense(1)
summary(model_rnn)

# compile model
model_rnn %>% compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = 'binary_accuracy')

# train model
history_rnn <- model_rnn %>%
  fit(x = x_train,
      y = y_train,
      validation_split = 0.3,
      epochs = 20)


# try SVM

# linear kernel
xy_train <- cbind(y_train, x_train)

model_svm_linear <- svm(y_train ~ .,
                        data = xy_train,
                        type = 'C-classification',
                        kernel = 'linear')

y_pred_linear <- predict(model_svm_linear, newdata = x_test)
cm_linear <- table(y_test, y_pred_linear)
accuracy_linear <- (cm_linear[1] + cm_linear[4]) / sum(cm_linear)


# polynomial kernel
model_svm_poly <- svm(y_train ~ .,
                        data = xy_train,
                        type = 'C-classification',
                        kernel = 'polynomial')

y_pred_poly <- predict(model_svm_poly, newdata = x_test)
cm_poly <- table(y_test, y_pred_poly)
accuracy_poly <- (cm_poly[1] + cm_poly[4]) / sum(cm_poly)


# sigmoid kernel
model_svm_sigmoid <- svm(y_train ~ .,
                      data = xy_train,
                      type = 'C-classification',
                      kernel = 'sigmoid')

y_pred_sigmoid <- predict(model_svm_sigmoid, newdata = x_test)
cm_sigmoid <- table(y_test, y_pred_sigmoid)
accuracy_sigmoid <- (cm_sigmoid[1] + cm_sigmoid[4]) / sum(cm_sigmoid)


# radial kernel
model_svm_radial <- svm(y_train ~ .,
                 data = xy_train,
                 type = 'C-classification',
                 kernel = 'radial')

y_pred_radial <- predict(model_svm_radial, newdata = x_test)
cm_radial <- table(y_test, y_pred_radial)
accuracy_radial <- (cm_radial[1] + cm_radial[4]) / sum(cm_radial)



########## multi classification

nlp_fn_mclass <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}


claims_dtm_mclass <- nlp_fn_mclass(claims_clean)

partitions_mclass <- claims_dtm_mclass %>%
  initial_split(prop=0.8)

train_dtm_mclass <- training(partitions_mclass)
test_dtm_mclass <- testing(partitions_mclass)

# create training set
x_train_mclass <- train_dtm_mclass %>%
  ungroup() %>%
  select(-.id, -mclass) %>%
  select(all_of(top_idf_cols)) %>%
  as.matrix()

y_train_mclass <- train_dtm_mclass %>%
  pull(mclass) %>%
  factor() %>%
  as.numeric() - 1


# create testing set
x_test_mclass <- test_dtm_mclass %>%
  ungroup() %>%
  select(-.id, -mclass) %>%
  select(all_of(top_idf_cols)) %>%
  as.matrix()

y_test_mclass <- test_dtm_mclass %>%
  pull(mclass) %>%
  factor() %>%
  as.numeric() - 1


# try neural network
model_mclass <- keras_model_sequential(input_shape = ncol(x_train_mclass)) %>%
  # layer_dense(512, activation = 'relu') %>%
  # layer_dense(256, activation = 'relu') %>%
  layer_dense(128, activation = 'relu') %>%
  # layer_dense(64, activation = 'relu') %>%
  layer_dense(16, activation = 'relu') %>%
  layer_dense(5) %>%
  layer_activation(activation = 'softmax')
summary(model_mclass)

# compile model
model_mclass %>% compile(loss = 'crossentropy',
                  optimizer = 'adamax',
                  metrics = 'accuracy')

# train model
history <- model_mclass %>%
  fit(x = x_train_mclass,
      y = y_train_mclass,
      validation_split = 0.2,
      epochs = 50)

evaluate(model_mclass, x_test_mclass, y_test_mclass)

setwd('~/Desktop/PSTAT197/module-2-group11/results')
save_model(model_mclass, 'mclass_model.keras')


# try SVM

# linear kernel
xy_train_mclass <- cbind(y_train_mclass, x_train_mclass)

model_svm_linear_mclass <- svm(y_train_mclass ~ .,
                        data = xy_train_mclass,
                        type = 'C-classification',
                        kernel = 'linear')

y_pred_linear_mclass <- predict(model_svm_linear_mclass, newdata = x_test_mclass)
cm_linear_mclass <- table(y_test_mclass, y_pred_linear_mclass)
accuracy_linear_mclass <- (cm_linear_mclass[1] + cm_linear_mclass[7] +
                             cm_linear_mclass[13] + cm_linear_mclass[19] +
                             cm_linear_mclass[25]) / sum(cm_linear_mclass)


# polynomial kernel
model_svm_poly_mclass <- svm(y_train_mclass ~ .,
                      data = xy_train_mclass,
                      type = 'C-classification',
                      kernel = 'polynomial')

y_pred_poly_mclass <- predict(model_svm_poly_mclass, newdata = x_test_mclass)
cm_poly_mclass <- table(y_test_mclass, y_pred_poly_mclass)
accuracy_poly_mclass <- (cm_poly_mclass[1] + cm_poly_mclass[7] +
                           cm_poly_mclass[13] + cm_poly_mclass[19] +
                           cm_poly_mclass[25]) / sum(cm_poly_mclass)


# sigmoid kernel
model_svm_sigmoid_mclass <- svm(y_train_mclass ~ .,
                         data = xy_train_mclass,
                         type = 'C-classification',
                         kernel = 'sigmoid')

y_pred_sigmoid_mclass <- predict(model_svm_sigmoid_mclass, newdata = x_test_mclass)
cm_sigmoid_mclass <- table(y_test_mclass, y_pred_sigmoid_mclass)
accuracy_sigmoid_mclass <- (cm_sigmoid_mclass[1] + cm_sigmoid_mclass[7] +
                              cm_sigmoid_mclass[13] + cm_sigmoid_mclass[19] +
                              cm_sigmoid_mclass[25]) / sum(cm_sigmoid_mclass)


# radial kernel
model_svm_radial_mclass <- svm(y_train_mclass ~ .,
                        data = xy_train_mclass,
                        type = 'C-classification',
                        kernel = 'radial')

y_pred_radial_mclass <- predict(model_svm_radial_mclass, newdata = x_test_mclass)
cm_radial_mclass <- table(y_test_mclass, y_pred_radial_mclass)
accuracy_radial_mclass <- (cm_radial_mclass[1] + cm_radial_mclass[7] +
                             cm_radial_mclass[13] + cm_radial_mclass[19] +
                             cm_radial_mclass[25]) / sum(cm_radial_mclass)


