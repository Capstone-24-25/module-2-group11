setwd('~/code/pstat197a/module-2-group11/scripts')
source('preprocessing.R')

setwd('~/code/pstat197a/module-2-group11/data')
load('claims-test.RData')
load('claims-raw.RData')
load('claims-clean-example.RData')

setwd("~/code/pstat197a/module-2-group11/results")
bclass_model <- load_model("bclass_model.keras")
mclass_model <- load_model("mclass_model.keras")

####################################### BINARY CLASSIFICATION

# create document term matrix
claims_dtm <- nlp_fn(claims_clean)

# partition
set.seed(641)
partitions <- claims_dtm %>%
  initial_split(prop=0.8)

train_dtm <- training(partitions)
test_dtm <- testing(partitions)

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

# Setting class label for binary and multi classification
class_labels <- claims_raw %>%
  pull(bclass) %>%
  levels()

class_labels_multi <- claims_raw %>%
  pull(mclass) %>%
  levels()


# BINARY: Generating predictions and accuracy/sensitivity/specificity
preds_binary <- predict(bclass_model, x_test) %>% 
  as.numeric()

actual_classes <- factor(y_test, labels = class_labels)
pred_classes_binary <- factor(preds_binary > 0.5, labels = class_labels)
cm_binary_nn <- table(pred_classes_binary, actual_classes)

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


############################### MULTI CLASS 

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

# MULTICLASS: Generating predictions and accuracy/sensitivity/specificity

preds_multi <- predict(mclass_model, x_test_mclass) %>% 
  as.numeric()

pred_classes_multi <- factor(levels = c(
  "N/A: No relevant content.",
  "Physical Activity",
  "Possible Fatality",
  "Potentially unlawful activity",
  "Other claim content"
))
iterate <- length(preds_multi)/5
for (i in 1:iterate) {
  p0 <- preds_multi[i]
  p1 <- preds_multi[iterate + i]
  p2 <- preds_multi[2 * iterate + i]
  p3 <- preds_multi[3 * iterate + i]
  p4 <- preds_multi[4 * iterate + i]

  pred <- max(p0, p1, p2, p3, p4)
  if (p0 == pred) {
    pred_classes_multi[i] <- "N/A: No relevant content."
  } else if (p1 == pred) {
    pred_classes_multi[i] <- "Physical Activity"
  } else if (p2 == pred) {
    pred_classes_multi[i] <- "Possible Fatality"
  } else if (p3 == pred) {
    pred_classes_multi[i] <- "Potentially unlawful activity"
  } else {
    pred_classes_multi[i] <- "Other claim content"
  }
}

actual_classes_multi <- factor(y_test_mclass, labels = class_labels_multi)
cm_multi_nn <- table(pred_classes_multi, actual_classes_multi)


# linear kernel
xy_train_mclass <- cbind(y_train_mclass, x_train_mclass)

model_svm_linear_mclass <- svm(y_train_mclass ~ .,
                               data = xy_train_mclass,
                               type = 'C-classification',
                               kernel = 'linear')

y_pred_linear_mclass <- predict(model_svm_linear_mclass, newdata = x_test_mclass)
cm_linear_mclass <- table(y_test_mclass, y_pred_linear_mclass)
accuracy_linear_mclass <- accuracy(cm_linear_mclass)


# polynomial kernel
model_svm_poly_mclass <- svm(y_train_mclass ~ .,
                             data = xy_train_mclass,
                             type = 'C-classification',
                             kernel = 'polynomial')

y_pred_poly_mclass <- predict(model_svm_poly_mclass, newdata = x_test_mclass)
cm_poly_mclass <- table(y_test_mclass, y_pred_poly_mclass)
accuracy_poly_mclass <- accuracy(cm_poly_mclass)


# sigmoid kernel
model_svm_sigmoid_mclass <- svm(y_train_mclass ~ .,
                                data = xy_train_mclass,
                                type = 'C-classification',
                                kernel = 'sigmoid')

y_pred_sigmoid_mclass <- predict(model_svm_sigmoid_mclass, newdata = x_test_mclass)
cm_sigmoid_mclass <- table(y_test_mclass, y_pred_sigmoid_mclass)
accuracy_sigmoid_mclass <- accuracy(cm_sigmoid_mclass)


# radial kernel
model_svm_radial_mclass <- svm(y_train_mclass ~ .,
                               data = xy_train_mclass,
                               type = 'C-classification',
                               kernel = 'radial')

y_pred_radial_mclass <- predict(model_svm_radial_mclass, newdata = x_test_mclass)
cm_radial_mclass <- table(y_test_mclass, y_pred_radial_mclass)
accuracy_radial_mclass <- accuracy(cm_radial_mclass)

# Create 'matrices' directory if it doesn't exist
if (!dir.exists("../matrices")) {
  dir.create("../matrices")
}

# Save confusion matrices to the 'matrices' folder
setwd('~/code/pstat197a/module-2-group11/data')
save(cm_binary_nn, file = "../matrices/cm_binary_nn.RData")
save(cm_linear, file = "../matrices/cm_linear.RData")
save(cm_poly, file = "../matrices/cm_poly.RData")
save(cm_sigmoid, file = "../matrices/cm_sigmoid.RData")
save(cm_radial, file = "../matrices/cm_radial.RData")
save(cm_multi_nn, file = "../matrices/cm_multi_nn.RData")
save(cm_linear_mclass, file = "../matrices/cm_linear_mclass.RData")
save(cm_poly_mclass, file = "../matrices/cm_poly_mclass.RData")
save(cm_sigmoid_mclass, file = "../matrices/cm_sigmoid_mclass.RData")
save(cm_radial_mclass, file = "../matrices/cm_radial_mclass.RData")

# Binary Accuracy Values
print("----Binary Neural Network Accuracy Values----")
sensitivity(cm_binary_nn)
specificity(cm_binary_nn)
accuracy(cm_binary_nn)
print("---------------------------------------------")

# Multiclass  Accuracy Values
## SVM
print("----Binary SVM Linear Accuracy Values----")
sensitivity(cm_linear)
specificity(cm_linear)
accuracy(cm_linear)
print("---------------------------------------------")

print("----Binary SVM Polynomial Accuracy Values----")
sensitivity(cm_poly)
specificity(cm_poly)
accuracy(cm_poly)
print("---------------------------------------------")

print("----Binary SVM Sigmoid Accuracy Values----")
sensitivity(cm_sigmoid)
specificity(cm_sigmoid)
accuracy(cm_sigmoid)
print("---------------------------------------------")

print("----Binary SVM Radial Accuracy Values----")
sensitivity(cm_radial)
specificity(cm_radial)
accuracy(cm_radial)
print("---------------------------------------------")
## Neural Network
print("----Multiclass Neural Network Accuracy Values----")
sensitivity(conf_matrix_multi)
specificity(conf_matrix_multi)
accuracy(conf_matrix_multi)
print("---------------------------------------------")

## SVM
print("----Multiclass SVM Linear Accuracy Values----")
sensitivity(cm_linear_mclass)
specificity(cm_linear_mclass)
accuracy(cm_linear_mclass)
print("---------------------------------------------")

print("----Multiclass SVM Polynomial Accuracy Values----")
sensitivity(cm_poly_mclass)
specificity(cm_poly_mclass)
accuracy(cm_poly_mclass)
print("---------------------------------------------")

print("----Multiclass SVM Sigmoid Accuracy Values----")
sensitivity(cm_sigmoid_mclass)
specificity(cm_sigmoid_mclass)
accuracy(cm_sigmoid_mclass)
print("---------------------------------------------")

print("----Multiclass SVM Radial Accuracy Values----")
sensitivity(cm_radial_mclass)
specificity(cm_radial_mclass)
accuracy(cm_radial_mclass)
print("---------------------------------------------")
