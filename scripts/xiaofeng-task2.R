require(tidyverse)
library(tidymodels)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
library(pROC)

# word-tokenized data (same as question1)
load("./data/claims-raw.RData")

#### without header
parse_fn_wo <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data_wo <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_wo(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data_wo.out){
  out <- parse_data_wo.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}
# will take a couple minutes
claims_clean_wo <- claims_raw %>%
  parse_data_wo()

## PCA
claims_nlp_wo <- nlp_fn(claims_clean_wo)

set.seed(123)
data_split_wo <- initial_split(claims_nlp_wo, prop = 0.7, strata = bclass)
train_data_wo <- training(data_split_wo)
test_data_wo <- testing(data_split_wo)

X_train_tfidf_wo <- train_data_wo %>% 
  select(-c(.id, bclass)) %>% # exclude non-numeric column
  as.matrix() %>% 
  na.omit()
X_test_tfidf_wo <- test_data_wo %>% 
  select(-c(.id, bclass)) %>% 
  as.matrix() %>% 
  na.omit()

common_columns_wo <- intersect(colnames(X_train_tfidf_wo), colnames(X_test_tfidf_wo))
X_train_tfidf_wo <- X_train_tfidf_wo[, common_columns_wo]
X_test_tfidf_wo <- X_test_tfidf_wo[, common_columns_wo]

train_data_wo$bclass <- factor(train_data_wo$bclass, levels = c("N/A: No relevant content.", "Relevant claim content"))
train_data_wo$bclass <- as.numeric(train_data_wo$bclass) - 1  # Convert to 0 and 1

test_data_wo$bclass <- factor(test_data_wo$bclass, levels = c("N/A: No relevant content.", "Relevant claim content"))
test_data_wo$bclass <- as.numeric(test_data_wo$bclass) - 1 

# training data
X_train_scaled_wo <- X_train_tfidf_wo[, apply(X_train_tfidf_wo, 2, var) != 0]

# PCA transformation
pca_wo <- prcomp(X_train_scaled_wo, center = T, scale. = T)

# Calculate cumulative variance and select the number of components
explained_variance_wo <- summary(pca_wo)$importance[2, ]
cumulative_variance_wo <- cumsum(explained_variance_wo)
num_components_wo <- which(cumulative_variance_wo >= 0.90)[1]

X_train_pca_wo <- data.frame(pca_wo$x[, 1:num_components_wo])
X_train_pca_wo$bclass <- train_data_wo$bclass

# will take a couple minutes
model_wo <- glm(bclass ~ ., data = X_train_pca_wo, family = binomial)
summary(model_wo)

# also apply PCA on testing
X_test_pca_wo <- predict(pca_wo, X_test_tfidf_wo)[, 1:num_components_wo]
X_test_pca_wo <- data.frame(X_test_pca_wo)
X_test_pca_wo$bclass <- test_data_wo$bclass

pred_test_wo <- predict(model_wo, X_test_pca_wo, type = "response")
pred_bin_test_wo <- ifelse(pred_test_wo > 0.5, 1, 0)
conf_matrix_test_wo <- table(Predicted = pred_bin_test_wo, Actual = X_test_pca_wo$bclass)
auc_test_wo <- roc(X_test_pca_wo$bclass, pred_test_wo, levels = c(0, 1), direction = "<")$auc

################## bigrams
nlp_fn_bigram <- function(parse_data_out) {
  parse_data_out %>%
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'ngrams', 
                  n = 2, 
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    filter(!str_detect(token, '[[:punct:]]')) %>%
    count(.id, bclass, token, name = 'n') %>%
    bind_tf_idf(term = token, document = .id, n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'), 
                names_from = 'token', 
                values_from = 'tf_idf', 
                values_fill = 0)
}
claims_clean <- claims_raw %>%
  parse_data_wo()
claims_word <- nlp_fn_bigram(claims_clean)

set.seed(123)
data_split_word <- initial_split(claims_word, prop = 0.7, strata = bclass)
train_word <- training(data_split_word)
test_word <- testing(data_split_word)

X_train_tfidf <- train_word %>% 
  select(-c(.id, bclass)) %>% # exclude non-numeric column
  as.matrix()

X_test_tfidf <- test_word %>% 
  select(-c(.id, bclass)) %>% 
  as.matrix()

common_columns <- intersect(colnames(X_train_tfidf), colnames(X_test_tfidf))
X_train_tfidf <- X_train_tfidf[, common_columns]
X_test_tfidf <- X_test_tfidf[, common_columns]

# Remove columns where all rows are zero
X_train_tfidf <- X_train_tfidf[, colSums(X_train_tfidf != 0) > 0]
X_test_tfidf <- X_test_tfidf[, colSums(X_test_tfidf != 0) > 0]

train_word$bclass <- factor(train_word$bclass, levels = c("N/A: No relevant content.", "Relevant claim content"))
train_word$bclass <- as.numeric(train_word$bclass) - 1  # Convert to 0 and 1

test_word$bclass <- factor(test_word$bclass, levels = c("N/A: No relevant content.", "Relevant claim content"))
test_word$bclass <- as.numeric(test_word$bclass) - 1 

set.seed(123)
sample_idx <- sample(nrow(X_train_tfidf), 200)
X_train_sample <- X_train_tfidf[sample_idx, ]
X_train_sample <- X_train_sample[, apply(X_train_sample, 2, var) != 0]
X_test_tfidf <- X_test_tfidf[, apply(X_test_tfidf, 2, var) != 0]

common_cols <- intersect(colnames(X_train_sample), colnames(X_test_tfidf))
X_train_sample <- X_train_sample[, common_cols]
X_test_tfidf <- X_test_tfidf[, common_cols]

pca <- prcomp(X_train_sample, center = T, scale. = T)

# If X_train_pca_wo has extra rows, subset it to match train_word
X_train_pca_wo <- X_train_pca_wo[1:nrow(train_word), ]

# Now predict the log-odds with the correctly aligned data
train_word$log_odds <- predict(model_wo, newdata = X_train_pca_wo, type = "link") # type="link" gives log-odds
test_word$log_odds <- predict(model_wo, newdata = X_test_pca_wo, type = "link")

# Extract the first 10 PCA components for training and testing
pca_train <- data.frame(pca$x[, 1:10])
pca_test <- predict(pca, newdata = X_test_tfidf)[, 1:10]

# Subset train_word to match the size of the sampled data
train_word_sampled <- train_word[sample_idx, ]

# Combine the PCA components with the predicted log-odds for the second model
train_data_combined <- cbind(train_word_sampled %>% select(bclass, log_odds), pca_train)
test_data_combined <- cbind(test_word %>% select(bclass, log_odds), pca_test)

logit_combined <- glm(bclass ~ log_odds + ., data = train_data_combined, family = binomial())

predicted_probs <- predict(logit_combined, test_data_combined, type = "response")

# Evaluate the model performance using ROC curve and AUC
roc_curve <- roc(test_data_combined$bclass, predicted_probs)
auc_value <- auc(roc_curve)
auc_value
