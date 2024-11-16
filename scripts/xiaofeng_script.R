require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('h1, h2, h3, h4, h5, h6, p') %>%
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
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
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

load("./data/claims-raw.RData")

claims_clean <- claims_raw %>%
  parse_data()

# PCA
library(caret)
X <- claims_clean %>%
  select(-bclass) %>%
  as.data.frame()

y <- claims_clean$bclass

# Standardize the features (important for PCA)
X_scaled <- scale(X)

# Perform PCA
pca <- prcomp(X, center = TRUE, scale. = TRUE)

# Check the variance explained by each principal component
summary(pca)

# Use the first few principal components that explain most of the variance
# Choose how many components you want to keep, e.g., 5
X_pca <- data.frame(pca$x[, 1:5])

# Train a logistic regression model using the first 5 principal components
model <- glm(y ~ ., data = X_pca, family = binomial)

# Summary of the model
summary(model)

# Make predictions
pred <- predict(model, X_pca, type = "response")

# Convert predictions to binary outcome (if predicted probability > 0.5, classify as 1, else 0)
pred_bin <- ifelse(pred > 0.5, 1, 0)

# Evaluate the model performance (e.g., using confusion matrix)
conf_matrix <- table(Predicted = pred_bin, Actual = y)
conf_matrix