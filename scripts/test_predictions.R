require(tidyverse)
require(keras)
require(tensorflow)

setwd("~/code/pstat197a/module-2-group11/data")
load("claims-test.RData")
load("claims-raw.RData")
load("claims-clean-example.RData")

setwd("~/code/pstat197a/module-2-group11/results")
bclass_model <- load_model("bclass_model.keras")
mclass_model <- load_model("mclass_model.keras")



# preprocessing functions
# function to parse html and clean text
parse_fn <- function(.html) {
  read_html(.html) %>%
    html_elements("p") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(paste(
      c(
        "\n",
        "[[:punct:]]",
        "nbsp",
        "[[:digit:]]",
        "[[:symbol:]]"
      ),
      collapse = "|"
    ), " ") %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df) {
  out <- .df %>%
    filter(str_detect(text_tmp, "<!")) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean)
  return(out)
}

nlp_fn <- function(parse_data.out) {
  out <- parse_data.out %>%
    unnest_tokens(
      output = token,
      input = text_clean,
      token = "words",
      stopwords = str_remove_all(
        stop_words$word,
        "[[:punct:]]"
      )
    ) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    # filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = "n") %>%
    bind_tf_idf(
      term = token.lem,
      document = .id,
      n = n
    ) %>%
    pivot_wider(
      id_cols = c(".id", "bclass"),
      names_from = "token.lem",
      values_from = "tf_idf",
      values_fill = 0
    )
  return(out)
}

# get predictors
claims_dtm <- nlp_fn(claims_clean)

set.seed(641)
partitions <- claims_dtm %>%
  initial_split(prop = 0.8)

train_dtm <- training(partitions)

top_idf_cols <- colSums(train_dtm[, -1:-2], na.rm = TRUE) %>%
  sort(decreasing = TRUE) %>%
  head(1000) %>%
  names()

clean_df <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean) %>%
  mutate(bclass = -1)

# some rows are removed during preprocessing
# set them to N/A
missing_rows <- setdiff(claims_test$.id, clean_df$.id)

clean_dtm <- nlp_fn(clean_df) %>%
  select(any_of(top_idf_cols))

# some predictors are missing
# add them as columns of 0
missing_cols <- setdiff(top_idf_cols, colnames(clean_dtm))
clean_dtm[missing_cols] <- 0

# create final dtm for model input
claims_test_clean <- clean_dtm[, top_idf_cols] %>%
  as.matrix()

########## Binary Prediction
str(claims_test_clean)
str(claims_raw)
# compute predictions
preds <- predict(bclass_model, claims_test_clean) %>%
  as.numeric()

class_labels <- claims_raw %>%
  pull(bclass) %>%
  levels()

pred_classes <- factor(preds > 0.5, labels = class_labels)
conf_matrix <- table(pred_classes, clean_df$bclass)
conf_matrix
########## Binary Prediction on claims_clean data
preds_clean <- predict(bclass_model, claims_clean) %>%
  as.numeric()

pred_classes_clean <- factor(preds_clean > 0.5, labels = class_labels)

conf_matrix_clean <- table(pred_classes_clean, claims_clean$bclass)
conf_matrix_clean
sensitivity(conf_matrix_clean)
sensitivity(conf_matrix)
########## Prediction on claims_clean data
preds <- predict(bclass_model, claims_clean) %>%
  as.numeric()

class_labels <- claims_raw %>%
  pull(bclass) %>%
  levels()

pred_classes <- factor(preds > 0.5, labels = class_labels)

length(pred_classes)
########## Multiclass Prediction
preds_multi <- predict(mclass_model, claims_test_clean) %>%
  as.numeric()

class_labels_multi <- claims_raw %>%
  pull(mclass) %>%
  levels()

# preds_multi_split <- split(preds_multi, cut(seq_along(preds_multi), 5))
pred_classes_multi <- c()
for (i in 1:915) {
  p0 <- preds_multi[i]
  p1 <- preds_multi[915 + i]
  p2 <- preds_multi[2 * 915 + i]
  p3 <- preds_multi[3 * 915 + i]
  p4 <- preds_multi[4 * 915 + i]

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




# save predictions
# label removed rows as N/A
missing_preds <- data.frame(
  .id = missing_rows,
  bclass.pred = "N/A: No relevant content.",
  mclass.pred = "N/A: No relevant content."
)

pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  mutate(mclass.pred = pred_classes_multi) %>%
  select(.id, bclass.pred, mclass.pred) %>%
  rbind(missing_preds)



setwd("~/code/pstat197a/module-2-group11/results")
saveRDS(pred_df, "preds-group11.RData")
