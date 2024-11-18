require(tidyverse)
require(keras)
require(tensorflow)

setwd('~/Desktop/PSTAT197/module-2-group11/scripts')
source('preprocessing.R')

setwd('~/Desktop/PSTAT197/module-2-group11/data')
load('claims-test.RData')
load('claims-raw.RData')

setwd('~/Desktop/PSTAT197/module-2-group11/results')
bclass_model <- load_model('bclass_model.keras')

clean_test <- claims_test %>%
  parse_data()