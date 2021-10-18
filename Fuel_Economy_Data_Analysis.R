# Importing libraries 
library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue) 
library(h2o)  

#importing dataset 
raw<-ggplot2::mpg;

#We are given the task that using the x variables year, cyl and displ we need to built a model for predicting target - cty

#viewing data and its summary
raw %>% view()

raw %>% skim()    #the data has no missing values

names(raw) <- names(raw) %>% 
  str_replace_all(" ","_") %>% 
  str_replace_all("-","_") %>% 
  str_replace_all("/","_")

# ----------------------------- Data Preprocessing -----------------------------

# 1.Filling  NA's:  as the data is 100% complete as seen from summary, we do not need this step

# raw %>% 
#   inspect_na() %>% 
#   filter(pcnt<30) %>% 
#   pull(col_name) -> variables
# 
# raw <- raw %>% select(variables)

df <- raw %>%   #selecting numerical variables cty ~ year + cyl + displ.
  select_if(is.numeric) %>%
  select(cty,year, cyl, displ)   #target y birinci, xler sonra gelsin 
 

names(df) <- names(df) %>% 
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","") %>% 
  str_replace_all("\\)","") %>% 
  str_replace_all("\\'","")

# ----------------------------- Multicollinearity -----------------------------

#selecting y and x variables  -features 
target <- 'cty'
features <- df %>% select(-cty) %>% names()

#glm structure: y=kx+b formula detinition as cty ~ year + cyl + displ.
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]   #multicollinearity checking 
features <- features[!features %in% coef_na]       #multi coll olmayan featureleri sec, oxsar olan columnlari at

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))   
glm <- glm(f, data = df)

glm %>% summary()

# VIF (Variance Inflation Factor) ----
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5){   #if the score is greater than 1.5 
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()  #-1 baxdigina tezeden baxmasin
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}

glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 

df <- df %>% select(cty,features)  
#we see that displ column dropped from features as it has multicollinearity
#data is ready for scaling and modeling

# Standardize (Normalize) ----
df %>% glimpse()

df[,-1] <- df[,-1] %>% scale() %>% as.data.frame()   #scaling 

# --------------------------------- Modeling ----------------------------------
h2o.init()

h2o_data <- df %>% as.h2o()

# Splitting the data ----
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'cty'
features <- df %>% select(-cty) %>% names()


# Fitting h2o model ----
model <- h2o.glm(  #glm-generalized linear model
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%    #checking p_values 
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))

# the p-value of the year variable is higher than 0.05, the hypothesis testing results will not be correct
#so it will be deleted by applying Stepwise Backward Elimination

# Significance levels of P_value:
# 0     < p_val < 0.001 ***
# 0.001 < p_val < 0.05  **
# 0.05  < p_val < 0.01  *
# 0.01  < p_val < 0.1   .

# Stepwise Backward Elimination ----
#p>0.05 leri atiriq 
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {      #if the p-value is greater, the hypothesis is not correct
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>% #intercepti atiriq, yeni b-ni , x-lere focus oluruq 
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}


#checking that all the remaining
model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) 

# Predicting the Test set results ----
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict

# ----------------------------- Model evaluation -----------------------------
test_set <- test %>% as.data.frame()
residuals = test_set$cty - y_pred$predict  #the difference between actual and predicted 

# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))

# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$cty)

tss = sum((test_set$cty - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size

k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)

# Plotting actual & predicted ----
my_data <- cbind(predicted = y_pred$predict,
                 observed = test_set$cty) %>% 
  as.data.frame()

g <- my_data %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Test: Adjusted R2 = {round(enexpr(Adjusted_R2),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g %>% ggplotly()

# Checking overfitting ----
y_pred_train <- model %>% h2o.predict(newdata = train) %>% as.data.frame()

train_set <- train %>% as.data.frame()
residuals = train_set$cty - y_pred_train$predict

RMSE_train = sqrt(mean(residuals^2))
y_train_mean = mean(train_set$cty)

tss = sum((train_set$cty - y_train_mean)^2)
rss = sum(residuals^2)

R2_train = 1 - (rss/tss); R2_train

n <- train_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2_train = 1-(1-R2_train)*((n-1)/(n-k-1))


# Plotting actual & predicted
my_data_train <- cbind(predicted = y_pred_train$predict,
                       observed = train_set$cty) %>% 
  as.data.frame()

g_train <- my_data_train %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Train: Adjusted R2 = {round(enexpr(Adjusted_R2_train),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g_train %>% ggplotly()

#we observe from evaluation results of prediction for test and train data that there is underfitting


# Compare  as both in one frame 
library(patchwork)
g_train + g

tibble(RMSE_train = round(RMSE_train,1),
       RMSE_test = round(RMSE,1),
       
       Adjusted_R2_train,
       Adjusted_R2_test = Adjusted_R2)

#We see that there is no significat difference between evaluation results of train and  test data, therefore no overfitting problem.
# Moreover, the scores are relatively high, it means model can find main dependence line in the dataset, so no underfitting
