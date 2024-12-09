#Homework 3

#set seed
set.seed(418518)

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# If pacman is not already installed, then install it
if (!require(pacman)) install.packages("pacman")

# Load packages
pacman::p_load(ISLR2, caret, randomForest, data.table, ggplot2, glmnet)

#setwd
setwd("~/Downloads")

#load data
dt <- read.csv("ECON_418-518_HW3_Data.csv")


#Data Cleaning

#i
dt$fnlwgt <- NULL
dt$occupation <- NULL
dt$relationship <- NULL
dt$capital.gain <- NULL
dt$capital.loss <- NULL
dt$educational.num <- NULL

################################################################################

#ii
  #a
dt$income <- ifelse(dt$income == ">50K", 1, 0)

  #b
dt$race <- ifelse(dt$race == "White", 1, 0)

  #c
dt$gender <- ifelse(dt$gender == "Male",1, 0)

  #d
dt$workclass <- ifelse(dt$workclass == "Private", 1, 0)

  #e
dt$native.country <- ifelse(dt$native.country == "United-States", 1, 0)

  #f
dt$marital.status <- ifelse(dt$marital.status == "Married-civ-spouse", 1, 0)

  #g
dt$education <- ifelse(dt$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

  #h
dt$age_sq <- (dt$age)^2

  #i
dt[, c("age", "age_sq", "hours.per.week")] <- scale(dt[, c("age", "age_sq", 
                                                           "hours.per.week")])

################################################################################

#iii

  #a
iii.a <- sum(dt$income)
print(iii.a)

  #b
iii.b <- sum(dt$workclass) / 48842

  #c
iii.c <- sum(dt$marital.status) / 48842

  #d
iii.d <- (sum(dt$gender) - 1) / 48842

  #e
dt[dt == ""] <- NA
iii.e <- sum(is.na(dt))

  #d
dt$income <- as.factor(dt$income)

################################################################################

#iv

  #a, b, & c
train_size <- floor(0.7 * nrow(dt)) #finding amount for each indices 
test_size <- ceiling(0.3 * nrow(dt))
shuffled_indices <- sample(nrow(dt)) #shuffle orignial data set
train_indices <- shuffled_indices[1:train_size] #split indices
test_indices <- shuffled_indices[(train_size + 1):nrow(dt)]
dt_train <- dt[train_indices, ] #create data sets
dt_test <- dt[test_indices, ]

################################################################################

#v

  #b

# Set up the train control for cross-validation
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the Lasso regression model
reg_lasso <- train(income ~ ., 
                   data = dt_train, 
                   method = "glmnet",  # Specify glmnet for Lasso
                   trControl = train_control,
                   preProcess = c("center", "scale"),  # Standardize predictors
                   tuneGrid = expand.grid(alpha = 1,  # Alpha = 1 for Lasso regression
                                          lambda = seq(10^-2, 10^5, length = 50)))  # Range of lambda values

reg_lasso

  #d

# Extract coefficients from the trained Lasso model
best_lambda <- reg_lasso$bestTune$lambda  # Best lambda selected by train()
lasso_coefficients <- coef(reg_lasso$finalModel, s = best_lambda)
print(lasso_coefficients)


  #e

#lasso 2
reg_lasso2 <- train(income ~ age + education + marital.status + race + 
                     hours.per.week + native.country, 
                   data = dt_train, 
                   method = "glmnet",  # Specify glmnet for Lasso
                   trControl = train_control,
                   preProcess = c("center", "scale"),  # Standardize predictors
                   tuneGrid = expand.grid(alpha = 1,  # Alpha = 1 for Lasso regression
                   lambda = seq(10^-2, 10^5, length = 50)))  # Range of lambda values
reg_lasso2


# Get the best lambda for lasso
best_lambda_lasso2 <- reg_lasso2$bestTune$lambda

# Make predictions using the lasso model on the training data
pred_probs_lasso2 <- predict(reg_lasso2, s = best_lambda_lasso2, type = "prob")

# Extract first column if it's a matrix
pred_probs_lasso2 <- as.numeric(pred_probs_lasso2[, 1])

# Convert probabilities to binary predictions
predictions_lasso2 <- ifelse(pred_probs_lasso2 > 0.5, 1, 0)

# Convert predictions to factor
predictions_lasso2 <- factor(predictions_lasso2, levels = c(1, 2))

# Ensure dt_train$income is also a factor
dt_train$income <- factor(dt_train$income, levels = c(1, 2))

# Compute the confusion matrix
confusion_matrix_lasso2 <- confusionMatrix(data = predictions_lasso2, reference = dt_train$income)

# View the confusion matrix
print(confusion_matrix_lasso2)



#ridge1
reg_ridge <- train(income ~ age + education + marital.status + race + 
                      hours.per.week + native.country, 
                    data = dt_train, 
                    method = "glmnet",  
                    trControl = train_control,
                    preProcess = c("center", "scale"),  # Standardize predictors
                    tuneGrid = expand.grid(alpha = 0,  # Alpha = 0 for Ridge regression
                    lambda = seq(10^-2, 10^5, length = 50)))  # Range of lambda values
reg_ridge

#classification accuracy test
# Get the best lambda for ridge
best_lambda_ridge <- reg_ridge$bestTune$lambda

# Make predictions using the ridge model on the training data
pred_probs_ridge <- predict(reg_ridge, s = best_lambda_ridge, type = "prob")

# Extract first column if it's a matrix
pred_probs_ridge <- as.numeric(pred_probs_ridge[, 1])

# Convert probabilities to binary predictions
predictions_ridge <- ifelse(pred_probs_ridge > 0.5, 2, 1)

# Convert predictions to factor
predictions_ridge <- factor(predictions_ridge, levels = c(1, 2))

# Ensure dt_train$income is also a factor
dt_train$income <- factor(dt_train$income, levels = c(1, 2))

# Compute the confusion matrix
confusion_matrix_ridge <- confusionMatrix(data = predictions_ridge, reference = dt_train$income)

# View the confusion matrix
print(confusion_matrix_ridge)

################################################################################

#vi

  #b

# Define the grid of mtry values 
mtry_grid <- expand.grid(mtry = c(2, 5, 9, ncol(dt_train) - 1))

# Initialize a list to store models
models_rf <- list()

# Create three random forest models with a different number of trees in each forest
for (t in c(100, 200, 300))
{
  # Print current number of trees in the forest
  print(paste0(t, " trees in the forest."))
  
  # Define the model type
  reg_rf <- train(
    income ~ .,        
    data = dt_train,        
    method = "rf",  
    tuneGrid = mtry_grid,  
    trControl = trainControl(method = "cv", number = 5),
    ntree = t
  )
  # Store the model in the list
  models_rf[[paste0("ntree_", t)]] <- reg_rf

  print(models_rf)
}

  #e - Confusion matrix
#Access model with 300 trees
model_rf <- models_rf[["ntree_300"]]

# Make predictions on the testing data
predictions_rf300 <- predict(model_rf, newdata = dt_train)

#Factorize
predictions_rf300 <- factor(predictions_rf300)
dt_train$income <- factor(dt_train$income)

# Create confusion matrix
confusion_matrix_rf300 <- confusionMatrix(predictions_rf300, dt_train$income)

# Print the confusion matrix and other performance metrics
print(confusion_matrix_rf300)

################################################################################

#VII

# Make predictions on the testing data
predictions_rf_test <- predict(model_rf, newdata = dt_test)

#Factorize
predictions_rf_test <- factor(predictions_rf_test)
dt_test$income <- factor(dt_test$income)

# Create confusion matrix
confusion_matrix_rf_test <- confusionMatrix(predictions_rf_test, dt_test$income)

# Print the confusion matrix and other performance metrics
print(confusion_matrix_rf_test)

