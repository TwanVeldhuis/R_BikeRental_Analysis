# step 0 - Get all Packages and the data ####
library(rstudioapi)
library(lubridate)
library(corrplot)
library(car)
library(e1071)
library(randomForest)
library(caret)
library(gbm)

# Set current script file location
file.path <- getSourceEditorContext()$path 
dir.path <- dirname(normalizePath(file.path))
setwd(dir.path)

# function for evaluating the regression models
evaluate_regression_model <- function(y_pred, y_actual, num_predictors) {
  # Mean Squared Error (MSE)
  mse <- mean((y_pred - y_actual)^2)
  
  # Root Mean Squared Error (RMSE)
  rmse <- sqrt(mse)
  
  # Mean Absolute Error (MAE)
  mae <- mean(abs(y_pred - y_actual))
  
  # R-squared (R²) coefficient
  ss_total <- sum((y_actual - mean(y_actual))^2)
  ss_residual <- sum((y_pred - y_actual)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Adjusted R-squared
  n <- length(y_actual)
  p <- num_predictors
  adjusted_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  # Printing the results
  cat("Mean Squared Error (MSE):", mse, "\n")
  cat("Root Mean Squared Error (RMSE):", rmse, "\n")
  cat("Mean Absolute Error (MAE):", mae, "\n")
  cat("R-squared (R²):", r_squared, "\n")
  cat("Adjusted R-squared:", adjusted_r_squared, "\n")
}

# import the data
data <- read.csv("../data/processed/train.csv")
rm(list = setdiff(ls(), c("data", "evaluate_regression_model")))

# step 1 - Descriptive Analysis ####
set.seed(1)

# split the data into train and test
train <- sample(dim(data)[1], dim(data)[1]*0.7)
data.train <- data[train, ]
data.test <- data[-train, ]

# get the correlations
corr = cor(data)
corrplot(corr, order = "AOE", addCoef.col = "black", 
          col = COL2("PiYG"))

# step 2 - Backward Selection ####
set.seed(1)
# split the data into train and test
train <- sample(dim(data)[1], dim(data)[1]*0.7)
data.train <- data[train, ]
data.test <- data[-train, ]

# execute backward selection
# define intercept-only model
summary(intercept_only <- lm(count ~ 1, data = data.train))

# define model with all predictors
summary(all <- lm(count ~ ., data = data.train))

# perform backward stepwise regression
backward <- step(all, direction='backward', scope=formula(all), trace=0)

# view results of backward stepwise regression
backward$anova

# view final model
backward$coefficients

# Extract the list of selected variables from the terms attribute
selected_vars <- attr(backward$terms, "term.labels")
dependent_vars <- "count"
# step 3 - Prepare data ####
set.seed(1)
# change the data type of the column
data$weather <- as.factor(data$weather)
data$workingday <- as.factor(data$workingday)
data$holiday <- as.factor(data$holiday)
data$season <- as.factor(data$season)
data$weekday <- as.factor(data$weekday)
data$hour <- as.factor(data$hour)
data$day <- as.factor(data$day)
data$month <- as.factor(data$month)
data$year <- as.factor(data$year)

rm(list = setdiff(ls(), c("data", "dependent_vars", "selected_vars", "evaluate_regression_model")))

# step 4a - Linear Regression - Without Backward Selection ####
set.seed(1)
# split the data into train and test
train <- sample(dim(data)[1], dim(data)[1]*0.7)
data.train <- data[train, ]
data.test <- data[-train, ]

# fit linear model using all features
train.control <- trainControl(method = "cv", number = 10)
lr.model1 <- train(count ~ ., data = data.train, method = "lm",
                   trControl = train.control)


# Use the linear regression model to make predictions on the test data
lr.pred1 <- predict(lr.model1, data.test)
lr.pred1 <- pmax(lr.pred1, 0)

# give information about the data
summary(lr.model1)

# plot the information against actual value
plot(lr.pred1, data.test$count) + abline(0,1)

# the distribution of model residuals 
hist(residuals(lr.model1), col = "steelblue")

# create fitted value vs residual plot
plot(fitted(lr.model1), residuals(lr.model1)) + abline(h = 0, lty = 2)

# Determine the metrics
evaluate_regression_model(y_pred = lr.pred1, y_actual = data.test$count, num_predictors = 13)

# step 4b - Linear Regression - With Backward Selection ####
# fit linear model using all features
train.control <- trainControl(method = "cv", number = 10)
lr.model2 <- train(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))), 
                   data = data.train, method = "lm",
                   trControl = train.control)


# Use the linear regression model to make predictions on the test data
lr.pred2 <- predict(lr.model2, data.test)
lr.pred2 <- pmax(lr.pred2, 0)

# give information about the data
summary(lr.model2)

# plot the information against actual value
plot(lr.pred2, data.test$count) + abline(0,1)

# the distribution of model residuals 
hist(residuals(lr.model2), col = "steelblue")

# create fitted value vs residual plot
plot(fitted(lr.model2), residuals(lr.model2)) + abline(h = 0, lty = 2)

# Determine the metrics
evaluate_regression_model(y_pred = lr.pred2, y_actual = data.test$count, num_predictors = 7)

# step 4c - Linear Regression - Final ####
# based on the metrics the standard model without backward selection will be used.
# but we need to take over fitting into consideration, so in the end the choice would go out for the model with bs
# even though this is a bad prediction, it is a good insight regarding expectations for the upcoming models

# step 5a - SVM Regression - Without Backward Selection ####
set.seed(1)

# split the data into train and test
train <- sample(dim(data)[1], dim(data)[1]*0.7)
data.train <- data[train, ]
data.test <- data[-train, ]

# create a svm model with all the features
# Create an empty dataframe
results_df <- data.frame(Cost = numeric(),
                         Gamma = numeric(),
                         MSE = numeric())

# Loop over the parameter values and populate the dataframe
for(cost in seq(30, 60, by=3)) {
  for(gamma in seq(0.030, 0.060, by=0.003)) {
    svm.model <- svm(count ~ .,
                     data = data.train, 
                     kernel="radial",
                     cost = cost, gamma = gamma)
    svm.pred <- predict(svm.model, data.test)
    svm.pred <- pmax(svm.pred, 0)
    mse <- mean((svm.pred - data.test$count)^2)
    
    print(paste("Cost:", cost, "; Gamma:", gamma, "; MSE:", mse))
    results_df <- rbind(results_df, data.frame(Cost = cost, Gamma = gamma, MSE = mse))
  }
}
# based on this the best is probably: cost of 33 and gamma of 0.036
svm.model1 <- svm(count ~ ., data = data.train, kernel = "radial", cost = 33, gamma = 0.036, cross = 10)

# Use the linear regression model to make predictions on the test data
svm.pred1 <- predict(svm.model1, data.test)
svm.pred1 <- pmax(svm.pred1, 0)

# give information about the data
summary(svm.model1)

# plot the information against actual value
plot(svm.pred1, data.test$count) + abline(0,1)

# the distribution of model residuals 
hist(residuals(svm.model1), col = "steelblue")

# create fitted value vs residual plot
plot(fitted(svm.model1), residuals(svm.model1)) + abline(h = 0, lty = 2)

# Determine the metrics
evaluate_regression_model(y_pred = svm.pred1, y_actual = data.test$count, num_predictors = 13)

# step 5b - SVM Regression - With Backward Selection ####
# Create an empty dataframe
results_df2 <- data.frame(Cost = numeric(),
                         Gamma = numeric(),
                         MSE = numeric())

# Loop over the parameter values and populate the dataframe
for(cost in seq(20, 50, by=3)) {
  for(gamma in seq(0.020, 0.050, by=0.003)) {
    svm.model <- svm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))),
                     data = data.train, 
                     kernel="radial",
                     cost = cost, gamma = gamma)
    svm.pred <- predict(svm.model, data.test)
    svm.pred <- pmax(svm.pred, 0)
    mse <- mean((svm.pred - data.test$count)^2)
    
    print(paste("Cost:", cost, "; Gamma:", gamma, "; MSE:", mse))
    results_df2 <- rbind(results_df2, data.frame(Cost = cost, Gamma = gamma, MSE = mse))
  }
}

# create a svm model with backward selected features
svm.model2 <- svm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))),  
                  data = data.train, kernel = "radial", 
                  cost = 50, gamma = 0.05, cross = 10)

# Use the linear regression model to make predictions on the test data
svm.pred2 <- predict(svm.model2, data.test)
svm.pred2 <- pmax(svm.pred2, 0)

# give information about the data
summary(svm.model2)

# plot the information against actual value
plot(svm.pred2, data.test$count) + abline(0,1)

# the distribution of model residuals 
hist(residuals(svm.model2), col = "steelblue")

# create fitted value vs residual plot
plot(fitted(svm.model2), residuals(svm.model2)) + abline(h = 0, lty = 2)

# Determine the metrics
evaluate_regression_model(y_pred = svm.pred2, y_actual = data.test$count, num_predictors = 7)

# step 5c - SVM Regression - Final ####
# based on these results the models that was tuned without backward selection performed by far the best
# these will be used in the final prediction of the values

# step 6a - Random Forest Regression - Bagging ####
set.seed(1)

# split the data into train and test
train <- sample(dim(data)[1], dim(data)[1]*0.7)
data.train <- data[train, ]
data.test <- data[-train, ]

# Execute bagging without backward selection executed beforehand
bag.model1 <- randomForest(count ~ ., 
                           data = data.train, mtry = 13, ntree = 200, importance=TRUE)

# Use the bagged tree to make predictions on the test data
bag.pred1 <- predict(bag.model1, data.test)
bag.pred1 <- pmax(bag.pred1, 0)

# Determine the metrics
evaluate_regression_model(y_pred = bag.pred1, y_actual = data.test$count, num_predictors = 13)
plot(bag.model1)

# Execute bagging with backward selection executed beforehand
bag.model2 <- randomForest(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))), 
                          data = data.train, mtry = 7, ntree = 200, importance=TRUE)

# Use the bagged tree to make predictions on the test data
bag.pred2 <- predict(bag.model2, data.test)
bag.pred2 <- pmax(bag.pred2, 0)

# Determine the metrics
evaluate_regression_model(y_pred = bag.pred2, y_actual = data.test$count, num_predictors = 7)
plot(bag.model2)

# step 6b - Random Forest Regression - Random Forest ####
# fit the random forest model without backward selection
for(predictors in seq(1, 12, by = 1)) {
  rf.model <- randomForest(count ~ ., data = data.train, mtry = predictors)
  rf.pred <- predict(rf.model, data.test)
  rf.pred <- pmax(rf.pred, 0)
  mse <- mean((rf.pred - data.test$count)^2)
  
  print(paste("MTRY:", predictors, "; MSE:", mse))
}
# based on this the model predicts that the best mtry is 11

rf.model1 <- randomForest(count ~ ., data = data.train, mtry = 11, ntree = 200)

# find number of trees that produce lowest test MSE
which.min(rf.model1$mse)
rf.model1$mse[which.min(rf.model1$mse)]

#plot the test MSE by number of trees
plot(rf.model1)

# produce variable importance plot
varImpPlot(rf.model1)

# Use the rf model to predict on the test data
rf.pred1 <- predict(rf.model1, data.test)
rf.pred1 <- pmax(rf.pred1, 0)

# Determine the metrics
evaluate_regression_model(y_pred = rf.pred1, y_actual = data.test$count, num_predictors = 11)


# fit the random forest model with backward selection
for(predictors in seq(1, 6, by = 1)) {
  rf.model <- randomForest(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))), 
                           data = data.train, mtry = predictors)
  rf.pred <- predict(rf.model, data.test)
  rf.pred <- pmax(rf.pred, 0)
  mse <- mean((rf.pred - data.test$count)^2)
  
  print(paste("MTRY:", predictors, "; MSE:", mse))
}
# based on this the model predicts that the best mtry is 6

rf.model2 <- randomForest(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))), 
                          data = data.train, mtry = 6, ntree = 200)

# find number of trees that produce lowest test MSE
which.min(rf.model2$mse)
rf.model2$mse[which.min(rf.model2$mse)]

#plot the test MSE by number of trees
plot(rf.model2)

# produce variable importance plot
varImpPlot(rf.model2)

# Use the rf model to predict on the test data
rf.pred2 <- predict(rf.model2, data.test)
rf.pred2 <- pmax(rf.pred2, 0)

# Determine the metrics
evaluate_regression_model(y_pred = rf.pred2, y_actual = data.test$count, num_predictors = 6)

# step 6c - Random Forest Regression - Boosting ####
# Execute boosting without backward selection executed beforehand
for(depth in seq(1, 10, by = 1)) {
  boost.model <- gbm(count ~ ., data = data.train, 
                      distribution="gaussian", n.trees=500, interaction.depth=depth, cv.folds=10)
  boost.pred <- predict(boost.model, data.test)
  boost.pred <- pmax(boost.pred, 0)
  mse <- mean((boost.pred - data.test$count)^2)
  
  print(paste("Depth:", depth, "; MSE:", mse))
}
# based on this the model predicts that the best depth is 6

boost.model1 <- gbm(count ~ ., data = data.train, 
                   distribution="gaussian", n.trees=500, interaction.depth=6, cv.folds=10)

# The summary function shows the relative influence statistics for each variable
summary(boost.model1)

# Use the boosted model to predict on the test data
boost.pred1 <- predict(boost.model1, data.test)
boost.pred1 <- pmax(boost.pred1, 0)

# Determine the metrics
evaluate_regression_model(y_pred = boost.pred1, y_actual = data.test$count, num_predictors = 13)
plot(boost.pred1, data.test$count)
plot(boost.model1$train.error)
# The results of boosting without backward selection are 1792.903 MSE

# Execute boosting with backward selection executed beforehand
for(depth in seq(1, 10, by = 1)) {
  boost.model <- gbm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))),
                     data = data.train, distribution="gaussian", n.trees=1000, interaction.depth=depth, cv.folds=10)
  boost.pred <- predict(boost.model, data.test)
  boost.pred <- pmax(boost.pred, 0)
  mse <- mean((boost.pred - data.test$count)^2)
  
  print(paste("Depth:", depth, "; MSE:", mse))
}
# based on this the model predicts that the best depth is 7, when maximum tree count is under 500

boost.model2 <- gbm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))),
                    data = data.train, distribution="gaussian", n.trees=500, interaction.depth=7, cv.folds=10)

# The summary function shows the relative influence statistics for each variable
summary(boost.model1)

# Use the boosted model to predict on the test data
boost.pred2 <- predict(boost.model2, data.test)
boost.pred2 <- pmax(boost.pred2, 0)

# Determine the metrics
evaluate_regression_model(y_pred = boost.pred2, y_actual = data.test$count, num_predictors = 7)
plot(boost.pred2, data.test$count)

# The results of boosting with backward selection are 2731.134 MSE
# Boosting results in a test MSE that is slightly better than random forest

# step 6d - Random Forest Regression - Final ####
# based on these results the models that were boosted performed by far the best
# these will be used in the final prediction of the values

# step 7 - Predict ####
data <- read.csv("../data/processed/test.csv")

# remove the unneeded variables
rm(list = setdiff(ls(), c("data", "lr.model2", "svm.model1", "boost.model1")))

data.train <- read.csv("../data/processed/train.csv")
data.full <- rbind(data.train[,1:13], data)

# change the data.full type of the column
data.full$weather <- as.factor(data.full$weather)
data.full$workingday <- as.factor(data.full$workingday)
data.full$holiday <- as.factor(data.full$holiday)
data.full$season <- as.factor(data.full$season)
data.full$weekday <- as.factor(data.full$weekday)
data.full$hour <- as.factor(data.full$hour)
data.full$day <- as.factor(data.full$day)
data.full$month <- as.factor(data.full$month)
data.full$year <- as.factor(data.full$year)
data <- data.full[8709:10886,]

rm(list = setdiff(ls(), c("data", "lr.model2", "svm.model1", "boost.model1")))

# start the modeling
lr.pred <- predict(lr.model2, newdata=data)
lr.pred <- pmax(lr.pred, 0)

#svm prediction
svm.pred <- predict(svm.model1, newdata=data)
svm.pred <- pmax(svm.pred, 0)

#rf prediction
rf.pred <- predict(boost.model1, newdata=data)
rf.pred <- pmax(rf.pred, 0)

# write all results to file
all.pred <- data.frame(LinearRegression = lr.pred, SVM = svm.pred, BoostedDecisionTree = rf.pred)
write.csv(all.pred, "../data/final/results_2.csv", row.names = FALSE)
