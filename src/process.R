# step 0 - Get all Packages ####
library(rstudioapi)
library(lubridate)

# Set current script file location
path.file <- getSourceEditorContext()$path
path.dir <- dirname(normalizePath(path.file))
setwd(path.dir)

# step 1 - Pre-process the train data ####
data.train <- read.csv("../data/raw/train.csv")

# create new dummy variables
data.train$weekday <- as.integer(wday(data.train$datetime))
data.train$hour <- as.integer(hour(data.train$datetime))
data.train$day <- as.integer(day(data.train$datetime))
data.train$month <- as.integer(month(data.train$datetime))
data.train$year <- as.integer(year(data.train$datetime))

# scale the numeric columns
data.train$temp <- scale(data.train$temp)
data.train$atemp <- scale(data.train$atemp)
data.train$windspeed <- scale(data.train$atemp)

# select only the usable columns
data.train.clean <- data.train[,c("season","holiday","workingday",
                                  "weather","temp","atemp","humidity", 
                                  "windspeed","weekday","hour",
                                  "day","month", "year","count")]

write.csv(data.train.clean, "../data/processed/train.csv", row.names = FALSE)

# step 2 - Pre-process the test data ####
data.test <- read.csv("../data/raw/test.csv")

# create new dummy variables
data.test$weekday <- as.integer(wday(data.test$datetime))
data.test$hour <- as.integer(hour(data.test$datetime))
data.test$day <- as.integer(day(data.test$datetime))
data.test$month <- as.integer(month(data.test$datetime))
data.test$year <- as.integer(year(data.test$datetime))

# scale the numeric columns
data.test$temp <- scale(data.test$temp)
data.test$atemp <- scale(data.test$atemp)
data.test$windspeed <- scale(data.test$atemp)

# select only the usable columns
data.test.clean <- data.test[,c("season","holiday","workingday",
                                 "weather","temp","atemp","humidity", 
                                 "windspeed","weekday","hour",
                                 "day","month", "year")]

write.csv(data.test.clean, "../data/processed/test.csv", row.names = FALSE)
