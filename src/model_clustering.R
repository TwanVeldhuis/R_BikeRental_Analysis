# step 0 - Get all Packages and the data ####
library(rstudioapi) # step 0
library(lubridate)  # step 0
library(corrplot)   # step 0
library(caret)      # step 0
library(ggplot2)    # step 1
library(cluster)    # step 3
library(dplyr)      # step 5

# Set current script file location
file.path <- getSourceEditorContext()$path 
dir.path <- dirname(normalizePath(file.path))
setwd(dir.path)

# import the data
data.train <- read.csv("../data/processed/train.csv")
data.test <- read.csv("../data/processed/test.csv")

# Combine the two dataframes vertically
data <- rbind(data.train[, -which(names(data.train) == 'count')], data.test)
data.y <- data.train[, which(names(data.train) == 'count')]

rm(list = setdiff(ls(), c("data", "data.y")))

# step 1 - Principal Components Analysis (PCA) ####
set.seed(1)
# The prcomp function is used to perform PCA
data.pca <- prcomp(data)

# The rotation feature indicates the principal component loading vectors
data.pca$rotation

# Plot the first two principal components
biplot(data.pca, main = "Biplot", xlab = "PC1", ylab = "PC2")

# The variance explained by each principal component is obtained by squaring the standard deviation component
data.pca.var <- data.pca$sdev^2

# The proportion of variance explained (PVE) of each component
data.pca.pve <- data.pca.var/sum(data.pca.var)

# Plotting the PVE of each component
data.pca.pve <- data.frame(Component = seq(1, 13, by=1), PVE = data.pca.pve)
ggplot(data.pca.pve, aes(x = Component, y = PVE)) + geom_point() + geom_line()

# conclusion here is that the dataframe can be reduced to 4 columns

data.reduced <- as.data.frame(data.pca$x[, 1:3])
plot(data.reduced[, 1], data.reduced[, 2])

data <- data.reduced
rm(list = setdiff(ls(), c("data", "data.y")))

# step 2 - Elbow method cluster selection ####
set.seed(1)

# Create initial vector and number of clusters that will be tested
cluster.count <- 15
elbow.points <- vector("numeric", length = cluster.count)

# Calculate WCSS for different k values
for (k in 1:cluster.count) {
  kmeans.model <- kmeans(data, centers = k)
  elbow.points[k] <- kmeans.model$tot.withinss
}

# Plot the elbow plot
plot(1:cluster.count, elbow.points, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (k)", ylab = "Within-Cluster Sum of Squares (WCSS)",
     main = "Elbow Plot")

# conclusion here is, 3 or 4 clusters the optimal number
rm(list = setdiff(ls(), c("data", "data.y")))

# step 3 - Average Silhouette method cluster selection ####
set.seed(1)

# Create a vector to store the average silhouette widths for different k values
cluster.count <- 15
silhouette.points <- vector("numeric", length = cluster.count)

# Calculate average silhouette widths for different k values
for (k in 2:cluster.count) {
  kmeans.model <- kmeans(data, centers = k)
  dissimilarity <- dist(data)
  clusters <- kmeans.model$cluster
  silhouette.points[k] <- mean(as.data.frame(silhouette(clusters, dissimilarity))$sil_width)
  print(paste("Cluster count:", k))
}

# Plot the average silhouette widths
plot(1:cluster.count, silhouette.points, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (k)", ylab = "Average Silhouette Width",
     main = "Average Silhouette Method")

# conclusion here is that the optimal number of clusters is 3
rm(list = setdiff(ls(), c("data", "data.y")))

# step 4 - K-Means Clustering ####
set.seed(1)

# The nstart argument runs K-Means with multiple initial cluster assignments
# Running with multiple initial cluster assignments is desirable to minimize within-cluster variance
data.train <- read.csv("../data/processed/train.csv")
data.test <- read.csv("../data/processed/test.csv")
data.raw <- rbind(data.train[, -which(names(data.train) == 'count')], data.test)
kmeans.model <- kmeans(data, centers = 3, nstart = 20)
data$cluster <- kmeans.model$cluster
data.raw$cluster <- kmeans.model$cluster

# The tot.withinss component contains the within-cluster variance
kmeans.model$tot.withinss
rm(list = setdiff(ls(), c("data", "data.raw", "data.y")))

# step 5 - K-Means regression ####
set.seed(1)

clusters <- 3
data.train <- data[1:8708, ]
data.test <- data[8709:10886, ]
data.raw.train <- data.raw[1:8708, ]
data.raw.test <- data.raw[8709:10886, ]

data.train$count <- data.y
data.raw.train$count <- data.y

# plot the PC1 against the PC2 value with the clusters colour coded
plot(data.train$PC1, data.train$PC2, col = data.train$cluster, pch = 19, cex = 0.7,
     xlab = "PC1", ylab="PC2")
legend("topright", legend = unique(data.train$cluster), col = unique(data.train$cluster), pch = 19)

# give summaries of all the actual count values from each cluster
summary(data.raw.train$count[data.raw.train$cluster == 1])
summary(data.raw.train$count[data.raw.train$cluster == 2])
summary(data.raw.train$count[data.raw.train$cluster == 3])

# Create an empty dataframe with column names
cluster.stats <- data.frame(cluster_number = numeric(),
                            avg = numeric(),
                            min = numeric(),
                            max = numeric(),
                            sd = numeric(),
                            stringsAsFactors = FALSE)

for(cluster.number in 1:clusters) {
  # get the data from the cluster number
  cluster.data <- data.train[data.train$cluster == cluster.number, ]
  
  # create new row
  new.row <- data.frame(cluster_number = as.numeric(cluster.number),
                        avg = as.numeric(mean(cluster.data$count)),
                        min = as.numeric(min(cluster.data$count)),
                        max = as.numeric(max(cluster.data$count)),
                        sd = as.numeric(sd(cluster.data$count)),
                        stringsAsFactors = FALSE)
  
  # Append the new row to the existing dataframe
  cluster.stats <- rbind(cluster.stats, new.row)
}
cluster.stats

# Merge the cluster.stats dataframe with data.test based on the 'cluster_number' column
data.train.merged <- merge(data.raw.train, 
                          cluster.stats[, c("cluster_number", "avg")], 
                          by.x = "cluster", 
                          by.y = "cluster_number", 
                          all.x = TRUE) %>% rename(prediction = avg)

# Evaluate the model"s performance on the test data
mse <- mean((data.train.merged$count - data.train.merged$prediction)^2)
print(paste("Mean Squared Error (MSE) on test data:", mse))
ggplot(data.train.merged, aes(x = factor(cluster), y = count)) + 
  geom_boxplot() +
  labs(x = "Cluster", y = "Count")

anova_result <- aov(count ~ factor(cluster), data = data.train.merged)
summary(anova_result)