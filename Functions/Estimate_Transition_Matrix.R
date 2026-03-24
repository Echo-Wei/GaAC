# Install package
require(glmnet)
library(urca)
require(magrittr)

library(caret)
library(DirectedClustering)
library(huge)
library(igraph)

# Transitin Matrix Estimate
# Load Data ---------------------------------------------------------
horizon = 2
T = 700; d = 100;
lasso_index = 90
X <- read.csv("path_to/X_700_100.csv", header = FALSE)
X_normalize <- scale(X)

# Transition Matrix Estiamtion ---------------------------------------------------------
# Adaptive Lasso
W = matrix(NA, nrow = d, ncol = d)
X_S = X_normalize[c(1:T-1),]
X_T = X_normalize[c(2:T),]

# Training and testing splitting for ar method
# smp_siz = 0.7; size = floor(T * smp_siz)
size = T - horizon
x.train <- X_normalize[c(1:(size-1)), ]                    # Training and Testing x and y
y.train <- X_normalize[c(2:size), ] 
x.test <- X_normalize[-c(1:(size-1)), ]
y.test <- X_normalize[-c(1:size), ]

lambda_sum = 0
for (i in c(1:d)){
  # print(i)
  dest = y.train[,i]; src = x.train
  
  # # Ridge coef, adaptive LASSO -----------------------------------------------------------------
  ridge1_cv <- cv.glmnet(x = data.matrix(src), y = dest,
                         type.measure = "mse",
                         nfold = 10,
                         alpha = 0)
  best_ridge_coef <- as.numeric(coef(ridge1_cv, s = ridge1_cv$lambda.min))[-1]
  
  # Adaptive Lasso coef
  index = 1
  adaLasso = cv.glmnet(x = as.matrix(src), y = dest,
                       type.measure = "mse",
                       nfold = 10,
                       alpha = 1,
                       penalty.factor = 1 / (abs(best_ridge_coef))^index,
                       keep = TRUE)
  W[i,] = as.numeric(coef(adaLasso, s = adaLasso$lambda.min))[-1]
}


# Export Data
# write.csv(W, "path")

# Threshold selection while clustering coefficient --------------------------------------------------
# Load W matrix
# W <- read.csv("path", header = TRUE)
# W <- W[,-1]

cluster_coef = c()
norm_W = c()
for (f in seq(0.1, 0.9, 0.1)){ # filter parameter
  # Filter estimated transition matrix
  W_keep = as.matrix(W)
  threshold <- apply(W_keep, 1, function (x) max(x) * f)
  filter <- W_keep >= threshold
  W_keep[filter == 0] <- 0; 
  rownames(W_keep) <- colnames(W_keep)
  W_keep = pmax(W_keep, t(W_keep))
  huge.plot(W_keep)
  # cluster_coef = c(cluster_coef, ClustBCG(W_keep, type = "undirected", isolates = "NaN")$GlobalCC)
  # cluster_coef <- c(cluster_coef, ClustBCG(W_keep, type = "undirected")$GlobalCC)
  local_coef <- ClustBCG(W_keep, type = "undirected")$LocalCC
  local_coef[local_coef == 0] <- NaN
  cluster_coef = c(cluster_coef, mean(na.omit(local_coef)))
  norm_W <- c(norm_W, norm(W_keep, type = '2'))
}

plot(seq(0.1, 0.9, 0.1), cluster_coef, type = 'b')
plot(seq(0.1, 0.9, 0.1), norm_W, type = 'b')

# Filtered transition matrix --------------------------------------------------
filter_portion = 0.6 # this threshold is selected from steps above maximizing Clustering Coefficient
W_keep = as.matrix(W)
threshold <- apply(W_keep, 1, function (x) max(x) * filter_portion)
filter <- W_keep >= threshold
W_keep[filter == 0] <- 0
W_keep = pmax(W_keep, t(W_keep))
# Export Filtered Transition Matrix
write.csv(W_keep, "path")

