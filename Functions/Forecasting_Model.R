# Load Library
require(glmnet)
require(forecast)
require(magrittr)
require(prophet)
require(modeltime.gluonts)
library(tidymodels)
library(tidyverse)
library(timetk)
library(urca)
library(cluster)
library(reshape)


# Changeable Parameters ---------------------------------------------------------
horizon = 2
pdf_name = 'horizon'
Begining_date <- "2020/06/14"
Ending_date <- as.Date(Begining_date) + 200 - horizon
# Ending_date <- "2022/05/13"
transition_path <- "path to transition matrix"
cluster_path <- "path to clustering result"

# Load Data ---------------------------------------------------------
T = 700; d = 100;
X <- read.csv("path to simulated dataset", header = FALSE)
X <- cbind(date = seq(as.Date(Begining_date), by = "day", length.out = T), X)

# Stationary ---------------------------------------------------------
# Check stationary
# index_stationary <- c()
# for (i in c(1:d)){
#   root_test <- X[,i+1] %>% ur.kpss() %>% summary()
#   if (root_test@teststat > root_test@cval[4]){
#     index_stationary <- c(index_stationary, i + 1)
#   }
# }

# 1st Differentiating
# index_stationary_1st <- c()
# X_1st_dif <- X[-1,]
# for (i in c(1:d)){
#   X_1st_dif[,i+1] <- X[,i+1] %>% diff()
#   root_test_1st <- X_1st_dif[,i+1] %>% ur.kpss() %>% summary()
#   if (root_test_1st@teststat > root_test_1st@cval[4]){
#     index_stationary_1st <- c(index_stationary_1st, i + 1)
#   }
# }
# T = T - 1

X_normalize <- cbind.data.frame(date = X[,1], scale(X[,-1]))

# Transition Matrix Estiamtion ---------------------------------------------------------
# Adaptive Lasso
X_S = X_normalize[c(1:T-1),-1]
X_T = X_normalize[c(2:T),-1]

# Training and testing splitting for ar method
# smp_siz = 0.7; size = floor(T * smp_siz)
size = T - horizon
x.train <- X_normalize[c(1:(size-1)), -1]                    # Training and Testing x and y
y.train <- X_normalize[c(2:size), -1] 
x.test <- X_normalize[-c(1:(size-1)), -1]
y.test <- X_normalize[-c(1:size), -1]

# Training and testing splitting for arima, prophet method
df_train <- X_normalize[c(1:size), -1]
df_test <- X_normalize[-c(1:size), -1]

# Forecasting by transition matrix -------------------------------------------------
start.time <- Sys.time()
mse_ar_recursive <- c(); error_ar_recursive <- c()
W_keep <- read.csv(transition_path, header = TRUE)
W_keep <- W_keep[,-1]
x_input <- t(x.test[1,])
for (i in c(1:horizon)){
  # Forecasting on transition matrix
  predict.adaLasso <- as.matrix(W_keep) %*% as.matrix(x_input)
  x_input <- predict.adaLasso[,1]
  # Forecasting error
  mse_ar_recursive <- c(mse_ar_recursive, mean((as.matrix(y.test[i,]) - predict.adaLasso[,1])^2))
  error_ar_recursive <- c(error_ar_recursive, abs(as.matrix(y.test[i,]) - predict.adaLasso[,1]))
  # plot(y.test[,i], predict.adaLasso)
}
end.time <- Sys.time()
ar_recursive_time <- end.time - start.time

# Forecasting by GraphVAR -------------------------------------------------
start.time <- Sys.time()
mse_graph_recursive <- c(); error_graph_recursive <- c()
GraphVAR_model <- graphicalVAR(x.train, gamma = 0, nLambda = 3)
intercepts <- GraphVAR_model$beta[, 1]          # First column: intercepts
coefficients <- GraphVAR_model$beta[, -1]       # Remaining columns: VAR coefficients
x_input <- t(x.test[1,])
for (i in 1:horizon) {
  if (i == 1) {
    # 1-step ahead: use last actual observation
    predict.graph <- coefficients %*% x_input + intercepts
    mse_graph_recursive <- c(mse_graph_recursive, mean(as.matrix(((y.test[i,]) - (predict.graph))^2)))
    error_graph_recursive <- c(error_graph_recursive, abs(as.matrix((y.test[i,]) - predict.graph)))
  } else {
    # Multi-step ahead: use previous forecast
    predict.graph <- coefficients %*% predict.graph + intercepts
    mse_ar_recursive <- c(mse_graph_recursive, mean(as.matrix(((y.test[i,]) - (predict.graph))^2)))
    error_graph_recursive <- c(error_graph_recursive, abs(as.matrix((y.test[i,]) - predict.graph)))
  }
}
end.time <- Sys.time()
GraphVAR_time <- end.time - start.time

# Forecasting by ARIMA -------------------------------------------------
# allowdrift = FALSE, setting
start.time <- Sys.time()
mse_arima <- c(); error_arima <- c()
for (i in c(1:d)){
  forecast_model <- auto.arima(df_train[,i], allowdrift = FALSE)
  # prediction_arima <- forecast(forecast_model, h = 70)
  prediction_arima <- predict(forecast_model, n.ahead = T - size)
  mse_arima <- c(mse_arima, mean((df_test[,i] - prediction_arima$pred)^2))
  error_arima <- c(error_arima, abs(df_test[,i] - prediction_arima$pred))
}
end.time <- Sys.time()
arima_time <- end.time - start.time

# Forecasting by Prophet -------------------------------------------------
start.time <- Sys.time()
mse_prophet <- c(); error_prophet <- c()
for (i in c(1:d)){
  df_tmp <- data.frame(ds = X_normalize$date[1:size], y = df_train[,i])
  prophet_model <- prophet(df_tmp)
  future <- make_future_dataframe(prophet_model, periods = T - size, freq = "day", include_history = FALSE)
  forecast_prophet <- predict(prophet_model, future)
  mse_prophet <- c(mse_prophet, mean((df_test[,i] - forecast_prophet$yhat)^2))
  error_prophet <- c(error_prophet, abs(df_test[,i] - forecast_prophet$yhat))
}
end.time <- Sys.time()
prophet_time <- end.time - start.time

# Forecasting by DeepAR (spectral cluster) -------------------------------------------------
# If DeepAR cannot run successfully, test the version of your glunots, mxnet in the python environment you built
# https://github.com/business-science/modeltime.gluonts
# https://www.jiqizhixin.com/articles/2020-06-11-12

# Forecasting by DeepAR -------------------------------------------------
start.time <- Sys.time()
mse_deepar_whole <- c(); error_deepar_whole <- c()

# generate training data for DeepAR
id <- as.character(c(rep(c(1:d), each = size)))
df_deepar <- data.frame(id = id, date = seq(as.Date(Begining_date), by = "day", length.out = size), value = unlist(df_train))
# generate testing data for DeepAR
id_test <- as.character(c(rep(c(1:d), each = horizon)))
df_deepar_test <- data.frame(id = id_test, date = seq(as.Date(Ending_date), by = "day", length.out = horizon), value = NA)

# DeepAR Modeling
model_fit_deepar <- deep_ar(
  id                    = "id",
  freq                  = "D",
  prediction_length     = horizon,
  lookback_length       = size,
  epochs                = 5
) %>%
  set_engine("gluonts_deepar") %>%
  fit(value ~ ., df_deepar)

# DeepAR Forecasting
modeltime_forecast_tbl <- modeltime_table(
  model_fit_deepar) %>%
  modeltime_forecast(new_data = df_deepar_test, actual_data = df_deepar, keep_data   = TRUE) %>%
  group_by(id)

prediction_deepar_whole <- modeltime_forecast_tbl$.value[(size*d + 1):(T*d)]
mse_deepar_whole <-  (unlist(df_test) - prediction_deepar_whole)^2
error_deepar_whole <- abs(unlist(df_test) - prediction_deepar_whole)
end.time <- Sys.time()
deepar_whole_time <- end.time - start.time

print(c(mean(mse_ar_recursive), mean(mse_arima), mean(mse_graph_recursive), mean(mse_prophet), mean(mse_deepar_whole)))
print(c(mean(error_ar_recursive), mean(error_arima), mean(error_graph_recursive), mean(error_prophet), mean(error_deepar_whole)))
print(c(ar_recursive_time, arima_time, GraphVAR_time, prophet_time, deepar_whole_time))



