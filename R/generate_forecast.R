source("R/tsglm.R")

sample_data <- read.csv('data/sample_data.csv')

cases <- sample_data$Cases
pop <- sample_data$Population
rain <- sample_data$Rainfall
incidence <- (cases / pop) * 1e5

n_train <- 48
n_test <- 12

y_train <- head(cases, n_train)
xreg_lag <- lag_matrix(rain, 1, label = "xreg")
xreg_train <- head(xreg_lag, n_train)
xreg_test <- tail(xreg_lag, n_test)

pop_train <- head(pop, n_train)

model <- tsglm(y = y_train, lags = 1, xreg = xreg_train, offset = log(pop_train))
newxreg <- cbind(xreg_test, rep(log(1e5), n_test))
forecast <- predict.tsglm(model, h = n_test, xreg = newxreg)

cat(paste(forecast, collapse = ","))
cat("\n")
