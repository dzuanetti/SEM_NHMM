library(glmnet)
library(dplyr)
library(tidyr)
library(lubridate)
library(forecast)

options(digits=4)
options(scipen=999)

mainDir = paste("/Users/daianezuanetti/Library/CloudStorage/Dropbox/artigo_Gustavo/Códigos",sep = "")
setwd(file.path(mainDir))
lag_var<-TRUE

data <- read.csv(file.choose())
data$Data <- dmy(data$Data)


data <- data %>%
  mutate(Week = floor_date(Data, "week"))


dados_semanal <- data %>%
  group_by(Week) %>%
  summarize(
    Chuva = sum(Chuva, na.rm = TRUE),
    Temp_Ins = mean(Temp_Ins, na.rm = TRUE),
    Temp_Max = mean(Temp_Max, na.rm = TRUE),
    Tem_Min = mean(Temp_Min, na.rm = TRUE),
    Umi_Ins = mean(Umi_Ins, na.rm = TRUE),
    Umi_Max = mean(Umi_Max, na.rm = TRUE),
    Umi_Min = mean(Umi_Min, na.rm = TRUE),
    Pto_Orvalho = mean(Pto_Orv_Ins, na.rm = TRUE),
    Pto_Orvalho_Max = mean(Pto_Orv_Max, na.rm = TRUE),
    Pto_Orvalho_Min = mean(Pto_Orv_Min, na.rm = TRUE),
    Pressao_Ins = mean(Press_Ins, na.rm = TRUE),
    Pressao_Max = mean(Pres_Max, na.rm = TRUE),
    Pressao_Min = mean(Press_Min, na.rm = TRUE),
    Vel_Vento = mean(Vel_Vento, na.rm = TRUE),
    Dir_Vento = mean(Dir_Vento, na.rm = TRUE),
    Raj_Vento = mean(Raj_Vento, na.rm = TRUE)
  )

if (lag_var) {
  dados_semanal <- dados_semanal %>%
    arrange(Week) %>%
    mutate(Chuva = lag(Chuva, order_by = Week))
}

dados_semanal <- drop_na(dados_semanal)

Y <- dados_semanal$Chuva
X <- dados_semanal %>% select(-Chuva, -Week)
T <- length(Y)

# Convert tibble to data.frame
X <- as.data.frame(X)
# Convert columns to numeric
X <- X %>%
  mutate(across(everything(), as.numeric))
X <- X %>%
  mutate(across(everything(), ~ (.-mean(.))/sd(.)))
# Convert data.frame to matrix
X <- as.matrix(X)
Const <- rep(1,length(Y))
X <- cbind(Const,X) 

#   SEPARAR BASES EM TREINO< VALIDAÇÃO E TESTE
##############################################

train_size = 0.80
validation_size = 0.15
test_size = 0.05

#Calcula os indices de corte
cutoff_treino = length(Y)*train_size
cutoff_validation = length(Y)*(train_size+validation_size)

#Cria as bases 
Y_training = Y[1:cutoff_treino]
X_training = X[1:cutoff_treino, c(1,3,6,9,12,14,15,16)]

Y_validation = Y[(cutoff_treino+1):cutoff_validation]
X_validation = X[(cutoff_treino+1):cutoff_validation, c(1,3,6,9,12,14,15,16)]

Y_test = Y[(cutoff_validation+1):T]
X_test = X[(cutoff_validation+1):T, c(1,3,6,9,12,14,15,16)]

############################################
# GLMNET

glmnet_mod <- cv.glmnet(X_training, Y_training)
Y_hat_test_glmnet <- predict(glmnet_mod, newx = X_test, s = "lambda.min")
MSPE_Teste_glmnet <- sum((Y_test - Y_hat_test_glmnet)^2)/length(Y_test) 
MSPE_Teste_glmnet
#############################################

############################################
# ARIMA

arima_mod <- try(auto.arima(y=Y_training, xreg = data.matrix(X_training)))
Y_hat_test_arima <- forecast(arima_mod,xreg=data.matrix(X_test))  
Y_hat_test_arima_DF <- Y_hat_test_arima$mean
MSPE_Teste_arima <- sum((Y_test - Y_hat_test_arima_DF)^2)/length(Y_test)
MSPE_Teste_arima

plot(Y_test,type='l')
lines(Y_hat_test_glmnet,col='red')
lines(c(Y_hat_test_arima_DF),col='blue')


