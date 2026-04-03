library(dplyr)
library(tidyr)
library(lubridate)
#
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
#
dados_semanal <- drop_na(dados_semanal)

Y <- dados_semanal$Chuva
X <- dados_semanal %>% select(-Chuva, -Week)
T <- length(Y)

train_size = 0.80
validation_size = 0.15
test_size = 0.05

cutoff_treino = length(Y)*train_size
cutoff_validation = length(Y)*(train_size+validation_size)

#Cria as bases 
Y_training = Y[1:cutoff_treino]
X_training = X[1:cutoff_treino, ]

Y_validation = Y[(cutoff_treino+1):cutoff_validation]
X_validation = X[(cutoff_treino+1):cutoff_validation, ]

Y_test = Y[(cutoff_validation+1):T]
X_test = X[(cutoff_validation+1):T, ]

pdf("/Users/Daiane/Downloads/whole_serie.pdf",height=3,width=6)
par(mar=c(3.5,3.5,0.5,0.5),mgp=c(2.0,0.5,0)) # aqui vc consegue mexer no tamanho das margens do gráficos (em mar), a margem abaixo, à esquerda, acima e à direita e aumentar o tamanho do gráfico no espaço que vc tem. Por exemplo, se não quiser colocar título na figura porque vc vai colocar no latex, no rodapé da figura, não precisa de uma margem grande acima dele.
plot(Y,type='l',main=" ",xlab="Weeks",ylab="Values of Precipitation")
abline(v=round(cutoff_treino),col = "red", lty = 2)
abline(v=round(cutoff_validation),col = "red", lty = 2)
dev.off() # não pode esquecer desse comando.

library(forecast)
arima_mod <- try(auto.arima(y=Y_training, xreg = data.matrix(X_training)))
