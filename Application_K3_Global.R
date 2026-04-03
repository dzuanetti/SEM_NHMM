###########################################################################
#                                                                         #
#            APPLICATION GLOBAL LASSO - 3 ESTADOS OCULTOS                 #
#                                                                         #
###########################################################################

library('label.switching')
library(glmnet) # Regressão Linear Penalizada
library(forecast) # Auto Regressive Integrated Moving Average
library(e1071)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)

options(digits=4)
options(scipen=999)



#####Função para gerar valores uniformes discretos
rDiscreta<-function(p){
  u<-runif(1)
  P<-cumsum(p)
  val<-sum(P<u)+1
  return(val)}
#####


train_size = 0.80
validation_size = 0.15
test_size = 0.05


zero_threshold = 0.05
K=3   #Numero de estados ocultos
D=16   #Quantidade de Covariaveis
tol<-0.0000001 #Nivel de tolerancia que estabelecemos como criterio de parada do EM Est
tolval=NULL
tolval[1]=1
optim_algo = "BFGS" #Algorithm to use in the optimization process
n_max_iter_EM = 25
Tempo <- NULL
lag_var = TRUE

mainDir = paste("/home/gustavo/Documents/Academic/Cenarios/Application/Code/K3/Global",sep = "")
subDir = paste("Lagged_",toString(lag_var),"_Resultados_Application_Global_K",toString(K),sep = "")
dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
setwd(file.path(mainDir, subDir))

set.seed(2)
lambdas <- seq(0, 4, by=0.025)

#Metricas de Performance Preditiva 
MSPE_Validação <- NULL
MSPE_Teste <- NULL

# 
#Metricas de Performance de Estimação dos ParÂmetros das VA observáveis
Best_Beta_Arrays <- array(rep(0,K*D*K), dim=c(K,D,K))


## SEÇÃO DE DEFINICAÇÃO DOS PARAMETROS PARA SIMULAÇÃO DE DADOS ##
################################################################
P0=rep(1/K,K) #Inicializamos vetor de probabilidades inciais para o HMM

################################################################
## FIM DA SEÇÃO DE DEFINICAÇÃO DOS PARAMETROS PARA SIMULAÇÃO DE DADOS ##


##########################
# glmnet
MSPE_Teste_glmnet <- NULL
##########################

##########################
# arima 
MSPE_Teste_arima <- NULL
##########################

tempo_inicial<-Sys.time()

#   INICIO DE CAPTURA E TRATAMENTO DE DADOS ##
#########################################

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

ggplot(dados_semanal, aes(x = Chuva)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Weekly Value",
       x = "Value",
       y = "Count") +
  theme_minimal()

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

print("Segmentando Base de Dados em Treino, Validação e Teste...")
#   SEPARAR BASES EM TREINO< VALIDAÇÃO E TESTE
##############################################

#Calcula os indices de corte
cutoff_treino = length(Y)*train_size
cutoff_validation = length(Y)*(train_size+validation_size)

#Cria as bases 
Y_training = Y[1:cutoff_treino]
X_training = X[1:cutoff_treino, ]

Y_validation = Y[(cutoff_treino+1):cutoff_validation]
X_validation = X[(cutoff_treino+1):cutoff_validation, ]

Y_test = Y[(cutoff_validation+1):T]
X_test = X[(cutoff_validation+1):T, ]

##############################################
# FIM DE SEPARAÇÃO DAS BASES EM TREINO, VALIDATION E TESTE


# INICIO DO PROCESSO DE ESTIMAÇÃO
##########################################
# Primeiro geramos uma sequência não observavel de treinamento
P_Treino=rep(1/K,K) #Vetor de probabilidade utilizadas para gerar a sequência de treino
S_treino<-NULL # Inicializamos a sequência oculta de treinamento

init1 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_1
init2 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_2
init3 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_3

lasso_iterator = 1 #Criamos um contador para iterar a traves dos valores de lambda

# Algumas estruturas para almacenar valores gerados pelo LASSO
lasso_RMSE <- NULL
lasso_S <- matrix(nrow = length(lambdas), ncol = length(Y_validation))
lasso_Y <- matrix(nrow = length(lambdas), ncol = length(Y_validation))
lasso_mu_hat_estimates <- matrix(nrow = length(lambdas), ncol = K)
lasso_sigma_hat_estimates <- matrix(nrow = length(lambdas), ncol = K)
lasso_Beta_estimates <- matrix(nrow = length(lambdas), ncol = D*K*(K-1))
lasso_Beta_arrays <- array(rep(0,K*D*K*length(lambdas)), dim=c(K,D,K,length(lambdas)))
lasso_lambdas <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_S_training <- matrix(nrow = length(lambdas), ncol = length(Y_training))
lasso_Y_training <- matrix(nrow = length(lambdas), ncol = length(Y_training))
# INICIO DO LASSO
###############################################
pb <- txtProgressBar(min = 1, max = length(lambdas), style = 3)
print('Executando LASSO...')
for (h in 1:length(lambdas)){
  #setTxtProgressBar(pb, lasso_iterator)
  
  #Estruturas necessarias no processo de estimação
  mu_hat = NULL #Variavel para estimar os mus em cada iteração do EM Estocástico
  sigma_hat = NULL #Variavel para estimar os sigmas em cada iteração do EM Estocástico
  BetaArray = array(0, dim=c(K,D,K)) #Estrutura para guardar as estimativas dos Betas em cada iteração do EM
  
  
  VerProx<-NULL
  VerAct<-NULL
  
  lambda = lambdas[lasso_iterator]
  
  #######   Escrevemos as funções que serão o objetivo da optimização   ######
  # Com o temos um array de Betas, utilizaremos tres funções para achar os valores otimos
  # Uma para a matriz Betas[,,1] uma para a matriz Betas[,,2] e uma para 
  # a matriz Betas[,,3]
  FSM1 <-function(params){#função a maximizar para achar os Betas_1
    resp <- sum(1 - log(1 + exp(Xtemp11%*%params[1:D])+ exp(Xtemp11%*%params[(D+1):(2*D)]))) + sum((Xtemp12%*%params[1:D]) - log( 1 + exp(Xtemp12%*%params[1:D])+ exp(Xtemp12%*%params[(D+1):(2*D)]) )) + sum((Xtemp13%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp13%*%params[1:D])+ exp(Xtemp13%*%params[(D+1):(2*D)]) ))
  }
  
  FSM2 <-function(params){#função a maximizar para achar os Betas_2
    resp <- sum(1 - log(1 + exp(Xtemp21%*%params[1:D])+ exp(Xtemp21%*%params[(D+1):(2*D)]))) + sum((Xtemp22%*%params[1:D]) - log( 1 + exp(Xtemp22%*%params[1:D])+ exp(Xtemp22%*%params[(D+1):(2*D)]) )) + sum((Xtemp23%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp23%*%params[1:D])+ exp(Xtemp23%*%params[(D+1):(2*D)]) ))
  }
  
  FSM3 <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp31%*%params[1:D])+ exp(Xtemp31%*%params[(D+1):(2*D)]))) + sum((Xtemp32%*%params[1:D]) - log( 1 + exp(Xtemp32%*%params[1:D])+ exp(Xtemp32%*%params[(D+1):(2*D)]) )) + sum((Xtemp33%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp33%*%params[1:D])+ exp(Xtemp33%*%params[(D+1):(2*D)]) ))
  }
  
  
  FSM1_B <-function(params){#função a maximizar para achar os Betas_1
    resp <- sum(1 - log(1 + exp(Xtemp11%*%params[1:D])+ exp(Xtemp11%*%params[(D+1):(2*D)]))) + sum((Xtemp12%*%params[1:D]) - log( 1 + exp(Xtemp12%*%params[1:D])+ exp(Xtemp12%*%params[(D+1):(2*D)]) )) + sum((Xtemp13%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp13%*%params[1:D])+ exp(Xtemp13%*%params[(D+1):(2*D)]) )) - lambda*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }
  
  FSM2_B <-function(params){#função a maximizar para achar os Betas_2
    resp <- sum(1 - log(1 + exp(Xtemp21%*%params[1:D])+ exp(Xtemp21%*%params[(D+1):(2*D)]))) + sum((Xtemp22%*%params[1:D]) - log( 1 + exp(Xtemp22%*%params[1:D])+ exp(Xtemp22%*%params[(D+1):(2*D)]) )) + sum((Xtemp23%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp23%*%params[1:D])+ exp(Xtemp23%*%params[(D+1):(2*D)]) )) - lambda*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }
  
  FSM3_B <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp31%*%params[1:D])+ exp(Xtemp31%*%params[(D+1):(2*D)]))) + sum((Xtemp32%*%params[1:D]) - log( 1 + exp(Xtemp32%*%params[1:D])+ exp(Xtemp32%*%params[(D+1):(2*D)]) )) + sum((Xtemp33%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp33%*%params[1:D])+ exp(Xtemp33%*%params[(D+1):(2*D)]) )) - lambda*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }
  
  
  #   Procedimento de Estimação   
  #Geramos uma sequência de treinamento
  for (i in 1:length(Y_training)) {
    S_treino[i] = rDiscreta(P_Treino)
  }
  
  ## Escrevemos a função para recalcular a matriz de transição em cada iteração
  ## do algoritmo EM Estocástico.
  Mat_trans <-function(covar){
    B = matrix(nrow=K, ncol=K)
    for (j in 1:K) {
      for (i in 1:K){
        numerator = exp(covar%*%BetaArray[i,,j])
        denom = 0
        for (l in 1:K){
          denom = denom + exp(covar%*%BetaArray[l,,j])
        }
        B[i,j] = numerator/denom
      }  
    }
    return(B)
  }
  
  val=1
  tolval[1]=1
  #Agora executamos o Algoritmo EM Estocástico
  while ( abs(tolval[val])>tol && val < n_max_iter_EM){
    #print(val)
    #print(tolval[val])
    #VeroSimActual=VeroSimProxima
    #Aqui devemos calcular a diferença entre a L.V. em na iteração atual e na anterior  
    LL_parte1 = 0
    LL_parte2 = 0
    LL_parte3 = 0
    LL_parte4 = 0
    VeroSimActual=0
    
    for (k in 1:K){
      id = S_treino == k
      mu_hat[k] = sum(id*Y_training)/sum(id)
      Y_id_list = split(Y_training,id)
      Y_id = unlist(Y_id_list[2], use.names = FALSE)
      sigma_hat[k] = sqrt((sum((Y_id - mu_hat[k])^2)) / (sum(id) - 1)) #DECIDIR SOBRE O ESTIMADOR DA VARIANCIA (VICIADO OU NṼICIADO)
    }
    
    #print(mu_hat)
    #print(sigma_hat)
    #Calculo da Verosimilhança como valor de tolerança
    LL_parte1 = -.5*length(Y_training)*log(2*pi)
    
    for (i in 1:length(Y_training)) {#Calculo do primeiro segmento da LL
      LL_parte2 = LL_parte2 -.5*log(sigma_hat[S_treino[i]]) 
    }
    for (i in 1:length(Y_training)) {#Calculo do segundo segmento da LL
      LL_parte3 = LL_parte3 -(1/(2*sigma_hat[S_treino[i]]))*((Y_training[i]-mu_hat[S_treino[i]])^2)
    }
    temp=NULL
    for (i in 2:length(Y_training)) {#Calculo do terceiro segmento da LL
      for (g in 1:K) {
        temp[g]<-exp(X[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
      }
      LL_parte4 = LL_parte4 + (X[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
    }
    VeroSimActual <- log(P0[S_treino[1]]) + LL_parte1 + (LL_parte2 + LL_parte3) + LL_parte4 #calculo da LogVerosim
    
    
    #Calculamos a sequência S_treino utilizando os Betas
    #Atualizados na iteração passada e os valores observados Y
    S_treino[1]=which.max(dnorm(Y[1], mu_hat, sigma_hat))
    for (i in 2:length(Y_training)) {
      A_hat_t = Mat_trans(X[i,])
      if (any(is.na(A_hat_t))){
        print("NaN encountered in Transition Matrix Calculation")
        A_hat_t[is.nan(A_hat_t)] = 1 
      }
      prob<-(A_hat_t[S_treino[i], ]*dnorm(Y_training[i], mu_hat, sigma_hat))/sum(A_hat_t[S_treino[i], ]*dnorm(Y_training[i], mu_hat, sigma_hat))
      #S_treino[i]=rDiscreta(prob)
      if (any(is.na(prob))){
        print("NaN encountered in S_treino update")
        S_treino[i]=which.max(A_hat_t[S_treino[i], ])
      } else {
        #S_treino[i]=which.max(prob)  
        S_treino[i]=which.max(prob)  
      }
    }
    
    S_treino[is.na(S_treino)] <- 1
    
    if (length(S_treino[is.na(S_treino)]) > 0){
      print(length(S_treino[is.na(S_treino)]))
    }
    
    
    #Este segmento de codigo testa se aconteceram todas as transições possiveis
    #No caso que elas não tinham acontecido, as que
    #não aconteceram são forçadas a acontecer
    TransCount <- matrix(data = c(rep(0,K^2)), nrow = K, ncol = K)
    for (i in 2:length(S_treino)) {
      for (j in 1:K) {
        for (k in 1:K) {
          if (S_treino[i]==j && S_treino[i-1]==k)
            TransCount[k,j]=TransCount[k,j]+1
        }
      }
    }
    
    for (j in 1:K) {
      for (k in 1:K) {
        if (TransCount[k,j]==0){
          positions = sample(2:length(S_treino), 4)
          for (d in 1:4) {
            S_treino[positions[d]]=j
            S_treino[positions[d]-1]=k
          }
        }
      }
    }
    
    #### Aqui inicia a filtragem dos dados para cada iteração
    Xtemp11<-NULL
    Xtemp12<-NULL
    Xtemp13<-NULL
    Xtemp21<-NULL
    Xtemp22<-NULL
    Xtemp23<-NULL
    Xtemp31<-NULL
    Xtemp32<-NULL
    Xtemp33<-NULL
    
    for (t in 2:length(Y_training)) {
      #filtros indo para o Estado # 1
      if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
        Xtemp11<-rbind(Xtemp11, X[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
        Xtemp21<-rbind(Xtemp21, X[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%3)
        Xtemp31<-rbind(Xtemp31, X[t,])
      
      #Filtros indo para o Estado # 2
      if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
        Xtemp12<-rbind(Xtemp12, X[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
        Xtemp22<-rbind(Xtemp22, X[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%3)
        Xtemp32<-rbind(Xtemp32, X[t,])
      
      #Filtros indo para o Estado # 3
      if(S_treino[t]%in%3 && S_treino[t-1]%in%1)
        Xtemp13<-rbind(Xtemp13, X[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%2)
        Xtemp23<-rbind(Xtemp23, X[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%3)
        Xtemp33<-rbind(Xtemp33, X[t,])
    }
    
    if (is.null(Xtemp11)){
      Xtemp11 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp11[,1] <- 1
      print("Encontrou-se X11 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp21)){
      Xtemp21 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp21[,1] <- 1
      print("Encontrou-se X21 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp31)){
      Xtemp31 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp31[,1] <- 1
      print("Encontrou-se X31 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp12)){
      Xtemp12 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp12[,1] <- 1
      print("Encontrou-se X12 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp22)){
      Xtemp22 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp22[,1] <- 1
      print("Encontrou-se X22 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp32)){
      Xtemp32 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp32[,1] <- 1
      print("Encontrou-se X32 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp13)){
      Xtemp13 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp13[,1] <- 1
      print("Encontrou-se X13 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp23)){
      Xtemp23 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp23[,1] <- 1
      print("Encontrou-se X23 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp33)){
      Xtemp33 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp33[,1] <- 1
      print("Encontrou-se X33 vazio. Gerando 1 valor aleatorio.")
    }
    
    ##O ajuste para estimar os parâmetros de transição é
    ##feito aqui usando a função optim e os valores das
    #covariaveis filtradas
    
    fit1 <- tryCatch( 
      {
        optim(par = init1, fn = FSM1, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
      },
      error = function(e) {
        print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
        optim(par = init1, fn = FSM1, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
      }
    )
    
    fit2 <- tryCatch( 
      {
        optim(par = init2, fn = FSM2, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
      },
      error = function(e) {
        print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
        optim(par = init2, fn = FSM2, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
      }
    )
    
    fit3 <- tryCatch( 
      {
        optim(par = init3, fn = FSM3, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
      },
      error = function(e) {
        print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
        optim(par = init3, fn = FSM3, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
      }
    )
    
    # Aqui atribuimos os valores estimados dos parâmetros de 
    # transição a um array que sera utilizado para recalcular 
    # a sequência S_treino na seguinte iteração do EM Est. 
    # Em outras palavras, aqui acontece a ATUALIZAÇÃO dos parâmetros de transição.
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,1]=0
        } else if (i == 2){
          BetaArray[i,d,1]=fit1$par[d]
        } else if (i == 3){
          BetaArray[i,d,1]=fit1$par[D+d]
        }
        
      }
    }
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,2]=0
        } else if (i == 2){
          BetaArray[i,d,2]=fit2$par[d]
        } else if (i == 3){
          BetaArray[i,d,2]=fit2$par[D+d]
        }
        
      }
    }
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,3]=0
        } else if (i == 2){
          BetaArray[i,d,3]=fit3$par[d]
        } else if (i == 3){
          BetaArray[i,d,3]=fit3$par[D+d]
        }
        
      }
    }
    
    
    LL2_parte1 = 0
    LL2_parte2 = 0
    LL2_parte3 = 0
    LL2_parte4 = 0
    VeroSimProxima=0
    
    #Calculo da Verosimilhança como valor de tolerança
    LL2_parte1 = -.5*length(Y_training)*log(2*pi)
    
    for (i in 1:length(Y_training)) {#Calculo do primeiro segmento da LL
      LL2_parte2 = LL2_parte2 -.5*log(sigma_hat[S_treino[i]]) 
    }
    for (i in 1:length(Y_training)) {#Calculo do segundo segmento da LL
      LL2_parte3 = LL2_parte3 -(1/(2*sigma_hat[S_treino[i]]))*((Y_training[i]-mu_hat[S_treino[i]])^2)
    }
    temp=NULL
    for (i in 2:length(Y_training)) {#Calculo do terceiro segmento da LL
      for (g in 1:K) {
        temp[g]<-exp(X[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
      }
      LL2_parte4 = LL2_parte4 + (X[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
    }
    VeroSimProxima <- log(P0[S_treino[1]]) + LL2_parte1 + (LL2_parte2 + LL2_parte3) + LL2_parte4 #calculo da LogVerosim
    
   
    
    val=val+1
    VerAct[val]<-VeroSimActual
    VerProx[val]<-VeroSimProxima
    tolval[val]<-VeroSimProxima - VeroSimActual
    if(is.nan(VeroSimProxima) | is.nan(VeroSimActual))
      tolval[val] <- 0 
    # print(tolval[val])
    
    message(paste('\r',"Lasso iteration # ",toString(lasso_iterator),"; Valor de Lambda = ",toString(lambdas[lasso_iterator]),"; Mu_hat:",toString(round(mu_hat,3)),". Sigma_hat:",toString(round(sigma_hat,3)),"                  ", collapse = ""), appendLF = FALSE) #Messagem indicando o numero da replica atual
  }#######Fim da primeira rodada do EM Estocastico#######
  
  # #Criar algumas matrizes para fazer calculos e manipular a saida MCMC
  # #nestas matrizes, as estimativas serão reordenadas usando o metodo ECR
  # mat_thetar<-matrix(nrow = 1, ncol = K)
  # reorder_S<-matrix(nrow = 1, ncol = length(Y_training))
  # mat_S<-matrix(nrow = 1, ncol = length(Y_training))
  # mat_S[1,]<-S_treino
  # zpvt_S = S #Como pivot para o metodo ECR usamos o S original
  # perms_S = ecr(zpivot = zpvt_S, z = mat_S, K = 3)# aplicamos o metodo ECR que retornara as permutações das dos estados ocultos que devem ser utilizadas para reordenar a saida do algoritmo bayesiano
  # 
  # # Reordenamos a saido do algoritmo EMEst usando as 
  # # permutações fornecidas pelo ECR para K=3 
  # # só rerotulamos a Sequência S_treino, e reordenamos os Thetas
  # # Os Betas serão estimados usando a sequência S_Treino rerotulada
  # # e os Thetas, na segunda etapa do EMEst
  # 
  # for (i in 1:1) {
  #   for (j in 1:length(Y_training)) {
  #     if(S_treino[j]!=Y_training[j] && ((perms_S$permutations[i,1]==2 && perms_S$permutations[i,2]==3 && perms_S$permutations[i,3]==1) | (perms_S$permutations[i,1]==3 && perms_S$permutations[i,2]==1 && perms_S$permutations[i,3]==2))){
  #       S_treino[j]=perms_S$permutations[i,perms_S$permutations[i,S_treino[j]]]
  #     }
  #     
  #     else {
  #       S_treino[j]=perms_S$permutations[i,perms_S$permutations[i,S[j]]]
  #     }
  #   }
  #   mu_hat<-mu_hat[perms_S$permutations[i,]]
  #   sigma_hat<-sigma_hat[perms_S$permutations[i,]]
  # }
  
  # repetimos o EM Estocastico, porque para K=3
  # Entraremos com a sequência S_treino estimada na rodada 
  # anterior E ja rotulada corretamente usando o ECR Para resolver
  # o problema dos Parametros Fantasmas e a troca de rotulos
  
  VeroSimProxima=1
  VeroSimActual=0
  val=1
  tolval=NULL
  tolval[1]=3
  tol2 = 2
  
  while (tolval[val]>tol2) {
    #print(tolval[val])
    #Aqui devemos calcular a diferença entre a L.V. em na iteração atual e na anterior  
    LL_parte1 = 0
    LL_parte2 = 0
    LL_parte3 = 0
    LL_parte4 = 0
    VeroSimActual=0
    
    LL_parte1 = -.5*T*log(2*pi)
    
    for (i in 1:length(Y_training)) {#Calculo do primeiro segmento da LL
      LL_parte2 = LL_parte1 +.5*log(sigma_hat[S_treino[i]]) 
    }
    for (i in 1:length(Y_training)) {#Calculo do segundo segmento da LL
      LL_parte3 = LL_parte3 +(1/(2*sigma_hat[S_treino[i]]))*((Y_training[i]-mu_hat[S_treino[i]])^2)
    }
    temp=NULL
    for (i in 2:length(Y_training)) {#Calculo do terceiro segmento da LL
      for (g in 1:K) {
        temp[g]<-exp(X[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
      }
      LL_parte4 = LL_parte4 + (X[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
    }
    VeroSimActual <- log(P0[S_treino[1]]) + LL_parte1 - (LL_parte2 + LL_parte3) + LL_parte4 #calculo da LogVerosim
    
    #Este segmento de codigo testa se aconteceram todas as transições possiveis
    #No caso que elas não tinham acontecido, as que
    #não aconteceram são forçadas a acontecer
    TransCount <- matrix(data = c(rep(0,K^2)), nrow = K, ncol = K)
    for (i in 2:length(S_treino)) {
      for (j in 1:K) {
        for (k in 1:K) {
          if (S_treino[i]==j && S_treino[i-1]==k)
            TransCount[k,j]=TransCount[k,j]+1
        }
      }
    }
    
    for (j in 1:K) {
      for (k in 1:K) {
        if (TransCount[k,j]==0){
          positions = sample(2:length(S_treino), 4)
          for (d in 1:4) {
            S_treino[positions[d]]=j
            S_treino[positions[d]-1]=k
          }
        }
      }
    }
    
    #filtragem dos dados
    Xtemp11<-NULL
    Xtemp12<-NULL
    Xtemp13<-NULL
    Xtemp21<-NULL
    Xtemp22<-NULL
    Xtemp23<-NULL
    Xtemp31<-NULL
    Xtemp32<-NULL
    Xtemp33<-NULL
    
    for (t in 2:length(Y_training)) {
      #filtros indo para o Estado # 1
      if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
        Xtemp11<-rbind(Xtemp11, X[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
        Xtemp21<-rbind(Xtemp21, X[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%3)
        Xtemp31<-rbind(Xtemp31, X[t,])
      
      #Filtros indo para o Estado # 2
      if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
        Xtemp12<-rbind(Xtemp12, X[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
        Xtemp22<-rbind(Xtemp22, X[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%3)
        Xtemp32<-rbind(Xtemp32, X[t,])
      
      #Filtros indo para o Estado # 3
      if(S_treino[t]%in%3 && S_treino[t-1]%in%1)
        Xtemp13<-rbind(Xtemp13, X[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%2)
        Xtemp23<-rbind(Xtemp23, X[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%3)
        Xtemp33<-rbind(Xtemp33, X[t,])
    }
    
    if (is.null(Xtemp11)){
      Xtemp11 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp11[,1] <- 1
    }
    if (is.null(Xtemp21)){
      Xtemp21 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp21[,1] <- 1
    }
    if (is.null(Xtemp31)){
      Xtemp31 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp31[,1] <- 1
    }
    if (is.null(Xtemp12)){
      Xtemp12 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp12[,1] <- 1
    }
    if (is.null(Xtemp22)){
      Xtemp22 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp22[,1] <- 1
    }
    if (is.null(Xtemp32)){
      Xtemp32 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp32[,1] <- 1
    }
    if (is.null(Xtemp13)){
      Xtemp13 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp13[,1] <- 1
    }
    if (is.null(Xtemp23)){
      Xtemp23 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp23[,1] <- 1
    }
    if (is.null(Xtemp33)){
      Xtemp33 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp33[,1] <- 1
    }
    
    fit1 <- optim(par = init1, fn = FSM1_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
    fit2 <- optim(par = init2, fn = FSM2_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
    fit3 <- optim(par = init3, fn = FSM3_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,1]=0
        } else if (i == 2){
          BetaArray[i,d,1]=fit1$par[d]
        } else if (i == 3){
          BetaArray[i,d,1]=fit1$par[D+d]
        }
        
      }
    }
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,2]=0
        } else if (i == 2){
          BetaArray[i,d,2]=fit2$par[d]
        } else if (i == 3){
          BetaArray[i,d,2]=fit2$par[D+d]
        }
        
      }
    }
    
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,3]=0
        } else if (i == 2){
          BetaArray[i,d,3]=fit3$par[d]
        } else if (i == 3){
          BetaArray[i,d,3]=fit3$par[D+d]
        }
        
      }
    }
    
    
    LL2_parte1 = 0
    LL2_parte2 = 0
    LL2_parte3 = 0
    LL2_parte4 = 0
    VeroSimProxima=0
    
    #Calculo da Verosimilhança como valor de tolerança
    LL2_parte1 = -.5*length(Y_training)*log(2*pi)
    
    for (i in 1:length(Y_training)) {#Calculo do primeiro segmento da LL
      LL2_parte2 = LL2_parte2 +.5*log(sigma_hat[S_treino[i]]) 
    }
    for (i in 1:length(Y_training)) {#Calculo do segundo segmento da LL
      LL2_parte3 = LL2_parte3 +(1/(2*sigma_hat[S_treino[i]]))*((Y_training[i]-mu_hat[S_treino[i]])^2)
      #print((1/(2*sigma_hat[S_treino[i]]))*((Y_training[i]-mu_hat[S_treino[i]])^2))
    }
    temp=NULL
    for (i in 2:length(Y_training)) {#Calculo do terceiro segmento da LL
      for (g in 1:K) {
        temp[g]<-exp(X[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
      }
      LL2_parte4 = LL2_parte4 + (X[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
    }
    VeroSimProxima <- log(P0[S_treino[1]]) + LL2_parte1 - (LL2_parte2 + LL2_parte3) + LL2_parte4 #calculo da LogVerosim
    
    
    
    val=val+1
    tolval[val]<-VeroSimProxima-VeroSimActual
    if(is.nan(VeroSimProxima) | is.nan(VeroSimActual))
      tolval[val] <- 0 
    # print(tolval[val])
  }###fim da segunda rodada do EM Estocastico###
  
  Y_hat_training <- NULL
  for (n in 2:length(Y_training)){
    prob <- NULL
    for (i in 1:K) prob[i]<-exp(X_training[n,]%*%matrix(BetaArray[i,,S_treino[n-1]],ncol=1))
    prob<-prob/sum(prob)
    Y_hat_training[n] <- sum(prob * mu_hat)
  }
  
  Y_hat_validation = NULL
  S_hat_validation = NULL
  S_hat_validation[1]<-rDiscreta(1/K) #O valor para o primeiro estado oculto
  Y_hat_validation[1]<-rnorm(1,mu_hat[S_hat_validation[1]],sigma_hat[S_hat_validation[1]])# O valor para o primeiro valor observavel
  for (t in 2:length(Y_validation)){
    prob<-NULL
    for (i in 1:K) prob[i]<-exp(X_validation[t,]%*%matrix(BetaArray[i,,S_hat_validation[t-1]],ncol=1))
    prob<-prob/sum(prob)
    S_hat_validation[t]<-which.max(prob)
    Y_hat_validation[t]<-sum(prob * mu_hat)
  }
  
  Beta_Estimates <- NULL
  for (i in 2:K) {
    for (j in 1:K){
      for (d in 1:D){
        Beta_Estimates <- c(Beta_Estimates, BetaArray[i,d,j]) 
      }
    }
  }
  lasso_Y[lasso_iterator,] <- Y_hat_validation
  lasso_S[lasso_iterator,] <- S_hat_validation
  lasso_Beta_estimates[lasso_iterator,] <- Beta_Estimates
  lasso_S_training[lasso_iterator, ] <- S_treino
  lasso_Y_training[lasso_iterator, ] <- Y_hat_training
  lasso_mu_hat_estimates[lasso_iterator,] <- mu_hat
  lasso_sigma_hat_estimates[lasso_iterator,] <- sigma_hat
  lasso_RMSE[lasso_iterator] <- (sum((Y_hat_validation - Y_validation)^2))/length(Y_validation)
  lasso_Beta_arrays[,,,lasso_iterator] <- BetaArray
  lasso_iterator = lasso_iterator+1
} ##################################################
# FIM DO PROCESSO DE ESTIMAÇÃO (LASSO)

# CAPTURA INDICE DO LAMBDA COM MELHORES RESULTADOS
min_index = which.min(lasso_RMSE)

# CAPTURA DE METRICAS PARA CADA REPLICA
##################################################################


# COLETANDO VALORES NO CONJUNTO DE VALIDAÇÃO

# Valor de Lambda optimo
Best_Lambdas <- lambdas[min_index]

# Coletar valores estimados dos parâmetros das VA observaveis
Mu_Hat <- lasso_mu_hat_estimates[min_index,]
Sigma_Hat <- lasso_sigma_hat_estimates[min_index,]
Best_Beta_Estimates <- lasso_Beta_estimates[min_index,]
Best_Beta_Arrays[,,] <- lasso_Beta_arrays[,,,min_index]


S_hat_train <- lasso_S_training[min_index, ]
Y_hat_train <- lasso_Y_training[min_index, ]
# Coletar o valor da melhor sequência S e Y no conjunto de Validação
Best_S <- lasso_S[min_index, ]
Best_Y <- lasso_Y[min_index, ]

#Metricas de Performance Preditiva 
MSPE_Validação <- lasso_RMSE[min_index] #Mean Square Predictive Error para o melhor lambda


##########################################################
#           AVALIAÇÃO NO CONJUNTO DE TESTE
#--------------------------------------------------------#
Y_hat_test <- NULL
S_hat_test <- NULL

S_hat_test[1]<-rDiscreta(1/K) #O valor para o primeiro estado oculto
Y_hat_test[1]<-rnorm(1,mu_hat[S_hat_test[1]],sigma_hat[S_hat_test[1]])# O valor para o primeiro valor observavel
for (t in 2:length(Y_test)){
  prob<-NULL
  for (i in 1:K) prob[i]<-exp(X_test[t,]%*%matrix(Best_Beta_Arrays[i,,S_hat_test[t-1]],ncol=1))
  prob<-prob/sum(prob)
  S_hat_test[t]<-which.max(prob)
  Y_hat_test[t]<-sum(prob * mu_hat)
}

MSPE_Teste <- (sum((Y_hat_test - Y_test)^2))/length(Y_test)
Y_test_DF <- Y_test
Y_hat_test_NHMM_DF <- Y_hat_test
S_hat_test_NHMM_DF<- S_hat_test

tempo_final<-Sys.time()
Tempo <- difftime(tempo_final, tempo_inicial, units = "secs")[[1]]/60

# Train dataset for other models (Concatenation of train and validation)
X_tr <- NULL
Y_tr <- NULL

X_tr <- rbind(X_training, X_validation)
Y_tr <- c(Y_training, Y_validation)

############################################
# GLMNET


glmnet_mod <- cv.glmnet(X_training, Y_training)
Y_hat_test_glmnet <- predict(glmnet_mod, newx = X_test, s = "lambda.min")
MSPE_Teste_glmnet <- sum((Y_test - Y_hat_test_glmnet)^2)/length(Y_test) 
Y_hat_test_glmnet_DF <- Y_hat_test_glmnet
#############################################

############################################
# ARIMA

arima_mod <- try(auto.arima(y=Y_training, xreg = data.matrix(X_training)))
Y_hat_test_arima <- forecast(arima_mod,xreg=data.matrix(X_test))  
Y_hat_test_arima_DF <- Y_hat_test_arima$mean
MSPE_Teste_arima <- sum((Y_test - Y_hat_test_arima_DF)^2)/length(Y_test)



df <- data.frame(x = as.numeric(1:length(Y_test)),
                 Real = as.numeric(Y_test),
                 NHMM_Global_LASSO = as.numeric(Y_hat_test_NHMM_DF),
                 ARIMA = as.numeric(Y_hat_test_arima_DF),
                 GLMNET = as.numeric(Y_hat_test_glmnet_DF)
)
# Convert to long format
df_long <- pivot_longer(df, 
                        cols = c(Real, NHMM_Global_LASSO, ARIMA, GLMNET), 
                        values_to = "Prediction",
                        names_to = "Algorithm")

# Plot
ggplot(df_long,
       aes(
         x = x,
         y = Prediction,
         color = Algorithm,
         linetype = Algorithm
       )) +
  geom_line(size = 0.7) +
  xlab("Index") +
  ylab("Predictions") +
  scale_x_continuous(breaks = seq(0, length(Y_test), by = 5), limits = c(1,length(Y_test))) +
  scale_linetype_manual(values = c(Real = "solid", NHMM_Global_LASSO = "dashed", ARIMA = "dotted", GLMNET = "dotted")) + 
  scale_color_manual(values = c("salmon1", "palegreen3", "steelblue3", "purple", "gray60"))+
  theme_minimal() 



##################################
### GENERATING CI(95%)
##################################
n_samples <- 30
Y_test_CI_matrix <- matrix(nrow = n_samples, ncol = length(Y_test))
S_test_CI_matrix <- matrix(nrow = n_samples, ncol = length(Y_test))
for (t in 1:length(Y_test)){
  if (t == 1){
    Y_test_CI_matrix[,1] <- rnorm(n_samples, Mu_Hat[S_hat_test[1]],Sigma_Hat[S_hat_test[1]])
    S_test_CI_matrix[,1] <- rep(S_hat_test[1], n_samples)
  } else {
    prob<-NULL
    for (i in 1:K) prob[i]<-exp(X_test[t,]%*%matrix(Best_Beta_Arrays[i,,S_hat_test[t-1]],ncol=1))
    prob<-prob/sum(prob)
    for (l in 1:n_samples){
      S_test_CI_matrix[l,t]<-rDiscreta(prob)
      #S_test_CI_matrix[l,t]<-which.max(prob)
      Y_test_CI_matrix[l,t]<-rnorm(1, Mu_Hat[S_test_CI_matrix[l,t]],Sigma_Hat[S_test_CI_matrix[l,t]])  
      #Y_test_CI_matrix[l,t]<-sum(prob * Mu_Hat)
    }
  }
}

Y_test_CI_matrix <- data.frame(Y_test_CI_matrix)

############################################
# Assuming Y_test_CI_matrix is your dataframe with 44 columns and 100 rows
# Calculate mean and 95% CI for each column

n <- nrow(Y_test_CI_matrix)

# Function to calculate mean, SD, and 95% CI
calculate_stats <- function(column) {
  mean_val <- mean(column, na.rm = TRUE)
  sd_val <- sd(column, na.rm = TRUE)
  se <- sd_val / sqrt(n)
  margin_error <- qt(p = 0.975, df = n - 1) * se
  lower <- mean_val - margin_error
  upper <- mean_val + margin_error
  return(c(mean = mean_val, margin_error = margin_error))
}

# Apply the function to each column
stats <- t(apply(Y_test_CI_matrix, 2, calculate_stats))
stats_df <- as.data.frame(stats)
stats_df$Column <- 1:ncol(Y_test_CI_matrix)

# Rename columns
colnames(stats_df) <- c("Mean","Margin", "Index")


stats_df$Arima_test <- Y_hat_test_arima_DF
stats_df$Glmnet_test <- Y_hat_test_glmnet
stats_df$NHMM_Individual <- Y_hat_test_NHMM_DF
stats_df$Real_Values <- Y_test
summary(stats_df)
# Create a new column 'value_nhmm_updated' by adding 'margin' to 'value_nhmm'
stats_df <- stats_df %>%
  mutate(Upper = NHMM_Individual + Margin)
stats_df <- stats_df %>%
  mutate(Lower = NHMM_Individual - Margin)

stats_df <- stats_df %>%
  mutate(
    Is_in_IC = ifelse(Real_Values >= Lower & Real_Values <= Upper, 1, 0)
  )

# Plot using ggplot2
ggplot(stats_df, aes(x = Index)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "lightgray", alpha = 0.5) + # Shade 95% CI
  geom_line(aes(y = NHMM_Individual), color = "blue", size = 1) + # Plot the mean line
  geom_point(aes(y = NHMM_Individual), color = "blue", size = 2) + # Add points at the means
  geom_line(aes(y = Lower), color = "red", linetype = "dashed", size = 0.8) + # Lower bound (dashed red line)
  geom_line(aes(y = Upper), color = "red", linetype = "dashed", size = 0.8) + # Upper bound (dashed red line)
  labs(
    x = "Column Index",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 19),  # Increase size of x-axis label
    axis.title.y = element_text(size = 19),  # Increase size of y-axis label
    axis.text.x = element_text(size = 17.5),  # Increase size of x-axis ticks
    axis.text.y = element_text(size = 17.5)   # Increase size of y-axis ticks
  )
############################################


df_train <- rbind(S_hat_train, Y_hat_train, Y_training)
df_validation <- rbind(Best_S, Best_Y, Y_validation)
df_test <- rbind(S_hat_test, Y_hat_test, Y_test)

df_train <- data.frame(df_train)
df_validation <- data.frame(df_validation)
df_test <- data.frame(df_test)
stats_df <- data.frame(stats_df)

Algo <- c("MSPE_NHMM_Ind", "MSPE_ARIMA", "MSPE_GLMNET")
MSPE <- c(MSPE_Teste, MSPE_Teste_arima, MSPE_Teste_glmnet)
df_MSPE<-data.frame(Algo,MSPE)

Param <- c("Mu_1","Mu_2","Mu_3", "Sigma_1", "Sigma_2", "Sigma_3")
Estimado <-c(Mu_Hat[1],Mu_Hat[2],Mu_Hat[3], Sigma_Hat[1],Sigma_Hat[2], Sigma_Hat[3])
df_params <- data.frame(Param, Estimado)

lasso_Beta_estimates <- data.frame(lasso_Beta_estimates)

write.csv(df_train, paste("1_DF_Train_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(df_validation, paste("2_DF_Validation_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(df_test, paste("3_DF_Test_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(df_MSPE, paste("4_DF_MSPE_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(df_params, paste("5_DF_Params_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(lasso_Beta_estimates, paste("6_DF_all_LASSO_Beta_Estimates_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)
write.csv(stats_df, paste("7_DF_Test_DataStats_Application_K",toString(K),"_Global.csv", sep = ""), row.names=FALSE)

saveRDS(Best_Beta_Arrays, paste("8_Best_LASSO_Betas_Estimates_Application_K",toString(K),"_Global.rds", sep = ""))
