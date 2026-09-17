###########################################################################
#                                                                         #
#            APPLICATION INDIVIDUAL LASSO - 2 HIDDEN STATES               #
#                                                                         #
###########################################################################

library('label.switching')
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

## Escrevemos a função para recalcular a matriz de transição em cada iteração
## do algoritmo EM Estocástico.
  Mat_trans <-function(covar,BetaArray){
    B = matrix(nrow=K, ncol=K)
    for (j in 1:K) {
      numerator<-NULL
      for (i in 1:K) numerator[i] = covar%*%BetaArray[i,,j]
      numerator<-exp(numerator-max(numerator))
      B[,j] = numerator/sum(numerator)
      }  
    return(B)
  }

  #######   Escrevemos as funções que serão o objetivo da optimização   ######
  # Com o temos um array de Betas, utilizaremos tres funções para achar os valores otimos
  # Uma para a matriz Betas[,,1] uma para a matriz Betas[,,2] e uma para 
  # a matriz Betas[,,3]
  FSM1 <-function(params){#função a maximizar para achar os Betas_1
    resp <- (sum(1 - log(1 + exp(Xtemp11%*%params))) + sum(Xtemp12%*%params - log(1 + exp(Xtemp12%*%params))))
  }
  
  FSM2 <-function(params){#função a maximizar para achar os Betas_2
    resp <- (sum(1 - log(1 + exp(Xtemp21%*%params))) + sum(Xtemp22%*%params - log(1 + exp(Xtemp22%*%params))))
  }
  
  FSM1_B <-function(params){#função a maximizar para achar os Betas_1
    resp <- (sum(1 - log(1 + exp(Xtemp11%*%params))) + sum(Xtemp12%*%params - log(1 + exp(Xtemp12%*%params)))) - lambda1*sum(abs(params[2:D])) 
  }
  
  FSM2_B <-function(params){#função a maximizar para achar os Betas_2
    resp <- (sum(1 - log(1 + exp(Xtemp21%*%params))) + sum(Xtemp22%*%params - log(1 + exp(Xtemp22%*%params))))  - lambda2*sum(abs(params[2:D]))
  }
  

train_size = 0.80
validation_size = 0.15
test_size = 0.05


zero_threshold = 0.05
K=2   #Numero de estados ocultos
D=8   #Quantidade de Covariaveis
tol<-0.0000001 #Nivel de tolerancia que estabelecemos como criterio de parada do EM Est
tolval=NULL
tolval[1]=1
optim_algo = "BFGS" #Algorithm to use in the optimization process
n_max_iter_EM = 11
n_max_iter_EM_2 = 51
Tempo <- NULL
lag_var = TRUE

mainDir = paste("/Users/daianezuanetti/Library/CloudStorage/Dropbox/artigo_Gustavo/Códigos",sep = "")
subDir = paste("Lagged_",toString(lag_var),"_Resultados_Application_Global_K",toString(K),sep = "")
dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
setwd(file.path(mainDir, subDir))

set.seed(100)
lambdas <- seq(0.0, 0.04, by=0.01)

#Metricas de Performance Preditiva 
MSPE_Validação <- NULL
MSPE_Teste <- NULL


#Metricas de Performance de Estimação dos ParÂmetros das VA observáveis
Best_Beta_Arrays <- array(rep(0,K*D*K), dim=c(K,D,K))

## SEÇÃO DE DEFINICAÇÃO DOS PARAMETROS PARA SIMULAÇÃO DE DADOS ##
################################################################
P0=rep(1/K,K) #Inicializamos vetor de probabilidades inciais para o HMM

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

###########################################
#   FIM DA SIMULAÇÂO DOS DADOS ##

#   SEPARAR BASES EM TREINO< VALIDAÇÃO E TESTE
##############################################

#Calcula os indices de corte
cutoff_treino = length(Y)*train_size
cutoff_validation = length(Y)*(train_size+validation_size)

#Cria as bases 
Y_training = Y[1:cutoff_treino]
X_training = X[1:cutoff_treino,c(1,3,6,9,12,14,15,16)]

Y_validation = Y[(cutoff_treino+1):cutoff_validation]
X_validation = X[(cutoff_treino+1):cutoff_validation,c(1,3,6,9,12,14,15,16)]

Y_test = Y[(cutoff_validation+1):T]
X_test = X[(cutoff_validation+1):T,c(1,3,6,9,12,14,15,16)]

##############################################
# FIM DE SEPARAÇÃO DAS BASES EM TREINO, VALIDATION E TESTE


# INICIO DO PROCESSO DE ESTIMAÇÃO
##########################################
# Primeiro geramos uma sequência não observavel de treinamento
P_Treino=rep(1/K,K) #Vetor de probabilidade utilizadas para gerar a sequência de treino
S_treino<-NULL # Inicializamos a sequência oculta de treinamento

init1 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_1
init2 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_2

lasso_iterator = 1 #Criamos um contador para iterar a traves dos valores de lambda


# Algumas estruturas para almacenar valores gerados pelo LASSO
lasso_RMSE <- NULL
lasso_S <- matrix(nrow = length(lambdas)^K, ncol = length(Y_validation))
lasso_Y <- matrix(nrow = length(lambdas)^K, ncol = length(Y_validation))
lasso_mu_hat_estimates <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_sigma_hat_estimates <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_Beta_estimates <- matrix(nrow = length(lambdas)^K, ncol = D*K*(K-1))
lasso_Beta_arrays <- array(rep(0,K*D*K*length(lambdas)^K), dim=c(K,D,K,length(lambdas)^K))
lasso_S_training <- matrix(nrow = length(lambdas)^K, ncol = length(Y_training))
lasso_Y_training <- matrix(nrow = length(lambdas)^K, ncol = length(Y_training))
# INICIO DO LASSO
###############################################
pb <- txtProgressBar(min = 1, max = length(lambdas), style = 3)
print('Executando LASSO...')
for (h in 1:length(lambdas)){
  for (w in 1:length(lambdas)){
    #setTxtProgressBar(pb, lasso_iterator)
    
    #Estruturas necessarias no processo de estimação
    mu_hat = NULL #Variavel para estimar os mus em cada iteração do EM Estocástico
    sigma_hat = NULL #Variavel para estimar os sigmas em cada iteração do EM Estocástico
    BetaArray = array(0, dim=c(K,D,K)) #Estrutura para guardar as estimativas dos Betas em cada iteração do EM
    
    
    VerProx<-NULL
    VerAct<-NULL
    
    lambda1 = lambdas[h]
    lambda2 = lambdas[w]
    
    #   Procedimento de Estimação   
    #Geramos uma sequência de treinamento
    for (i in 1:length(Y_training)) {
      S_treino[i] = rDiscreta(P_Treino)
    }
    
    val=1

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
          temp[g]<-exp(X_training[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
        }
        LL_parte4 = LL_parte4 + (X_training[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
      }
      VeroSimActual <- log(P0[S_treino[1]]) + LL_parte1 + (LL_parte2 + LL_parte3) + LL_parte4 #calculo da LogVerosim
      tolval[1]=VeroSimActual

    while (abs(tolval[val])>tol && val < n_max_iter_EM){
      
      val=val+1
      #Calculamos a sequência S_treino utilizando os Betas
      #Atualizados na iteração passada e os valores observados Y
#      S_treino[1]=which.max(dnorm(Y[1], mu_hat, sigma_hat))
      for (i in 2:length(Y_training)) {
        A_hat_t = Mat_trans(X_training[i,],BetaArray)
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
 
     for (k in 1:K){
      id = S_treino == k
      mu_hat[k] = sum(id*Y_training)/sum(id)
      Y_id_list = split(Y_training,id)
      Y_id = unlist(Y_id_list[2], use.names = FALSE)
      sigma_hat[k] = max(sqrt((sum((Y_id - mu_hat[k])^2)) / (sum(id) - 1)),0.001) #DECIDIR SOBRE O ESTIMADOR DA VARIANCIA (VICIADO OU NṼICIADO)
    }
     
      #### Aqui inicia a filtragem dos dados para cada iteração
      Xtemp11<-NULL
      Xtemp12<-NULL
      Xtemp21<-NULL
      Xtemp22<-NULL
      
      for (t in 2:length(Y_training)) {
        #filtros indo para o Estado # 1
        if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
          Xtemp11<-rbind(Xtemp11, X_training[t,])
        
        if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
          Xtemp21<-rbind(Xtemp21, X_training[t,])
        
        #Filtros indo para o Estado # 2
        if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
          Xtemp12<-rbind(Xtemp12, X_training[t,])
        
        if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
          Xtemp22<-rbind(Xtemp22, X_training[t,])
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
          temp[g]<-exp(X_training[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
        }
        LL2_parte4 = LL2_parte4 + (X_training[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
      }
      VeroSimProxima <- log(P0[S_treino[1]]) + LL2_parte1 + (LL2_parte2 + LL2_parte3) + LL2_parte4 #calculo da LogVerosim
      
      VerAct[val]<-VeroSimActual
      VerProx[val]<-VeroSimProxima
      tolval[val]<-VeroSimProxima - VeroSimActual
      VeroSimActual<-VeroSimProxima
      # print(tolval[val])
      
      message(paste('\r',"Lasso iteration # ",toString(lasso_iterator),"; Valor de Lambda = ",toString(c(lambda1,lambda2)),"; Mu_hat:",toString(round(mu_hat,3)),". Sigma_hat:",toString(round(sigma_hat,3)),"                  ", collapse = ""), appendLF = FALSE) #Messagem indicando o numero da replica atual
    }#######Fim da primeira rodada do EM Estocastico#######
        
    val=1
    tolval=NULL
    tolval[1]=10
    tol2 = 2

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
          temp[g]<-exp(X_training[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
        }
        LL_parte4 = LL_parte4 + (X_training[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
      }
      VeroSimActual <- log(P0[S_treino[1]]) + LL_parte1 - (LL_parte2 + LL_parte3) + LL_parte4 #calculo da LogVerosim
    
  while (abs(tolval[val])>tol2 && val < n_max_iter_EM_2) {

  	val<-val+1 
  	      
      #filtragem dos dados
      Xtemp11<-NULL
      Xtemp12<-NULL
      Xtemp21<-NULL
      Xtemp22<-NULL
      
      for (t in 2:length(Y_training)) {
        #filtros indo para o Estado # 1
        if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
          Xtemp11<-rbind(Xtemp11, X_training[t,])
        
        if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
          Xtemp21<-rbind(Xtemp21, X_training[t,])
        
        #Filtros indo para o Estado # 2
        if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
          Xtemp12<-rbind(Xtemp12, X_training[t,])
        
        if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
          Xtemp22<-rbind(Xtemp22, X_training[t,])
        
      }
      
      if (is.null(Xtemp11)){
        Xtemp11 <- matrix(rnorm(D), nrow = 1, ncol = D)
        Xtemp11[,1] <- 1
      }
      if (is.null(Xtemp21)){
        Xtemp21 <- matrix(rnorm(D), nrow = 1, ncol = D)
        Xtemp21[,1] <- 1
      }
      if (is.null(Xtemp12)){
        Xtemp12 <- matrix(rnorm(D), nrow = 1, ncol = D)
        Xtemp12[,1] <- 1
      }
      if (is.null(Xtemp22)){
        Xtemp22 <- matrix(rnorm(D), nrow = 1, ncol = D)
        Xtemp22[,1] <- 1
      }
      
      fit1 <- tryCatch( 
        {
          optim(par = init1, fn = FSM1_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        },
        error = function(e) {
          print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
          optim(par = init1, fn = FSM1_B, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
        }
      )
      
      fit2 <- tryCatch( 
        {
          optim(par = init2, fn = FSM2_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        },
        error = function(e) {
          print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
          optim(par = init2, fn = FSM2_B, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
        }
      )
      
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
      
    for (i in 2:length(Y_training)) {
      A_hat_t = Mat_trans(X_training[i,],BetaArray)
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
#
    for (k in 1:K){
      id = S_treino == k
      mu_hat[k] = sum(id*Y_training)/sum(id)
      Y_id_list = split(Y_training,id)
      Y_id = unlist(Y_id_list[2], use.names = FALSE)
      sigma_hat[k] = max(sqrt((sum((Y_id - mu_hat[k])^2)) / (sum(id) - 1)),0.001) #DECIDIR SOBRE O ESTIMADOR DA VARIANCIA (VICIADO OU NṼICIADO)
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
      }
      temp=NULL
      for (i in 2:length(Y_training)) {#Calculo do terceiro segmento da LL
        for (g in 1:K) {
          temp[g]<-exp(X_training[i,]%*%matrix(BetaArray[g,,S_treino[i-1]],ncol=1))
        }
        LL2_parte4 = LL2_parte4 + (X_training[i,]%*%matrix(BetaArray[S_treino[i],,S_treino[i-1]]) - log(sum(temp), base = exp(1)))
      }
      VeroSimProxima <- log(P0[S_treino[1]]) + LL2_parte1 - (LL2_parte2 + LL2_parte3) + LL2_parte4 #calculo da LogVerosim
      
    tolval[val]<-VeroSimProxima-VeroSimActual
    VeroSimActual<-VeroSimProxima

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
  prob<-NULL
  t<-1
  for (i in 1:K) prob[i]<-exp(X_validation[t,]%*%matrix(BetaArray[i,,S_treino[length(Y_training)]],ncol=1))
  prob<-prob/sum(prob)
  S_hat_validation[1]<-which.max(prob)
  Y_hat_validation[1]<-sum(prob * mu_hat)
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
  }
} ##################################################
# FIM DO PROCESSO DE ESTIMAÇÃO (LASSO)

# CAPTURA INDICE DO LAMBDA COM MELHORES RESULTADOS
min_index = which.min(lasso_RMSE)

# CAPTURA DE METRICAS PARA CADA REPLICA
##################################################################


# COLETANDO VALORES NO CONJUNTO DE VALIDAÇÃO

# Valor de Lambda optimo
Best_Lambdas <- lasso_lambdas[min_index,]

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

set.seed(100)
t<-1
prob<-NULL
for (i in 1:K) prob[i]<-exp(X_test[t,]%*%matrix(Best_Beta_Arrays[i,,lasso_S[min_index,ncol(lasso_S)]],ncol=1))
prob<-prob/sum(prob)
print(prob)
S_hat_test[t]<-which.max(prob)
Y_hat_test[t]<-sum(prob * Mu_Hat)
for (t in 2:length(Y_test)){
  prob<-NULL
  for (i in 1:K) prob[i]<-exp(X_test[t,]%*%matrix(Best_Beta_Arrays[i,,S_hat_test[t-1]],ncol=1))
  prob<-prob/sum(prob)
#  print(prob)
  S_hat_test[t]<-which.max(prob)
  Y_hat_test[t]<-sum(prob * Mu_Hat)
}

MSPE_Teste <- (sum((Y_hat_test - Y_test)^2))/length(Y_test)
MSPE_Teste
