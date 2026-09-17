###########################################################################
#                                                                         #
#            APPLICATION INDIVIDUAL LASSO - 4 HIDDEN STATES               #
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
        resp <- sum(1 - log(1 + exp(Xtemp11%*%params[1:D])+ exp(Xtemp11%*%params[(D+1):(2*D)]))) + sum((Xtemp12%*%params[1:D]) - log( 1 + exp(Xtemp12%*%params[1:D])+ exp(Xtemp12%*%params[(D+1):(2*D)]) )) + sum((Xtemp13%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp13%*%params[1:D])+ exp(Xtemp13%*%params[(D+1):(2*D)]) )) + sum((Xtemp14%*%params[1:D]) - log( 1 + exp(Xtemp14%*%params[1:D])+ exp(Xtemp14%*%params[(D+1):(2*D)]) ))
  }

  FSM2 <-function(params){#função a maximizar para achar os Betas_2
    resp <- sum(1 - log(1 + exp(Xtemp21%*%params[1:D])+ exp(Xtemp21%*%params[(D+1):(2*D)]))) + sum((Xtemp22%*%params[1:D]) - log( 1 + exp(Xtemp22%*%params[1:D])+ exp(Xtemp22%*%params[(D+1):(2*D)]) )) + sum((Xtemp23%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp23%*%params[1:D])+ exp(Xtemp23%*%params[(D+1):(2*D)]) )) + sum((Xtemp24%*%params[1:D]) - log( 1 + exp(Xtemp24%*%params[1:D])+ exp(Xtemp24%*%params[(D+1):(2*D)]) ))
  }

  FSM3 <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp31%*%params[1:D])+ exp(Xtemp31%*%params[(D+1):(2*D)]))) + sum((Xtemp32%*%params[1:D]) - log( 1 + exp(Xtemp32%*%params[1:D])+ exp(Xtemp32%*%params[(D+1):(2*D)]) )) + sum((Xtemp33%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp33%*%params[1:D])+ exp(Xtemp33%*%params[(D+1):(2*D)]) )) + sum((Xtemp34%*%params[1:D]) - log( 1 + exp(Xtemp34%*%params[1:D])+ exp(Xtemp34%*%params[(D+1):(2*D)]) ))
  }

  FSM4 <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp41%*%params[1:D])+ exp(Xtemp41%*%params[(D+1):(2*D)]))) + sum((Xtemp42%*%params[1:D]) - log( 1 + exp(Xtemp42%*%params[1:D])+ exp(Xtemp42%*%params[(D+1):(2*D)]) )) + sum((Xtemp43%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp43%*%params[1:D])+ exp(Xtemp43%*%params[(D+1):(2*D)]) )) + sum((Xtemp44%*%params[1:D]) - log( 1 + exp(Xtemp44%*%params[1:D])+ exp(Xtemp44%*%params[(D+1):(2*D)]) ))
  }
    
  FSM1_B <-function(params){#função a maximizar para achar os Betas_1
    resp <- sum(1 - log(1 + exp(Xtemp11%*%params[1:D])+ exp(Xtemp11%*%params[(D+1):(2*D)]))) + sum((Xtemp12%*%params[1:D]) - log( 1 + exp(Xtemp12%*%params[1:D])+ exp(Xtemp12%*%params[(D+1):(2*D)]) )) + sum((Xtemp13%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp13%*%params[1:D])+ exp(Xtemp13%*%params[(D+1):(2*D)]) )) + sum((Xtemp14%*%params[1:D]) - log( 1 + exp(Xtemp14%*%params[1:D])+ exp(Xtemp14%*%params[(D+1):(2*D)]) )) - lambda1*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }

  FSM2_B <-function(params){#função a maximizar para achar os Betas_2
    resp <- sum(1 - log(1 + exp(Xtemp21%*%params[1:D])+ exp(Xtemp21%*%params[(D+1):(2*D)]))) + sum((Xtemp22%*%params[1:D]) - log( 1 + exp(Xtemp22%*%params[1:D])+ exp(Xtemp22%*%params[(D+1):(2*D)]) )) + sum((Xtemp23%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp23%*%params[1:D])+ exp(Xtemp23%*%params[(D+1):(2*D)]) )) + sum((Xtemp24%*%params[1:D]) - log( 1 + exp(Xtemp24%*%params[1:D])+ exp(Xtemp24%*%params[(D+1):(2*D)]) )) - lambda2*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }

  FSM3_B <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp31%*%params[1:D])+ exp(Xtemp31%*%params[(D+1):(2*D)]))) + sum((Xtemp32%*%params[1:D]) - log( 1 + exp(Xtemp32%*%params[1:D])+ exp(Xtemp32%*%params[(D+1):(2*D)]) )) + sum((Xtemp33%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp33%*%params[1:D])+ exp(Xtemp33%*%params[(D+1):(2*D)]) )) + sum((Xtemp34%*%params[1:D]) - log( 1 + exp(Xtemp34%*%params[1:D])+ exp(Xtemp34%*%params[(D+1):(2*D)]) )) - lambda3*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }

  FSM4_B <-function(params){#função a maximizar para achar os Betas_3
    resp <- sum(1 - log(1 + exp(Xtemp41%*%params[1:D])+ exp(Xtemp41%*%params[(D+1):(2*D)]))) + sum((Xtemp42%*%params[1:D]) - log( 1 + exp(Xtemp42%*%params[1:D])+ exp(Xtemp42%*%params[(D+1):(2*D)]) )) + sum((Xtemp43%*%params[(D+1):(2*D)]) - log( 1 + exp(Xtemp43%*%params[1:D])+ exp(Xtemp43%*%params[(D+1):(2*D)]) )) + sum((Xtemp44%*%params[1:D]) - log( 1 + exp(Xtemp44%*%params[1:D])+ exp(Xtemp44%*%params[(D+1):(2*D)]) )) - lambda4*(sum(abs(params[2:D])) + sum(abs(params[(D+2):(2*D)])))
  }


train_size = 0.80
validation_size = 0.15
test_size = 0.05


zero_threshold = 0.05
K=4   #Numero de estados ocultos
D=8   #Quantidade de Covariaveis
tol<-0.0000001 #Nivel de tolerancia que estabelecemos como criterio de parada do EM Est
tolval=NULL
tolval[1]=1
optim_algo = "BFGS" #Algorithm to use in the optimization process
n_max_iter_EM = 11
n_max_iter_EM_2 = 71
Tempo <- NULL
lag_var = TRUE

mainDir = paste("/Users/daianezuanetti/Library/CloudStorage/Dropbox/artigo_Gustavo/Códigos",sep = "")
subDir = paste("Lagged_",toString(lag_var),"_Resultados_Application_Global_K",toString(K),sep = "")
dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
setwd(file.path(mainDir, subDir))

set.seed(4)
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
init3 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_3
init4 = c(rnorm(D*(K-1), 0, 5))#Valores iniciais para os Betas_4

lasso_iterator = 1 #Criamos um contador para iterar a traves dos valores de lambda

# Algumas estruturas para almacenar valores gerados pelo LASSO
lasso_RMSE <- NULL
lasso_S <- matrix(nrow = length(lambdas)^K, ncol = length(Y_validation))
lasso_Y <- matrix(nrow = length(lambdas)^K, ncol = length(Y_validation))
lasso_mu_hat_estimates <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_sigma_hat_estimates <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_Beta_estimates <- matrix(nrow = length(lambdas)^K, ncol = D*K*(K-1))
lasso_Beta_arrays <- array(rep(0,K*D*K*(length(lambdas))^K), dim=c(K,D,K,(length(lambdas))^K))
lasso_lambdas <- matrix(nrow = length(lambdas)^K, ncol = K)
lasso_S_training <- matrix(nrow = length(lambdas)^K, ncol = length(Y_training))
lasso_Y_training <- matrix(nrow = length(lambdas)^K, ncol = length(Y_training))
# INICIO DO LASSO
###############################################
for (h in 1:length(lambdas)){
  for (w in 1:length(lambdas)){
    for (b in 1:length(lambdas)){
    	for (ab in 1:length(lambdas)){ #### fechar esse lambdas em algum lugar
      #setTxtProgressBar(pb, lasso_iterator)
      
      #Estruturas necessarias no processo de estimação
      mu_hat = NULL #Variavel para estimar os mus em cada iteração do EM Estocástico
      sigma_hat = NULL #Variavel para estimar os sigmas em cada iteração do EM Estocástico
      BetaArray = array(0, dim=c(K,D,K)) #Estrutura para guardar as estimativas dos Betas em cada iteração do EM
            
      VerProx<-NULL
      VerAct<-NULL
      
      lambda1 = lambdas[h]
      lambda2 = lambdas[w]
      lambda3 = lambdas[b]
      lambda4 = lambdas[ab]
            
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
          sigma_hat[k] = max(sqrt((sum((Y_id - mu_hat[k])^2)) / (sum(id) - 1)),0.001) #DECIDIR SOBRE O ESTIMADOR DA VARIANCIA (VICIADO OU NṼICIADO)
        }
        
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
                
      #Agora executamos o Algoritmo EM Estocástico
      while ( abs(tolval[val])>tol && val < n_max_iter_EM){

        val<-val+1       
        #Calculamos a sequência S_treino utilizando os Betas
        #Atualizados na iteração passada e os valores observados Y
#        S_treino[1]=which.max(dnorm(Y[1], mu_hat, sigma_hat))
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
            #S_treino[i]=which.max(prob)  
            S_treino[i]=which.max(prob)  
          }
        }
                    
        if (length(S_treino[is.na(S_treino)]) > 0){
          print(length(S_treino[is.na(S_treino)]))
          S_treino[is.na(S_treino)] <- 1
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
              positions = sample(2:length(S_treino), 10)
              for (d in 1:10) {
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
    Xtemp13<-NULL
    Xtemp14<-NULL
    Xtemp21<-NULL
    Xtemp22<-NULL
    Xtemp23<-NULL
    Xtemp24<-NULL
    Xtemp31<-NULL
    Xtemp32<-NULL
    Xtemp33<-NULL
    Xtemp34<-NULL    
    Xtemp41<-NULL
    Xtemp42<-NULL
    Xtemp43<-NULL
    Xtemp44<-NULL
    
    for (t in 2:length(Y_training)) {
      #filtros indo para o Estado # 1
      if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
        Xtemp11<-rbind(Xtemp11, X_training[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
        Xtemp21<-rbind(Xtemp21, X_training[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%3)
        Xtemp31<-rbind(Xtemp31, X_training[t,])

      if(S_treino[t]%in%1 && S_treino[t-1]%in%4)
        Xtemp41<-rbind(Xtemp41, X_training[t,])
              
      #Filtros indo para o Estado # 2
      if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
        Xtemp12<-rbind(Xtemp12, X_training[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
        Xtemp22<-rbind(Xtemp22, X_training[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%3)
        Xtemp32<-rbind(Xtemp32, X_training[t,])

      if(S_treino[t]%in%2 && S_treino[t-1]%in%4)
        Xtemp42<-rbind(Xtemp42, X_training[t,])
              
      #Filtros indo para o Estado # 3
      if(S_treino[t]%in%3 && S_treino[t-1]%in%1)
        Xtemp13<-rbind(Xtemp13, X_training[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%2)
        Xtemp23<-rbind(Xtemp23, X_training[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%3)
        Xtemp33<-rbind(Xtemp33, X_training[t,])
        
      if(S_treino[t]%in%3 && S_treino[t-1]%in%4)
        Xtemp43<-rbind(Xtemp43, X_training[t,])
        
      #Filtros indo para o Estado # 4
      if(S_treino[t]%in%4 && S_treino[t-1]%in%1)
        Xtemp14<-rbind(Xtemp14, X_training[t,])
      
      if(S_treino[t]%in%4 && S_treino[t-1]%in%2)
        Xtemp24<-rbind(Xtemp24, X_training[t,])
      
      if(S_treino[t]%in%4 && S_treino[t-1]%in%3)
        Xtemp34<-rbind(Xtemp34, X_training[t,])
        
      if(S_treino[t]%in%4 && S_treino[t-1]%in%4)
        Xtemp44<-rbind(Xtemp44, X_training[t,])
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
    if (is.null(Xtemp41)){
      Xtemp41 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp41[,1] <- 1
      print("Encontrou-se X41 vazio. Gerando 1 valor aleatorio.")
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
    if (is.null(Xtemp42)){
      Xtemp42 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp42[,1] <- 1
      print("Encontrou-se X42 vazio. Gerando 1 valor aleatorio.")
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
    if (is.null(Xtemp43)){
      Xtemp43 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp43[,1] <- 1
      print("Encontrou-se X43 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp14)){
      Xtemp14 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp14[,1] <- 1
      print("Encontrou-se X14 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp24)){
      Xtemp24 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp24[,1] <- 1
      print("Encontrou-se X24 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp34)){
      Xtemp34 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp34[,1] <- 1
      print("Encontrou-se X34 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp44)){
      Xtemp44 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp44[,1] <- 1
      print("Encontrou-se X44 vazio. Gerando 1 valor aleatorio.")
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

    fit4 <- tryCatch( 
      {
        optim(par = init4, fn = FSM4, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
      },
      error = function(e) {
        print("Finite-NonFinite difference found when using BFGS .... Reverting to Nelder-Mead")
        optim(par = init4, fn = FSM4, control = list(fnscale=-1), method = "Nelder-Mead", hessian = FALSE)
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
        } else if (i == 4){
          BetaArray[i,d,1]=fit1$par[2*D+d]
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
        } else if (i == 4){
          BetaArray[i,d,2]=fit2$par[2*D+d]
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
        } else if (i == 4){
          BetaArray[i,d,3]=fit3$par[2*D+d]
        }
        
      }
    }

    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,4]=0
        } else if (i == 2){
          BetaArray[i,d,4]=fit4$par[d]
        } else if (i == 3){
          BetaArray[i,d,4]=fit4$par[D+d]
        } else if (i == 4){
          BetaArray[i,d,4]=fit4$par[2*D+d]
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
        if(is.nan(VeroSimProxima) | is.nan(VeroSimActual))
          tolval[val] <- 0 
        # print(tolval[val])
        
        message(paste('\r',"Lasso iteration # ",toString(lasso_iterator),"; Valor de Lambda = ",toString(c(lambda1,lambda2,lambda3,lambda4)),"; Mu_hat:",toString(round(mu_hat,3)),". Sigma_hat:",toString(round(sigma_hat,3)),"                  ", collapse = ""), appendLF = FALSE) #Messagem indicando o numero da replica atual
      }#######Fim da primeira rodada do EM Estocastico#######
            
      val=1
      tolval=NULL
      tolval[1]=10
      tol2 = 0.5

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
    Xtemp13<-NULL
    Xtemp14<-NULL
    Xtemp21<-NULL
    Xtemp22<-NULL
    Xtemp23<-NULL
    Xtemp24<-NULL
    Xtemp31<-NULL
    Xtemp32<-NULL
    Xtemp33<-NULL
    Xtemp34<-NULL    
    Xtemp41<-NULL
    Xtemp42<-NULL
    Xtemp43<-NULL
    Xtemp44<-NULL
    
    for (t in 2:length(Y_training)) {
      #filtros indo para o Estado # 1
      if(S_treino[t]%in%1 && S_treino[t-1]%in%1)
        Xtemp11<-rbind(Xtemp11, X_training[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%2)
        Xtemp21<-rbind(Xtemp21, X_training[t,])
      
      if(S_treino[t]%in%1 && S_treino[t-1]%in%3)
        Xtemp31<-rbind(Xtemp31, X_training[t,])

      if(S_treino[t]%in%1 && S_treino[t-1]%in%4)
        Xtemp41<-rbind(Xtemp41, X_training[t,])
              
      #Filtros indo para o Estado # 2
      if(S_treino[t]%in%2 && S_treino[t-1]%in%1)
        Xtemp12<-rbind(Xtemp12, X_training[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%2)
        Xtemp22<-rbind(Xtemp22, X_training[t,])
      
      if(S_treino[t]%in%2 && S_treino[t-1]%in%3)
        Xtemp32<-rbind(Xtemp32, X_training[t,])

      if(S_treino[t]%in%2 && S_treino[t-1]%in%4)
        Xtemp42<-rbind(Xtemp42, X_training[t,])
              
      #Filtros indo para o Estado # 3
      if(S_treino[t]%in%3 && S_treino[t-1]%in%1)
        Xtemp13<-rbind(Xtemp13, X_training[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%2)
        Xtemp23<-rbind(Xtemp23, X_training[t,])
      
      if(S_treino[t]%in%3 && S_treino[t-1]%in%3)
        Xtemp33<-rbind(Xtemp33, X_training[t,])
        
      if(S_treino[t]%in%3 && S_treino[t-1]%in%4)
        Xtemp43<-rbind(Xtemp43, X_training[t,])
        
      #Filtros indo para o Estado # 4
      if(S_treino[t]%in%4 && S_treino[t-1]%in%1)
        Xtemp14<-rbind(Xtemp14, X_training[t,])
      
      if(S_treino[t]%in%4 && S_treino[t-1]%in%2)
        Xtemp24<-rbind(Xtemp24, X_training[t,])
      
      if(S_treino[t]%in%4 && S_treino[t-1]%in%3)
        Xtemp34<-rbind(Xtemp34, X_training[t,])
        
      if(S_treino[t]%in%4 && S_treino[t-1]%in%4)
        Xtemp44<-rbind(Xtemp44, X_training[t,])
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
    if (is.null(Xtemp41)){
      Xtemp41 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp41[,1] <- 1
      print("Encontrou-se X41 vazio. Gerando 1 valor aleatorio.")
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
    if (is.null(Xtemp42)){
      Xtemp42 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp42[,1] <- 1
      print("Encontrou-se X42 vazio. Gerando 1 valor aleatorio.")
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
    if (is.null(Xtemp43)){
      Xtemp43 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp43[,1] <- 1
      print("Encontrou-se X43 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp14)){
      Xtemp14 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp14[,1] <- 1
      print("Encontrou-se X14 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp24)){
      Xtemp24 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp24[,1] <- 1
      print("Encontrou-se X24 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp34)){
      Xtemp34 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp34[,1] <- 1
      print("Encontrou-se X34 vazio. Gerando 1 valor aleatorio.")
    }
    if (is.null(Xtemp44)){
      Xtemp44 <- matrix(rnorm(D), nrow = 1, ncol = D)
      Xtemp44[,1] <- 1
      print("Encontrou-se X44 vazio. Gerando 1 valor aleatorio.")
    }
        
        fit1 <- optim(par = init1, fn = FSM1_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        fit2 <- optim(par = init2, fn = FSM2_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        fit3 <- optim(par = init3, fn = FSM3_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        fit4 <- optim(par = init4, fn = FSM4_B, control = list(fnscale=-1), method = optim_algo, hessian = FALSE)
        
    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,1]=0
        } else if (i == 2){
          BetaArray[i,d,1]=fit1$par[d]
        } else if (i == 3){
          BetaArray[i,d,1]=fit1$par[D+d]
        } else if (i == 4){
          BetaArray[i,d,1]=fit1$par[2*D+d]
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
        } else if (i == 4){
          BetaArray[i,d,2]=fit2$par[2*D+d]
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
        } else if (i == 4){
          BetaArray[i,d,3]=fit3$par[2*D+d]
        }
        
      }
    }

    for (i in 1:K){
      for (d in 1:D){
        if (i == 1){
          BetaArray[i,d,4]=0
        } else if (i == 2){
          BetaArray[i,d,4]=fit4$par[d]
        } else if (i == 3){
          BetaArray[i,d,4]=fit4$par[D+d]
        } else if (i == 4){
          BetaArray[i,d,4]=fit4$par[2*D+d]
        }
        
      }
    }    

#    S_treino[1]=which.max(dnorm(Y[1], mu_hat, sigma_hat))
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
      lasso_lambdas[lasso_iterator, ] <- c(lambdas[h],lambdas[w],lambdas[b],lambdas[ab])
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
  }
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
#  prob<-NULL
  for (i in 1:K) prob[i]<-exp(X_test[t,]%*%matrix(Best_Beta_Arrays[i,,S_hat_test[t-1]],ncol=1))
  prob<-prob/sum(prob)
#  print(prob)
  S_hat_test[t]<-which.max(prob)
  Y_hat_test[t]<-sum(prob * Mu_Hat)
}

MSPE_Teste <- (sum((Y_hat_test - Y_test)^2))/length(Y_test)
MSPE_Teste
