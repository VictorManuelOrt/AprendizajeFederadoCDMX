#deuda_publica_2024_01 <- read.csv("C:/Users/5/Downloads/deuda_publica_2024_01.csv")

deuda_publica_2024_01 <- read.csv("~/TESIS Ms Cs Mt/Differential Privacy Codigos y bases de datos/DeudaPCDMX/deuda_publica_2024_01.csv")


#Seleccion de variables

DATA=deuda_publica_2024_01[,c(7,11:22)]

DATANUM=DATA[,c (1:4, 6:9)]

#Limpieza de datos vacios

DATANUM <- DATANUM[rowSums(DATANUM == "**") == 0 & rowSums(DATANUM == "") == 0 & complete.cases(DATANUM), ]

DATANUM1=DATANUM[,1]

# Estandarización (Z-score)
DATANUM <- as.data.frame(lapply(DATANUM[,-1], function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}))

# Ver el data frame estandarizado
head(DATANUM)

DATANUM=cbind(DATANUM1,DATANUM)

NamesBancos=unique(DATANUM[,1])


Etiquetas=list()


#for (i in 1:length(NamesBancos)) {
#  Etiquetas[[i]]=which(DATANUM[,1]==NamesBancos[[i]])
#}

DatosPorBanco=list()


for (i in 1:length(NamesBancos)) {
  DatosPorBanco[[i]]=DATANUM[which(DATANUM[,1]==NamesBancos[[i]]),]
}


#Buscamos si hay correlacion

correlation_matrix <- cor(DatosPorBanco[[2]][,-1], use = "complete.obs")
print(correlation_matrix)


modelo=list()

# Ajustar el modelo de regresión lineal
for (i in 1:length(NamesBancos)) {
  
modelo[[i]] <- lm(saldo_periodo ~ dias_contrato                  
             + dias_restantes_contrato        
                
             + amortizaciones_periodo        
             + intereses_periodo , data = DatosPorBanco[[i]])

}
#             + pago_servicio_deuda + disposicion_inicial_credito 

# Resumen del modelo
summary(modelo)


# Gráficos de diagnóstico
#plot(modelo)

  
  # Predicciones
  predicciones=list()
  
  for (i in 1:length(NamesBancos)) {
    
  predicciones[[i]] <- predict(modelo[[i]], newdata = DatosPorBanco[[i]])
  }
  
  
  
  
  
  # Calcular el error cuadrático medio
  valores_observados=list()
  for (i in 1:length(NamesBancos)) {
  valores_observados[[i]] <- DatosPorBanco[[i]]$saldo_periodo  
  }
  
  
  mse=list()
  for (i in 1:length(NamesBancos)) {
  mse[[i]] <- mean((valores_observados[[i]] - predicciones[[i]])^2)
  }
  
  
  # Mostrar el resultado
  print(mse)
  
  
  
  localparametros=list()
  for (i in 1:length(NamesBancos)) {
    localparametros[[i]] <- modelo[[i]]$coefficients
  }
  


w_global <- Reduce("+", localparametros) /length(NamesBancos) 
w_globallist=list(Coefficients=w_global)


globalpredicciones=rep(0,nrow(DATANUM))
 for (j in 1:nrow(DATANUM)) {
  globalpredicciones[j]=  as.numeric(w_global[1])+sum(as.vector(w_global[-1])*DATANUM[j,c(2,3,5,6)])
  
}



MSEGLOBAL=mean((DATANUM$saldo_periodo - globalpredicciones)^2)
print(MSEGLOBAL)




#De manera teorica:
modeloTEOR <- lm(saldo_periodo ~ dias_contrato                  
                  + dias_restantes_contrato        
                  
                  + amortizaciones_periodo        
                  + intereses_periodo , data = DATANUM)


prediccionesTEOR <- predict(modeloTEOR, newdata = DATANUM)
MSEGLOBALTEOR=mean((DATANUM$saldo_periodo - prediccionesTEOR)^2)


print(MSEGLOBALTEOR)






print(mse)
print(MSEGLOBAL)
print(MSEGLOBALTEOR)




####SDG


# Paquetes necesarios
library(MASS)
library(dplyr)

# Función para recortar gradientes
clip_gradient <- function(grad, C) {
  norm_grad <- sqrt(sum(grad^2))
  if (norm_grad > C) {
    return(C * grad / norm_grad)
  } else {
    return(grad)
  }
}

# Función para añadir ruido gaussiano
add_noise <- function(grad, sigma, C, B) {
  noise <- rnorm(length(grad), mean = 0, sd = sigma * C / (B*epsilon))
  return(grad + noise)
}

# Función de costo: Error Cuadrático Medio (MSE)
mse_cost <- function(y_true, y_pred) {
  return(mean((y_true - y_pred)^2))
}

# Función ClientUpdate
ClientUpdate <- function(Xclient, yclient, w, learning_rate, epochs, batch_size, C, sigma) {
  n <- nrow(Xclient)
  for (epoch in 1:epochs) {
    # Barajar los datos
    indices <- sample(1:n)
    Xclient <- Xclient[indices, ]
    yclient <- yclient[indices]
    
    for (i in seq(1, n, by = batch_size)) {
      end_idx <- min(i + batch_size - 1, n)
      X_batch <- Xclient[i:end_idx, ,drop = FALSE]
      y_batch <- yclient[i:end_idx]
      
      # Predicción
      y_pred <- X_batch %*% w
      
      dif=(y_batch - y_pred)
      # Gradiente con respecto a MSE
      grad <- -2 * t(X_batch) %*% (y_batch - y_pred) / batch_size
      
      # Recortar gradiente
      grad <- clip_gradient(grad, C)
      
      # Añadir ruido gaussiano
      grad <- add_noise(grad, sigma, C, batch_size)
      
      # Actualizar los parámetros
      w <- w - learning_rate * grad
    }
  }
  return(w)
}

# Función FedAvg
FedAvg <- function(X_list, y_list, initial_w, learning_rate, epochs, batch_size, C, sigma, rounds) {
  K <- length(X_list)
  w_global <- initial_w
  
  for (round in 1:rounds) {
    w_local_list <- list()
    
    for (k in 1:K) {
      w_local <- ClientUpdate(X_list[[k]], y_list[[k]], w_global, learning_rate, epochs, batch_size, C, sigma)
      w_local_list[[k]] <- w_local
    }
    
    # Promediar los pesos locales
    w_global <- Reduce("+", w_local_list) / K
  }
  
  return(w_global)
}


# Datos simulados para cada cliente
n_clients <- length(NamesBancos)
IT <- 100
#Usando base de datos mtcars

X <- as.matrix(DATANUM[,c(2,3,5,6)])  # Variables independientes
y <- DATANUM[,8]  # Variable dependiente


X_list=list() 
for (j in 1:length(NamesBancos)) {
  X_list[[j]]=as.matrix(DatosPorBanco[[j]][,c(2,3,5,6)])
}

y_list =list()
for (j in 1:length(NamesBancos)) {
  y_list[[j]]=as.matrix(DatosPorBanco[[j]][,8])
}


# Parámetros
n_features <- ncol(X)
initial_w <- rep(0, n_features)
w_global <- initial_w


learning_rate <- 0.01
epochs <- 10

#2,3,6,10,15

batch_size <- 6
C <- 1  # Límite de recorte
sigma <- 0.1  # Desviación estándar del ruido
rounds <- 1  # Número de rondas de agregación





epsilon=1











set.seed(NULL)

IT <- 1000
w_global <- initial_w


Costos=rep(0, length(IT))

# Ejecutar FedAvg por IT iteraciones
for (i in 1:IT) {
  w_global <- FedAvg(X_list, y_list, w_global, learning_rate, epochs, batch_size, C, sigma, rounds)
  
  # Calcular costo global (promedio del costo de todos los clientes)
  Data=X
  rownames(Data) <- NULL
  colnames(Data) <- NULL
  
  costo_global <- sum((as.vector(y)-Data%*% as.vector(w_global) )^2) / nrow(X)
  Costos[i] <- costo_global
  # Imprimir resultados
  #cat("Iteración:", i, "\n")
  #cat("Pesos globales:", w_global, "\n")
  #cat("Costo global (MSE):", costo_global, "\n\n")
}



plot(Costos, type="l", xlab  =paste0("Número de Iteración"), ylab="Error" ,
     main = paste0(expression("Evolución del Error")))
legend("topright", # Posición de la leyenda
       legend=c(paste("Epsilon =", epsilon), 
                paste("# de iteraciones =", IT)), # Texto de la leyenda
       #col="black", # Color de la línea
       #lty=1, # Tipo de línea
       bty="n") # Eliminar el marco de la leyenda

boxplot(Costos,main = paste0(expression("Boxplot del Error")))
legend("topright", # Posición de la leyenda
       legend=c(paste("Epsilon =", epsilon), 
                paste("# de iteraciones =", IT)), # Texto de la leyenda
       #col="black", # Color de la línea
       #lty=1, # Tipo de línea
       bty="n") # Eliminar el marco de la leyenda

costo_global
MSEGLOBAL



MSEGLOBALTEOR



Prediccion= Data%*% as.vector(w_global)
DATANUM$saldo_periodo

plot(Prediccion,DATANUM$saldo_periodo, ylab = "valor teórico", xlim = c(0,4),
     ylim = c(0,4), main="Dispersión entre valores reales y predichos ")
abline(a = 0, b = 1, col = "red")  # a = 0, b = 1 para y = x


hist(Costos,breaks = 50, main="Histograma de rrores residuales")




Valepsilon=c(0.00001,0.0001,0.001,0.01,0.1,1,10,100, 1000)
ERRORporepsi=rep(0,9)
for (er in 1:length(Valepsilon)) {
  
  
  epsilon=Valepsilon[er]
  
  set.seed(NULL)
  
  IT <- 100
  w_global <- initial_w
  
  
  Costos=rep(0, length(IT))
  
  # Ejecutar FedAvg por IT iteraciones
  for (i in 1:IT) {
    w_global <- FedAvg(X_list, y_list, w_global, learning_rate, epochs, batch_size, C, sigma, rounds)
    
    # Calcular costo global (promedio del costo de todos los clientes)
    Data=X
    rownames(Data) <- NULL
    colnames(Data) <- NULL
    
    costo_global <- sum((as.vector(y)-Data%*% as.vector(w_global) )^2) / nrow(X)
    Costos[i] <- costo_global
    # Imprimir resultados
    #cat("Iteración:", i, "\n")
    #cat("Pesos globales:", w_global, "\n")
    #cat("Costo global (MSE):", costo_global, "\n\n")
  }
  ERRORporepsi[er]=mean(Costos)
}
mean(Costos)

Costos[length(Costos)]

mi_dataframe <- data.frame("valor de epsilon" = Valepsilon, "error promedio" = ERRORporepsi)



