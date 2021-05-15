########## William - Previsoes Intervalares ##########
library(dplyr)
library(keras)
library(ggplot2)
library(latex2exp)

### Importacao dos Dados de Treino e Teste
treino   <- readRDS("C:/Users/willi/Documents/UNB/8º Semestre/Tópicos em Estatística 1 - Redes Neurais/Seminários/Trabalho/Interpretacao-RNA/Treino_Lista1.rds")
teste    <- readRDS("C:/Users/willi/Documents/UNB/8º Semestre/Tópicos em Estatística 1 - Redes Neurais/Seminários/Trabalho/Interpretacao-RNA/Teste_Lista1.rds")
x_treino <- treino %>%
  select(x1.obs, x2.obs)
x_teste  <- teste %>%
  select(x1.obs, x2.obs)
y_treino <- treino$y
y_teste  <- teste$y

### Ajuste da Rede Neural Ruim
## Iniciamos um modelo sequencial
mod1 <- keras_model_sequential()
## Adicionamos as camadas
mod1 %>%
  layer_dense(units              = 2,
              activation         = "sigmoid",
              use_bias           = T,
              kernel_initializer = initializer_constant(0),
              bias_initializer   = initializer_constant(0),
              input_shape        = ncol(x_treino)) %>%
  layer_dense(units              = 1,
              activation         = "linear",
              use_bias           = T,
              kernel_initializer = initializer_constant(0),
              bias_initializer   = initializer_constant(0))
## Definimos a função de perda e o otimizador
mod1 %>%
  compile(optimizer = optimizer_sgd(lr = 0.1),
          loss      = "mse")
## Ajustamos a rede usando o Keras
inicio        <- Sys.time()
mod1_resultado <- mod1 %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = nrow(x_treino),
      epochs          = 100,
      callbacks       = list(callback_early_stopping(patience             = 100,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = F)
total_keras   <- Sys.time() - inicio
## Obtendo previsoes com essa rede
yhat1_treino <- mod1 %>% predict_on_batch(as.matrix(x_treino))
yhat1_teste  <- mod1 %>% predict_on_batch(as.matrix(x_teste))
## Calculo dos residuos de treino
res1_treino  <- c(y_treino - yhat1_treino)

### Procedimento 1: Assume erros i.i.d. ~ N(0, sigma^2)
sigma2hat  <- var(res1_treino)
## Treino
li1_treino <- as.vector(yhat1_treino) - qnorm(.975)*sqrt(sigma2hat)
ls1_treino <- as.vector(yhat1_treino) + qnorm(.975)*sqrt(sigma2hat)
## Teste
li1_teste  <- as.vector(yhat1_teste)  - qnorm(.975)*sqrt(sigma2hat)
ls1_teste  <- as.vector(yhat1_teste)  + qnorm(.975)*sqrt(sigma2hat)
## Cobertura media
# Treino
mean(li1_treino <= y_treino & y_treino <= ls1_treino) # 93,75%
# Teste
mean(li1_teste <= y_teste & y_teste <= ls1_teste) # 93,83%
## Metrica do artigo
# Treino
# Teste
## Grafico
ind1_cob_teste  <- as.numeric(li1_teste <= y_teste & y_teste <= ls1_teste)
dados_grid      <- x_teste
dados_grid$ind1 <- as.factor(ind1_cob_teste)
ggplot(dados_grid, aes(x1.obs, x2.obs)) +
  geom_point(aes(colour = ind1), size = 2, shape = 15) +
  coord_cartesian(expand = F) +
  scale_colour_manual(breaks = c(0, 1), values = c("#A11D21", "black"), name = TeX("\\hat{Y}(X_1, X_2)")) + 
  xlab(TeX("X_1")) + ylab(TeX("X_2"))

### Procedimento 2: Realiza bootstrap nao parametrico dos residuos
## Treino
set.seed(1)
boot_res  <- matrix(sample(res1_treino, 1000*nrow(treino), T), nrow(treino))
li1_treino <- numeric(nrow(treino))
ls1_treino <- numeric(nrow(treino))
for(i in 1:nrow(treino)) {
  boot_prev    <- c(boot_res[i,]) + yhat1_treino[i]
  li1_treino[i] <- quantile(boot_prev, .025)
  ls1_treino[i] <- quantile(boot_prev, .975)
}
## Teste
set.seed(1)
boot_res  <- matrix(sample(res1_treino, 1000*nrow(teste), T), nrow(teste))
li1_teste <- numeric(nrow(teste))
ls1_teste <- numeric(nrow(teste))
for(i in 1:nrow(teste)) {
  boot_prev    <- c(boot_res[i,]) + yhat1_teste[i]
  li1_teste[i] <- quantile(boot_prev, .025)
  ls1_teste[i] <- quantile(boot_prev, .975)
}
## Cobertura media
# Treino
mean(li1_treino <= y_treino & y_treino <= ls1_treino) # 94,80%
# Teste
mean(li1_teste <= y_teste & y_teste <= ls1_teste) # 94,93%
## Metrica do artigo
# Treino
# Teste
## Grafico
ind1_cob_teste  <- as.numeric(li1_teste <= y_teste & y_teste <= ls1_teste)
dados_grid      <- x_teste
dados_grid$ind1 <- as.factor(ind1_cob_teste)
ggplot(dados_grid, aes(x1.obs, x2.obs)) +
  geom_point(aes(colour = ind1), size = 2, shape = 15) +
  coord_cartesian(expand = F) +
  scale_colour_manual(breaks = c(0, 1), values = c("#A11D21", "black"), name = TeX("\\hat{Y}(X_1, X_2)")) + 
  xlab(TeX("X_1")) + ylab(TeX("X_2"))