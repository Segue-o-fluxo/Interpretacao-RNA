library(keras)
library(tidyverse)
library(latex2exp)

mod1 <- keras_model_sequential()

treino   <- readRDS("../Treino_Lista1.rds")
teste    <- readRDS("../Teste_Lista1.rds")
x_treino <- treino %>%
  select(x1.obs, x2.obs) %>% as.matrix()
x_teste  <- teste %>%
  select(x1.obs, x2.obs) %>% as.matrix()
y_treino <- treino$y
y_teste  <- teste$y

numero_predicoes = 100

numero_treinos = 100

## Adicionamos as camadas
mod1 %>%
  layer_dense(units              = 2,
              activation         = "sigmoid",
              use_bias           = T,
              kernel_initializer = initializer_constant(0),
              bias_initializer   = initializer_constant(0),
              input_shape        = ncol(x_treino)) %>%
  layer_dropout(.2) %>%
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
mod1_resultado <- mod1 %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = nrow(x_treino),
      epochs          = 100,
      callbacks       = list(callback_early_stopping(patience             = 100,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = F)

predicao_dropout = function(modelo, matriz_pred) {
  out = as.matrix(modelo(matriz_pred, training = T))
  return(out)
}

indicador_dentro_intervalo = function(modelo, x, y, n_preds = 100) {
  preds = do.call("cbind", lapply(seq_len(n_preds), function(i) predicao_dropout(modelo, x)))
  
  limites = apply(preds, 1, quantile, p = c(.025, .975))
  
  inds = limites[1,] <= y & y <= limites[2,]
  return(inds)
}

ind_teste = indicador_dentro_intervalo(mod1, x_teste, y_teste)
mean(ind_teste) #0.23905


dados_grid = as_tibble(x_teste) %>%
  mutate(ind = ind_teste)
ggplot(dados_grid, aes(x1.obs, x2.obs)) +
  geom_point(aes(colour = ind), size = 2, shape = 15) +
  coord_cartesian(expand = F) +
  scale_colour_manual("Dentro do intervalo", values = c("#A11D21", "black")) + 
  xlab(TeX("X_1")) + ylab(TeX("X_2"))

