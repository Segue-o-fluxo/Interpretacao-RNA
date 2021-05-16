library(dplyr)
library(keras)
library(ggplot2)
library(latex2exp)

source("rede_dropout.R")

### Importacao dos Dados de Treino e Teste
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

### Ajuste da Rede Neural Ruim
## Iniciamos um modelo sequencial

mod1_dropout = DropOut$new(x_treino, y_treino, x_teste, y_teste)

mod1_dropout$adiciona_camada(10, ativacao = "sigmoide",
                             p_dropout = .8)$
  adiciona_camada(8, ativacao = "sigmoide",
                  p_dropout = .8)$
  adiciona_camada(1)

mod1_dropout$treina_modelo(lr = 0.1, n_iter = numero_treinos, early_stop = 20)

mod1_dropout$pega_pesos()

indicador_dentro_intervalo = function(x, y, n_preds = 100) {
  preds_teste = do.call("cbind", lapply(seq_len(n_preds), function(i) mod1_dropout$fit(x)))
  
  limites = apply(preds_teste, 1, quantile, p = c(.025, .975))
  
  inds = limites[1,] <= y & y <= limites[2,]
  return(inds)
}

ind_teste = indicador_dentro_intervalo(x_teste, y_teste)
ind_treino = indicador_dentro_intervalo(x_treino, y_treino)

mean(ind_teste) #0.2762
mean(ind_treino) #0.2752


dados_grid = as_tibble(x_teste) %>%
  mutate(ind = ind_teste)
ggplot(dados_grid, aes(x1.obs, x2.obs)) +
  geom_point(aes(colour = ind), size = 2, shape = 15) +
  coord_cartesian(expand = F) +
  scale_colour_manual("Dentro do intervalo", values = c("#A11D21", "black")) + 
  xlab(TeX("X_1")) + ylab(TeX("X_2"))
