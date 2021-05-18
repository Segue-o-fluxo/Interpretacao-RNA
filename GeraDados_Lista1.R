########## Arquivo para gerar os dados da Lista 1 - Alvaro e William ##########

library(dplyr)
set.seed(2.2020)
n.obs <- 100000
dados <- tibble(x1.obs = runif(n.obs, -3, 3), 
                x2.obs = runif(n.obs, -3, 3)) %>%
  mutate(mu = abs(x1.obs^3 - 30*sin(x2.obs) + 10),
         y  = rnorm(n.obs, mu, 1))
saveRDS(dados, "Dados_Lista1.rds")


########## Divisão em Treino e Teste ##########
corte  <- 80000
treino <- dados[1:corte,]
teste  <- dados[(corte+1):nrow(dados),]
saveRDS(treino, "Treino_Lista1.rds")
saveRDS(teste, "Teste_Lista1.rds")