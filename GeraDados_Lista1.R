### Arquivo para gerar os dados da Lista 1 - Alvaro e William
library(dplyr)
set.seed(2.2020)
n.obs <- 100000
dados <- tibble(x1.obs = runif(n.obs, -3, 3), 
                x2.obs = runif(n.obs, -3, 3)) %>%
  mutate(mu = abs(x1.obs^3 - 30*sin(x2.obs) + 10),
         y  = rnorm(n.obs, mu, 1))
saveRDS(dados, "C:/Users/willi/Documents/UNB/8º Semestre/Tópicos em Estatística 1 - Redes Neurais/Seminários/Trabalho/Interpretacao-RNA/Dados_Lista1.rds")