########## William - Previsoes Intervalares ##########
library(dplyr)
library(keras)
library(ggplot2)
library(latex2exp)
library(DT)


########## Importando os Dados ##########
treino   <- readRDS("Incerteza/Treino_Lista1.rds")
teste    <- readRDS("Incerteza/Teste_Lista1.rds")
x_treino <- treino %>%
  select(x1.obs, x2.obs)
x_teste  <- teste %>%
  select(x1.obs, x2.obs)
y_treino <- treino$y
y_teste  <- teste$y


########## Importando as Redes ##########
mod_ruim   <- load_model_tf("Incerteza/Rede Ruim")
mod_bom    <- load_model_tf("Incerteza/Rede Boa")
mod_ruim_d <- load_model_tf("Incerteza/Rede Ruim Dropout")
mod_bom_d  <- load_model_tf("Incerteza/Rede Boa Dropout")


########## Importando as Metricas ##########
source("Incerteza/Metricas.R")


########## Dataframe para comparar resultados ##########
resultados <- NULL


########## Procedimento 1: Assume erros i.i.d. ~ N(0, sigma^2) ##########
# Predicoes pontuais
yhat_treino_ruim <- mod_ruim %>% predict_on_batch(as.matrix(x_treino)) %>% as.vector()
yhat_treino_bom  <- mod_bom  %>% predict_on_batch(as.matrix(x_treino)) %>% as.vector()
yhat_teste_ruim  <- mod_ruim %>% predict_on_batch(as.matrix(x_teste))  %>% as.vector()
yhat_teste_bom   <- mod_bom  %>% predict_on_batch(as.matrix(x_teste))  %>% as.vector()
# Residuos de treino
res_treino_ruim  <- y_treino - yhat_treino_ruim
res_treino_bom   <- y_treino - yhat_treino_bom
# Estimativa da variancia dos erros
sigma2hat_ruim <- var(res_treino_ruim)
sigma2hat_bom  <- var(res_treino_bom)
# Limites inferiores
linf_teste_ruim  <- yhat_teste_ruim  - qnorm(0.975)*sqrt(sigma2hat_ruim)
linf_teste_bom   <- yhat_teste_bom   - qnorm(0.975)*sqrt(sigma2hat_bom)
# Limites superiores
lsup_teste_ruim  <- yhat_teste_ruim  + qnorm(0.975)*sqrt(sigma2hat_ruim)
lsup_teste_bom   <- yhat_teste_bom   + qnorm(0.975)*sqrt(sigma2hat_bom)
# Cobertura media
(cob_teste_ruim  <- cob_media(ind_cob(y_teste,  linf_teste_ruim,  lsup_teste_ruim)))
(cob_teste_bom   <- cob_media(ind_cob(y_teste,  linf_teste_bom,   lsup_teste_bom)))
# CWC
R_teste          <- max(y_teste)  - min(y_teste)
(cwc_teste_ruim  <- cwc(amp_media_norm(linf_teste_ruim,  lsup_teste_ruim,  R_teste),  cob_teste_ruim))
(cwc_teste_bom   <- cwc(amp_media_norm(linf_teste_bom,   lsup_teste_bom,   R_teste),  cob_teste_bom))
# Funcao grafico da cobertura
grafico_cob <- function(ruim) {
  ind <- if (ruim) "ind_ruim" else "ind_bom"
  ggplot(dados_grid, aes(x1.obs, x2.obs)) +
    geom_point(aes_string(colour = ind), size = 3) +
    coord_cartesian(expand = F) +
    scale_colour_manual(breaks = c(0, 1), values = c("#A11D21", "Black"), name = "", labels = c("Fora", "Dentro")) + 
    xlab(TeX("X_1")) + ylab(TeX("X_2"))
}
# Grafico da cobertura
dados_grid          <- x_teste
dados_grid$ind_ruim <- as.factor(as.numeric(ind_cob(y_teste, linf_teste_ruim, lsup_teste_ruim)))
dados_grid$ind_bom  <- as.factor(as.numeric(ind_cob(y_teste, linf_teste_bom,  lsup_teste_bom)))
grafico_cob(ruim = T)
grafico_cob(ruim = F)
# Adicionando resultados
resultados <- rbind(resultados, data.frame(Metodo    = rep("Normal", 2), 
                                           Rede      = c("Ruim", "Boa"),
                                           Cobertura = c(cob_teste_ruim, cob_teste_bom),
                                           CWC       = c(cwc_teste_ruim, cwc_teste_bom)))


########## Procedimento 2: Realiza bootstrap nao parametrico dos residuos ##########
# Bootstrap dos residuos
set.seed(1)
boot_res_teste_ruim  <- matrix(sample(res_treino_ruim, 1000*nrow(teste), T),  nrow(teste))
boot_res_teste_bom   <- matrix(sample(res_treino_bom,  1000*nrow(teste), T),  nrow(teste))
# Bootstrap dos valores previstos
boot_prev_teste_ruim  <- boot_res_teste_ruim  + yhat_teste_ruim
boot_prev_teste_bom   <- boot_res_teste_bom   + yhat_teste_bom
# Limites de teste
linf_teste_ruim <- numeric(nrow(teste))
linf_teste_bom  <- numeric(nrow(teste))
lsup_teste_ruim <- numeric(nrow(teste))
lsup_teste_bom  <- numeric(nrow(teste))
for(i in 1:nrow(teste)) {
  quantis_teste_ruim   <- quantile(boot_prev_teste_ruim[i,], c(0.025, 0.975))
  quantis_teste_bom    <- quantile(boot_prev_teste_bom[i,],  c(0.025, 0.975))
  linf_teste_ruim[i]   <- quantis_teste_ruim[1]
  linf_teste_bom[i]    <- quantis_teste_bom[1]
  lsup_teste_ruim[i]   <- quantis_teste_ruim[2]
  lsup_teste_bom[i]    <- quantis_teste_bom[2]
}
# Cobertura media
(cob_teste_ruim  <- cob_media(ind_cob(y_teste,  linf_teste_ruim,  lsup_teste_ruim)))
(cob_teste_bom   <- cob_media(ind_cob(y_teste,  linf_teste_bom,   lsup_teste_bom)))
# CWC
(cwc_teste_ruim  <- cwc(amp_media_norm(linf_teste_ruim,  lsup_teste_ruim,  R_teste),  cob_teste_ruim))
(cwc_teste_bom   <- cwc(amp_media_norm(linf_teste_bom,   lsup_teste_bom,   R_teste),  cob_teste_bom))
# Grafico da cobertura
dados_grid          <- x_teste
dados_grid$ind_ruim <- as.factor(as.numeric(ind_cob(y_teste, linf_teste_ruim, lsup_teste_ruim)))
dados_grid$ind_bom  <- as.factor(as.numeric(ind_cob(y_teste, linf_teste_bom,  lsup_teste_bom)))
grafico_cob(ruim = T)
grafico_cob(ruim = F)
# Adicionando resultados
resultados <- rbind(resultados, data.frame(Metodo    = rep("Bootstrap NP", 2),
                                           Rede      = c("Ruim", "Boa"),
                                           Cobertura = c(cob_teste_ruim, cob_teste_bom),
                                           CWC       = c(cwc_teste_ruim, cwc_teste_bom)))


########## Procedimento 3: Dropout ##########
# Dropout
predicao_dropout <- function(modelo, matriz_pred) {
  out <- as.matrix(modelo(matriz_pred, training = T))
  return(out)
}
# Limites
limites <- function(modelo, x, n_preds = 100) {
  preds   <- do.call("cbind", lapply(seq_len(n_preds), function(i) predicao_dropout(modelo, x)))
  limites <- apply(preds, 1, quantile, p = c(.025, .975))
  return(limites)
}
limites_ruim <- limites(mod_ruim_d, as.matrix(x_teste))
limites_bom  <- limites(mod_bom_d, as.matrix(x_teste))
# Cobertura media
(cob_teste_ruim  <- cob_media(ind_cob(y_teste,  limites_ruim[1,],  limites_ruim[2,])))
(cob_teste_bom   <- cob_media(ind_cob(y_teste,  limites_bom[1,],   limites_bom[2,])))
# CWC
(cwc_teste_ruim  <- cwc(amp_media_norm(limites_ruim[1,],  limites_ruim[2,],  R_teste),  cob_teste_ruim))
(cwc_teste_bom   <- cwc(amp_media_norm(limites_bom[1,],   limites_bom[2,],   R_teste),  cob_teste_bom))
# Grafico da cobertura
dados_grid          <- x_teste
dados_grid$ind_ruim <- as.factor(as.numeric(ind_cob(y_teste, limites_ruim[1,], limites_ruim[2,])))
dados_grid$ind_bom  <- as.factor(as.numeric(ind_cob(y_teste, limites_bom[1,],  limites_bom[2,])))
grafico_cob(ruim = T)
grafico_cob(ruim = F)
# Adicionando resultados
resultados <- rbind(resultados, data.frame(Metodo    = rep("Dropout", 2),
                                           Rede      = c("Ruim", "Boa"),
                                           Cobertura = c(cob_teste_ruim, cob_teste_bom),
                                           CWC       = c(cwc_teste_ruim, cwc_teste_bom)))


########## Apresentação dos Resultados ##########
resultados %>%
  mutate(across(where(is.numeric), round, digits = 3)) %>%
  datatable()