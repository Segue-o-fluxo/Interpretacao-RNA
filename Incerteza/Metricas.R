########## Arquivo com as funcoes das metricas - Alvaro e William ##########
ind_cob <- function(ytrue, linf, lsup) {
  ytrue <- as.vector(ytrue); linf <- as.vector(linf); lsup <- as.vector(lsup)
  return(linf <= ytrue & ytrue <= lsup)
}
cob_media <- function(ind_cob) {
  return(mean(ind_cob))
}
amp_media_norm <- function(linf, lsup, R) {
  amp_media <- mean(lsup - linf)
  return(amp_media/R)
}
cwc <- function(amp_media_norm, cob_media, eta = 50, mu = 0.95) {
  return(amp_media_norm * (1 + as.numeric(cob_media < mu) * exp(- eta * (cob_media - mu))))
}