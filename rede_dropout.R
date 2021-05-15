library(R6)
library(tidyverse)

NeuralNetwork = R6Class("NeuralNetwork", public = list(
  X = NULL,
  Y = NULL,
  weights = NULL, # Pesos
  biases = NULL, # Vieses
  funcao = NULL, # vetor de ativacoes
  len_out = NULL, # Controla os inputs entre as camadas
  p_dropout = NULL,
  
  initialize = function(X, Y) {
    self$X = X
    self$Y = Y
    self$len_out = ncol(X)
  },
  
  adiciona_camada = function(units = 1, pesos_inicias = NULL, 
                             bias_inicial = NULL, ativacao = "linear",
                             p_dropout = NULL) {
    # Adiciona camada a rede neural
    # Pode-se adicionar multiplas funcoes
    # E é possível utilizar diferentes funções de ativação
    # Nota: É necessário alterar a função aplica_funcao
    # para adicionar novas funcoes
    dim_inp = self$len_out[length(self$len_out)]
    dim_w = units*dim_inp
    dim_bias = units
    
    if (is.null(pesos_inicias)) {
      pesos_inicias = runif(dim_w)
    } else if (length(pesos_inicias) != (dim_w)) {
      pesos_inicias = rep_len(pesos_inicias, dim_w)
    }
    
    if (is.null(bias_inicial)) {
      bias_inicial = runif(dim_bias)
    } else if (length(bias_inicial) != (dim_bias)) {
      bias_inicial = rep_len(bias_inicial, dim_bias)
    }
    
    if(is.null(p_dropout) || !is.numeric(p_dropout)) {p_dropout = 1}
    
    pesos_matricial = matrix(pesos_inicias, dim_inp, units)
    # Update
    self$weights = append(self$weights, list(pesos_matricial))
    self$biases = append(self$biases, list(bias_inicial))
    self$p_dropout = c(self$p_dropout, p_dropout)
    self$funcao = append(self$funcao, ativacao)
    self$len_out = c(self$len_out, units)
    
    return(invisible(self))
  },
  walk_neuron = function(input, weights, biases, ativacao, p_dropout) {
    # Predicao em um neuronio
    input = input * rbinom(length(input), 1, p_dropout)
    a = input %*% weights + biases
    output = self$aplica_funcao(a, ativacao)
    return(output)
  },
  fit = function(data = self$X) {
    # Ajuste da rede (feed_foward basicamente)
    h = data
    for(k in seq_along(self$weights)) {
      h = self$walk_neuron(h, self$weights[[k]], self$biases[[k]], self$funcao[[k]], self$p_dropout[[k]])
    }
    return(h)
  },
  sigmoide = function(x) {return(1 / (1 + exp(-x)))},
  
  del_sigmoide = function(x) {fx = self$sigmoide(x); return(fx * (1. - fx))},
  
  del_linear = function(x) {
    y = x
    y[] = 1
    return(y)
  },
  
  aplica_funcao = function(x, funcao, derivada = FALSE) {
    # Aplica a funcao de ativacao especificada
    ff = switch(funcao,
                "sigmoide" = if(derivada) self$del_sigmoide(x) else self$sigmoide(x),
                if(derivada) self$del_linear(x) else x # Padrao linear
    )
    return(ff)
    
  },
  pega_pesos = function() {
    cat("-------------------Pesos-----------------------------\n")
    print(self$weights)
    cat("------------------Vieses-----------------------------\n")
    print(self$biases)
    cat("------------------Ativacao-----------------------------\n")
    print(self$funcao)
  }
))

DropOut = R6Class("DropOut",
                        inherit = NeuralNetwork,
                        public = list(
                          output = NULL,
                          x_teste = NULL,
                          y_teste = NULL,
                          p_dropout = NULL,
                          
                          initialize = function(X_treino, Y_treino, X_teste=NULL, Y_teste=NULL) {
                            super$initialize(X_treino, Y_treino)
                            
                            if(is.null(X_teste) || is.null(Y_teste)) {
                              tamanho_treino = nrow(self$X)
                              treino_idx = sample.int(tamanho_treino, trunc(tamanho_treino*.8))
                              X_teste = self$X[-treino_idx,]
                              Y_teste = self$Y[-treino_idx]
                              self$X = self$X[treino_idx,]
                              self$Y = self$Y[treino_idx]
                            }
                            self$x_teste = X_teste
                            self$y_teste = Y_teste
                          },
                          
                          
                        
                          
                          feedfoward = function(data = self$X) {
                            self$output <- self$fit(data)
                            return(invisible(self))
                          },
                          
                          backpropagate = function(x = self$X, y = self$Y) {
                            nabla_w <- lapply(self$weights, function(x) array(0, dim(x)))
                            nabla_b <- lapply(self$biases, function(x) rep(0, length(x)))
                            
                            # feedfoward
                            h = x
                            u = rbinom(length(h), 1, self$p_dropout[[1]])
                            h = h*u # Dropout
                            as = NULL
                            hs = list(h) # Armazena os inputs
                            
                            n_camadas = length(self$weights)
                            for(k in seq_len(n_camadas)) {
                              
                              a = h %*% self$weights[[k]] + self$biases[[k]]
                              as = append(as, list(a))
                              h = self$aplica_funcao(a, self$funcao[[k]])
                              u = rbinom(length(h), 1, self$p_dropout[[k]])
                              h = h*u
                              hs = append(hs, list(h))
                            }
                            
                            # Backward
                            delta = self$del_square_loss(hs[[k+1]], y) *
                              self$aplica_funcao(as[[k]], self$funcao[[k]], derivada = TRUE)
                            nabla_b[[k]] = colSums(delta)
                            nabla_w[[k]] = t(hs[[k]]) %*% delta
                            
                            for(k in rev(seq_len(n_camadas-1))) {
                              flinha = self$aplica_funcao(as[[k]], self$funcao[[k]], derivada = TRUE)
                              delta = (delta %*% t(self$weights[[k+1]])) * flinha
                              nabla_b[[k]] = colSums(delta)
                              nabla_w[[k]] = t(hs[[k]]) %*% delta
                            }
                            return(list(nabla_b = nabla_b, nabla_w = nabla_w))
                          },
                          gradient_descent = function(lr=.1) {
                            grad <- self$backpropagate()
                            n_camadas = length(self$weights)
                            for(k in seq_len(n_camadas)) {
                              self$biases[[k]] = self$biases[[k]] - lr * grad$nabla_b[[k]]
                              self$weights[[k]] = self$weights[[k]] - lr * grad$nabla_w[[k]]
                            }
                          },
                          treina_modelo = function(lr=.1, n_iter=100, early_stop = Inf) {
                            best_loss = Inf
                            best_weights = NULL
                            int_sem_melhora =  0
                            for(i in seq_len(n_iter)) {
                              
                              self$gradient_descent(lr)
                              pred_teste = self$fit(self$x_teste)
                              loss = self$square_loss(pred_teste, self$y_teste)
                              cat("Epoch: ", i, "Loss: ", loss,"\n")
                              if(loss < best_loss) {
                                int_sem_melhora =  0
                                best_loss = loss
                                best_weights = list(self$weights, self$biases)
                              } else {
                                int_sem_melhora = int_sem_melhora + 1
                                if(int_sem_melhora >=early_stop){
                                  break
                                }
                              }
                            }
                            # restaura os melhores pesos
                            self$weights = best_weights[[1]]
                            self$biases = best_weights[[2]]
                          },
                          
                          
                          square_loss = function(yhat, y) {
                            return(mean((y - yhat)^2))
                          },
                          
                          
                          del_square_loss = function(yhat, y) {
                            return(-2*(y - yhat)/length(y))
                          }
                        ))

# 
# ff = DropOut$new(treino_x, treino_y, teste_x, teste_y)
# ff$adiciona_camada(2, 0, 0, ativacao = "sigmoide", p_dropout = .8)$
#   adiciona_camada(1, 0, 0, p_dropout = .8)
# ff$treina_modelo()
# ff$pega_pesos()
