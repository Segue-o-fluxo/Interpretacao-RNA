########## Arquivo para ajustar as redes da Lista 2 - Alvaro e William ##########
library(dplyr)
library(keras)


########## Importacao dos Dados de Treino e Teste ##########
treino   <- readRDS("Incerteza/Treino_Lista1.rds")
teste    <- readRDS("Incerteza/Teste_Lista1.rds")
x_treino <- treino %>%
  select(x1.obs, x2.obs)
x_teste  <- teste %>%
  select(x1.obs, x2.obs)
y_treino <- treino$y
y_teste  <- teste$y


########## Rede Neural Ruim ##########
mod_ruim <- keras_model_sequential()
mod_ruim %>%
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
mod_ruim %>%
  compile(optimizer = optimizer_sgd(lr = 0.1),
          loss      = "mse")
mod_ruim %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = nrow(x_treino),
      epochs          = 100,
      callbacks       = list(callback_early_stopping(patience             = 100,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = F)
mod_ruim %>% save_model_tf("Incerteza/Rede Ruim")


########## Rede Neural Ruim com Dropout ##########
mod_ruim_d <- keras_model_sequential()
mod_ruim_d %>%
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
mod_ruim_d %>%
  compile(optimizer = optimizer_sgd(lr = 0.1),
          loss      = "mse")
mod_ruim_d %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = nrow(x_treino),
      epochs          = 100,
      callbacks       = list(callback_early_stopping(patience             = 100,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = F)
mod_ruim_d %>% save_model_tf("Incerteza/Rede Ruim Dropout")


########## Rede Neural Boa ##########
mod_bom <- keras_model_sequential()
mod_bom %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal(),
              input_shape = ncol(x_treino)) %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal()) %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal()) %>%
  layer_dense(units              = 1,
              activation         = "linear",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal())
mod_bom %>%
  compile(optimizer = optimizer_adam(lr = 0.01),
          loss      = "mse")
mod_bom %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = 50,
      epochs          = 50,
      callbacks       = list(callback_early_stopping(patience             = 10,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = T)
mod_bom %>% save_model_tf("Incerteza/Rede Boa")


########## Rede Neural Boa com Dropout ##########
mod_bom_d <- keras_model_sequential()
mod_bom_d %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal(),
              input_shape = ncol(x_treino)) %>%
  layer_dropout(.2) %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal()) %>%
  layer_dropout(.2) %>%
  layer_dense(units              = 100,
              activation         = "relu",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal()) %>%
  layer_dropout(.2) %>%
  layer_dense(units              = 1,
              activation         = "linear",
              use_bias           = T,
              kernel_initializer = initializer_random_normal(),
              bias_initializer   = initializer_random_normal())
mod_bom_d %>%
  compile(optimizer = optimizer_adam(lr = 0.01),
          loss      = "mse")
mod_bom_d %>%
  fit(x               = as.matrix(x_treino),
      y               = y_treino,
      batch_size      = 50,
      epochs          = 50,
      callbacks       = list(callback_early_stopping(patience             = 10,
                                                     restore_best_weights = T)),
      validation_data = list(x_val = as.matrix(x_teste), y_val = y_teste),
      shuffle         = T)
mod_bom_d %>% save_model_tf("Incerteza/Rede Boa Dropout")