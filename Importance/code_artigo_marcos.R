pacman::p_load("neuralnet", 
               "tidyverse", 
               "latex2exp", 
               "knitr", 
               "NeuralNetTools", 
               "Cairo", 
               "hexbin", 
               "tictoc", 
               "microbenchmark", 
               "keras", 
               "cowplot")

# M?todo de retirada de covari?vel do artigo ----
cols = c("word_freq_make",         
  "word_freq_address",
  "word_freq_all",          
  "word_freq_3d",           
  "word_freq_our",          
  "word_freq_over",         
  "word_freq_remove",       
  "word_freq_internet",     
  "word_freq_order",        
  "word_freq_mail",         
  "word_freq_receive",      
  "word_freq_will",         
  "word_freq_people",       
  "word_freq_report",       
  "word_freq_addresses",    
  "word_freq_free",         
  "word_freq_business",
  "word_freq_email",        
  "word_freq_you",          
  "word_freq_credit",       
  "word_freq_your",         
  "word_freq_font",         
  "word_freq_000",          
  "word_freq_money",        
  "word_freq_hp",           
  "word_freq_hpl",          
  "word_freq_george",       
  "word_freq_650",          
  "word_freq_lab",          
  "word_freq_labs",         
  "word_freq_telnet",       
  "word_freq_857",          
  "word_freq_data",         
  "word_freq_415",          
  "word_freq_85",           
  "word_freq_technology",   
  "word_freq_1999",         
  "word_freq_parts",        
  "word_freq_pm",           
  "word_freq_direct",       
  "word_freq_cs",           
  "word_freq_meeting",      
  "word_freq_original",     
  "word_freq_project",      
  "word_freq_re",           
  "word_freq_edu",          
  "word_freq_table",        
  "word_freq_conference",   
  "char_freq_comma",            
  "char_freq_parentheses",            
  "char_freq_key",            
  "char_freq_exclamation",
  "char_freq_dollar",            
  "char_freq_hashtag",            
  "capital_run_length_average",
  "capital_run_length_longest", 
  "capital_run_length_total",
  "spam")
df_spam = read.csv("Importance/spambase.data", header = FALSE)
colnames(df_spam) = cols

metrics = function(x){
  if (ncol(x) == 2 & nrow(x) == 2){
    acc_m = sum(diag(x)) / sum(x) # accuracy
    #tpr = x[2,2] / sum(x[2,]) # true positive rate or recall
    #tnr = x[1,1] / sum(x[1,]) # true negative rate
    #precision = x[2,2] / sum(x[,2]) # precision
    #prevalence = sum(x[2,]) / sum(x) # prevalence
    fnr = x[2,1] / sum(x[2,])
    fpr = x[1,2] / sum(x[1,])
    #F1_measure = 2*(tpr*precision)/(tpr + precision)
    #F0.5_measure = (1.25*precision*tpr) / (0.25*precision + tpr)
    #F2_measure = (5*precision*tpr)/(4*precision + tpr)
    dt = data.frame(acc_m, fnr, fpr)
    return(dt)
  }
}

set.seed(47)
df_spam = df_spam %>% 
  mutate(spam = as.factor(spam))
summary(df_spam[,58])
rows = sample(nrow(df_spam)) # shuffle das linhas
df_spam = df_spam[rows, ]

n = dim(df_spam)[1]
ind = sample(1:n, size = round(n*.70))


df_treino = df_spam[ind,] 
df_teste = df_spam[-ind,]

df_treino_scale = df_treino %>% select(-c("spam")) %>% scale() %>% data.frame()
df_teste_scale = df_teste %>% select(-c("spam")) %>% scale() %>% data.frame()

y = to_categorical(df_treino$spam)
y_val = to_categorical(df_teste$spam) 

pass = TRUE
df_report = data.frame()
a = Sys.time()
covariaveis = c("None", colnames(df_spam[, -58]))
for (name in covariaveis){
  
  if (pass){
    x = as.matrix(df_treino_scale)
    dimnames(x) = NULL
    x_val = as.matrix(df_teste_scale)
    dimnames(x_val) = NULL
    pass = FALSE
  }
  
  else{
    # defini??o dos conjuntos de dados, com exclus?o de covari?vel
    x = as.matrix(df_treino_scale %>% select(-c(name)))
    dimnames(x) = NULL
    x_val = as.matrix(df_teste_scale %>% select(-c(name)))
    dimnames(x_val) = NULL
    }
  
  # defini??o da rede neural
  mod1 <- keras_model_sequential()
  mod1 %>%
    layer_dense(units = 3,
                input_shape = ncol(x),
                activation="relu") %>% 
    layer_dense(units = 2,
                activation = 'sigmoid') %>% 
    compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics = 'accuracy')
  parada <- callback_early_stopping(patience = 20, 
                                    restore_best_weights = TRUE)
  ajuste = mod1 %>%
    fit(x,
        y,
        batch_size = 64,
        epochs = 50,
        validation_data = list(x_val, y_val),
        callbacks = list(parada))
  
  # c?lculo de m?tricas de avalia??o
  target_hat = mod1 %>% 
    predict(x_val) %>% 
    round() %>% 
    .[, 2]
  target = y_val[,2]  
  conf_mat = table(target, target_hat)
  df_report = bind_rows(df_report,
                        data.frame(exclusion = name, metrics(conf_mat)))
}
b = Sys.time(); b-a
df_report

# Rank das covari?veis
comparison = df_report[1,]
### no artigo n?o aparece o seguinte caso: OA < , FP > , FN <
teste = df_report %>% 
  slice(2:58) %>% 
  mutate(rank = case_when(acc_m > comparison$acc_m &
                          fpr < comparison$fpr &
                          fnr < comparison$fnr ~ "unimportant",
                          acc_m > comparison$acc_m &
                            fpr > comparison$fpr &
                            fnr < comparison$fnr ~ "unimportant",
                          acc_m < comparison$acc_m &
                            fpr > comparison$fpr &
                            fnr > comparison$fnr ~ "important",
                          round(acc_m,2) == round(comparison$acc_m,2) &
                            round(fpr,2) == round(comparison$fnr,2) ~ "secondary",
                          TRUE ~ "unimportant"))

important_vars = teste %>% 
  filter(rank == 'important') %>% 
  select(exclusion) %>% 
  .[[1]]

# defini??o dos conjuntos de dados, com exclus?o de covari?vel
x = as.matrix(df_treino_scale %>% select(c(important_vars)))
dimnames(x) = NULL
x_val = as.matrix(df_teste_scale %>% select(c(important_vars)))
dimnames(x_val) = NULL


# defini??o da rede neural
mod1 <- keras_model_sequential()
mod1 %>%
  layer_dense(units = 3,
              input_shape = ncol(x),
              activation="relu") %>% 
  layer_dense(units = 2,
              activation = 'sigmoid') %>% 
  compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics = 'accuracy')
parada <- callback_early_stopping(patience = 20, 
                                  restore_best_weights = TRUE)
ajuste = mod1 %>%
  fit(x,
      y,
      batch_size = 64,
      epochs = 50,
      validation_data = list(x_val, y_val),
      callbacks = list(parada))

# c?lculo de m?tricas de avalia??o
target_hat = mod1 %>% 
  predict(x_val) %>% 
  round() %>% 
  .[, 2]
target = y_val[,2]  
conf_mat = table(target, target_hat)
metrics(conf_mat)
df_report = bind_rows(df_report,
                      data.frame(exclusion = "important_vars", metrics(conf_mat)))