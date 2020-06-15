##--------------- Regressão Linear 01 -----------------##

#-------------- Dataset

# Informações sobre casas em Boston(EUA)
# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxide concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town
# B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
# LSTAT: Percentage of lower status of the population
# MEDV: Median value of owner-occupied homes in $1000s


#---------- Objetivo
# Prever os valores dos preços das casas usando as variáveis disponíveis

# Carregando os pacotes
library(readr)
library(caret)
library(ggpubr)

#----------- Carregando o dataset
casas <- read_csv('HousingData.csv')
dim(casas)
View(casas)

#---- Pré-processamento e análise exploratória dos dados

# Verificando se há valores NA
sapply(casas, function(x) sum(is.na (x)))
casas <- na.omit(casas) # descartando os valores NA
dim(casas)

# Resumo estatístico das variáveis
summary(casas)

# Tabela de correlação
cor(casas)

# Gráfico de dispersão (scatterplot)
g1 <- ggscatter(casas, x = "RM", y = "MEDV",
          add = "reg.line", conf.int = TRUE,
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Média de Quartos por residência", ylab = "Valor")

g2 <- ggscatter(casas, x = "LSTAT", y = "MEDV",
                add = "reg.line", conf.int = TRUE,
                cor.coef = TRUE, cor.method = "pearson",
                xlab = "% de menor status (população)", ylab = "Valor")

ggarrange(g1,g2)

# Boxpplot das variáveis independentes (RM e LSTAT)
b1 <- ggboxplot(casas$RM, xlab="Média de Quartos")
b2 <- ggboxplot(casas$LSTAT, xlab="% de menor Status")
ggarrange(b1,b2)

boxplot.stats(casas$RM)$out
boxplot.stats(casas$LSTAT)$out

# Resumo estatístico da variável alvo (MEDV)
summary(casas$MEDV)

# Histograma e densidade da variável alvo (MEDV)
ht <- gghistogram(casas, x = "MEDV", bins=9, xlab ="Valor")
ds <- ggdensity(casas, x = "MEDV", add = "median", xlab = "Valor")

ggarrange(ht, ds)

#------- Criando os datasets de treino e teste

# Definindo amostras 
set.seed(50)
amostra <- sample(2, nrow(casas), replace=TRUE, prob=c(0.7,0.3))
amostra
treino <- casas[amostra==1,]
teste <- casas[amostra==2,]

dim(treino)
dim(teste)
  
# -------- Criando o modelo de regressão linear simples

# Equação de Regressão
# y = a + bx (simples)

# Treinando o modelo (dados de treino)
modelo_s <- lm(MEDV ~ RM, data = treino)
modelo_s

# Resumo do modelo (métricas)
summary(modelo_s)

# Atributos do objeto modelo_s
attributes(modelo_s)
modelo_s$coefficients

# Analisando os valores atuais e valores previstos
resultado <- data.frame(Valor_atual=treino$MEDV, Valor_previsto=predict(modelo_s))
head(resultado)

cor(resultado)

# Mean absolute percentage error (MAPE) 
mape <- mean(abs(modelo_s$residuals)/resultado$Valor_atual)*100
mape

# Gráfico - valores previstos e resíduos 
plot(resultado$Valor_previsto, modelo_s$residuals,pch=21,bg="red",col="red")
abline(0,0)

# -------- Criando o modelo de regressão linear múltipla

# Equação de Regressão
# y = a + b0x0 + b1x1 (múltipla)

# Treinando o modelo (dados de treino) 
modelo_m <- lm(MEDV ~ ., data = treino)
modelo_m

summary(modelo_m)

# Comparando os valores atuais e valores previstos
resultado <- data.frame(Valor_atual=treino$MEDV, Valor_previsto=predict(modelo_m))
head(resultado)

cor(resultado)

# Mean absolute percentage error (MAPE) 
mape <- mean(abs(modelo_m$residuals)/resultado$Valor_atual)*100
mape

# ------------- Melhorando a performance do modelo

# Eliminando as variáveis que não são relevantes para o modelo
modelo_v2 <- lm(MEDV ~ .-INDUS-AGE, data = treino)
summary(modelo_v2)

# Aplicando uma transformação (log transformation) na variável alvo (MEDV)
modelo_v3 <- lm(log(MEDV) ~ .-INDUS-AGE, data = treino)
summary(modelo_v3)

# ---- verificandp a não-linearidade do modelo

# Exemplo
ggplot(treino, aes(MEDV, LSTAT)) +
  geom_point() +
  geom_smooth(method = "lm")

ggplot(treino, aes(MEDV, LSTAT)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2))


# Criando uma variável quadrática 
treino$RM2 <- treino$RM ^ 2 

modelo_v4 <- lm(log(MEDV) ~ .-INDUS-AGE, data = treino)
summary(modelo_v4)

# Comparando os valores atuais e valores previstos
resultado <- data.frame(Valor_atual=treino$MEDV, Valor_previsto=exp(predict(modelo_v4)))
head(resultado)

cor(resultado)

# Mean absolute percentage error (MAPE) 
mape <- mean(abs(resultado$Valor_atual-resultado$Valor_previsto)/resultado$Valor_atual)*100
mape

#----------------- Analisando a importância das variáveis independentes
importancia <- varImp(modelo_v4, varImp.train=TRUE)
print(importancia)

importancia <- data.frame(var_ind = row.names(importancia), importancia)
ggbarplot(importancia, x='var_ind', y="Overall", sort.val="asc", 
          orientation='horiz', fill='blue')


#-------------- Testando o modelo

# Acrescentando a variável RM2 no conjunto teste
teste$RM2 <- teste$RM ^ 2

# Fazendo as previsões usando o modelo_v4
previsao <- predict(modelo_v4, teste)
View(previsao)

# Mean absolute percentage error (MAPE) 
mape <- mean(abs(teste$MEDV-exp(previsao))/teste$MEDV)*100
mape

# Root mean square error (RMSE)
rmse <- sqrt(sum((exp(previsao)-teste$MEDV)^2)/length(teste$MEDV))
rmse

# Visualizando as diferenças entre valores atuais e previstos
plot(teste$MEDV,type="l",col="red")
lines(exp(previsao),col="blue")
legend("topleft", 
       legend = c("Valores Atuais", "Valores Previstos"), 
       col = c('red', 'blue'), 
       pch = c(19,19), 
       bty = "n", 
       inset = c(0,0))

#------- Salvando o modelo
saveRDS(modelo_v4,"modelo_regressao.rds")

#------ Carregando o modelo
regr <- readRDS("modelo_regressao.rds")
previsao_final <- predict(regr, teste)

