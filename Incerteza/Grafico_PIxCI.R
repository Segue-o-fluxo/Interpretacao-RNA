set.seed(123)
x <- rnorm(100, 200, 20)
y <- x + rnorm(100, 0, 20)
library(ggplot2)
mod <- lm(y ~ x)
ci <- as.data.frame(cbind(x, predict(mod, interval = "confidence")))
pi <- as.data.frame(cbind(x, predict(mod, interval = "prediction")))
df <- data.frame(x, y)
ggplot(df, aes(x, y)) +
  geom_point(size = 2) +
  geom_line(data = ci, aes(y = fit), colour = "blue", size = 1.5) +
  geom_line(data = ci, aes(y = lwr), linetype = "longdash", size = .75) +
  geom_line(data = ci, aes(y = upr), linetype = "longdash", size = .75) +
  geom_ribbon(data = ci, aes(ymin = lwr, ymax = upr), linetype = 2, alpha = 0.4, fill = "gray70") +
  geom_line(data = pi, aes(y = lwr), linetype = "longdash", size = .75, colour = "red") +
  geom_line(data = pi, aes(y = upr), linetype = "longdash", size = .75, colour = "red") +
  theme_classic()
ggsave("Incerteza/PIxCI.png", width = 158, height = 93, units = "mm")