# Logistical Regression
线性回归公式为: $f(x) = w^{T}x+b$  
通过激活函数$sigmoid()$将输出映射至[0, 1]内,这里认为大于等于0.5的为标签1，小于0.5的为标签0  
  
损失函数: L = \sum_{i}y_iln\hat{y_i} + (1-y_i)ln(1-hat(y_i))
