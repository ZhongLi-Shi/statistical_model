# Ridge and Lasso model  
Ridge、Lasso与线性回归的差别仅在损失函数上有些差别  
Lasso添加了参数矩阵的L1范数正则项，Ridge添加了参数矩阵的L2范数正则项  
具体公式如下:  
### Lasso  
$L = \frac{1}{m}(\sum \limits_i^m(\hat{y_i} - y_i)^2) + \lambda \sum \limits_k|w_k|^2$
