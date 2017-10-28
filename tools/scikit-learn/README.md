# scikit-learn
### 常用
* 划分验证集

'''python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

### 实现
* [常用算法调用(LR/RF/GBDT)](./useful.py)
* [logistic回归](./sklearn_LR.py)

### Choosing the right estimator

![Choosing the right estimator](./choose.png)






