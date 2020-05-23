### 资料
* [官网](http://xgboost.readthedocs.io/en/latest/)
* [Python API](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [安装教程](http://m.blog.csdn.net/huangdunxian/article/details/53432432)

### 应用
* [demo](./xgboost.ipynb)
* [多分类](./xgboost_multi.ipynb)


### 调参
Xgboost 的调参。通常认为对它性能影响较大的参数有：
* eta：每次迭代完成后更新权重时的步长。越小训练越慢。
* num_round：总共迭代的次数。
* subsample：训练每棵树时用来训练的数据占全部的比例。用于防止 Overfitting。
* colsample_bytree：训练每棵树时用来训练的特征的比例，类似 RandomForestClassifier 的 max_features。
* max_depth：每棵树的最大深度限制。与 Random Forest 不同，Gradient Boosting 如果不对深度加以限制，最终是会 Overfit 的。
* early_stopping_rounds：用于控制在 Out Of Sample 的验证集上连续多少个迭代的分数都没有提高后就提前终止训练。用于防止 Overfitting。

#### 一般的调参步骤是：
1. 将训练数据的一部分划出来作为验证集。
2. 先将 eta 设得比较高（比如 0.1），num_round 设为 300 ~ 500。
3. 用 Grid Search 对其他参数进行搜索
4. 逐步将 eta 降低，找到最佳值。
5.以验证集为 watchlist，用找到的最佳参数组合重新在训练集上训练。注意观察算法的输出，看每次迭代后在验证集上分数的变化情况，从而得到最佳的 early_stopping_rounds。

```
X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.3)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'eta': 0.05,
    'max_depth': 7,
    'seed': 2016,
    'silent': 0,
    'eval_metric': 'rmse'
}
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(df_test))
```
所有具有随机性的 Model 一般都会有一个 seed 或是 random_state 参数用于控制随机种子。得到一个好的 Model 后，在记录参数时务必也记录下这个值，从而能够在之后重现 Model。
