### 1. LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(train_x, train_y)
y_pre = lr.predict(val_x)


### 2.RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(train_x, train_y)

y_pre = rf.predict(val_x)
y_pre[y_pre>0.5] = 1
y_pre[y_pre<0.5] = 0


### 3.GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, max_depth=3)
gbdt.fit(train_x, train_y)

y_pre=gbdt.predict(val_x)
y_pre[y_pre>0.5] = 1
y_pre[y_pre<0.5] = 0
