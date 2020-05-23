### 1. LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(train_x, train_y)
y_pre = lr.predict(val_x)


### 2.RF
#Random Forest 一般在 max_features 设为 Feature 数量的平方根附近得到最佳结果。
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

### 4.knn
from sklearn import neighbors 

knn = neighbors.KNeighborsClassifier(n_neighbors=8,leaf_size=30,p=3)
knn.fit(x,y)  


### 5.svm
#http://blog.csdn.net/u013709270/article/details/53365744 (d多分类)
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  
clf.predict([[2., 2.]])
