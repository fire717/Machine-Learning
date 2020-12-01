# code from https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/
# 自己加了点注释帮助理解，也方便自己以后使用
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds            #交叉验证集划分的折数
        self.stacker = stacker            #第二层stacking时使用的分类器
        self.base_models = base_models    #第一层的基本模型 们
    def fit_predict(self, X, y, T):
        X = np.array(X)   #train_x
        y = np.array(y)   #train_y
        T = np.array(T)   #test_x
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        #sklearn.cross_validation.KFold(n, n_folds=3, shuffle=False, random_state=None)
        #http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html
        #这里只是生成了index 的迭代器，根据index取数据在后面进行
        
        S_train = np.zeros((X.shape[0], len(self.base_models))) #第二层的训练数据
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        #数据条数不变，特征数变为模型数，因为每个模型产生一列
        
        for i, clf in enumerate(self.base_models):    #clf  Classification  
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]  #整个T的预测
            S_test[:, i] = S_test_i.mean(1)    #按行求平均值 即axis=1. 矩阵变成一列后加入S_test中
            
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
        
 '''
 据说获奖选手往往会使用比这复杂得多的 Ensemble，会出现三层、四层甚至五层，不同的层数之间有各种交互，
 还有将经过不同的 Preprocessing 和不同的 Feature Engineering 的数据用 Ensemble 组合起来的做法。
 '''
