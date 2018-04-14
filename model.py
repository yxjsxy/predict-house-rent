import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder, Binarizer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from mlxtend.classifier import StackingClassifier
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



data = pd.read_csv('/Users/xiujiayang/Downloads/train.csv')


'''
# Data Analysis
 
data.describe()
data.head(10)
data.info()
data.hist(bins=50, figsize=(20,15))
plt.show()
data['Accept'].value_counts()
data.plot(kind='scatter', x='Review', y='Price')
data.groupby('Beds')['Price'].mean()
corr_matrix = data.corr()
corr_matrix['Accept'].sort_values()
'''
# Feature Engineering
data['p/picqual'] = data['Price']/data['Pic Quality']
data['p/Rev'] = data['Price']/data['Review']
data['p/Beds'] = data['Price']/data['Beds']
for i in range(1, 11):
    data.loc[data['Region'] == i, 'expensive than average region'] = data.loc[data['Region'] == i, 'Price'] - \
                                                             data.loc[data['Region'] == i, 'Price'].mean()
for i in range(1, 8):
    data.loc[data['Weekday'] == i, 'expensive than average weekday'] = data.loc[data['Weekday'] == i, 'Price'] - \
                                                                      data.loc[data['Weekday'] == i, 'Price'].mean()
for i in range(1, 366):
    data.loc[data['Date'] == i, 'expensive than average date'] = data.loc[data['Date'] == i, 'Price'] - \
                                                                      data.loc[data['Date'] == i, 'Price'].mean()
for i in range(2):
    data.loc[data['Apartment'] == i, 'expensive than average apartment'] = data.loc[data['Apartment'] == i, 'Price'] - \
                                                                      data.loc[data['Apartment'] == i, 'Price'].mean()
for i in range(1, 5):
    data.loc[data['Beds'] == i, 'expensive than average bed'] = data.loc[data['Beds'] == i, 'Price'] - \
                                                                data.loc[data['Beds'] == i, 'Price'].mean()
threshold1 = Binarizer(threshold=3.0)
res1 = pd.DataFrame(threshold1.transform(data['Review'].values.reshape(-1, 1)))
threshold2 = Binarizer(threshold=80)
res2 = pd.DataFrame(threshold2.transform(data['Price'].values.reshape(-1, 1)))
pf = PolynomialFeatures(degree=2, interaction_only=True,
                        include_bias=False)

res3 = pd.DataFrame(pf.fit_transform(data[['Apartment', 'Beds', 'Review', 'Pic Quality', 'Price']]))


encoder = OneHotEncoder()
data_region1hot = encoder.fit_transform(data['Region'].values.reshape(-1, 1))
data_region = pd.DataFrame(data_region1hot.toarray())
data_weekday1hot = encoder.fit_transform(data['Weekday'].values.reshape(-1, 1))
data_weekday = pd.DataFrame(data_weekday1hot.toarray())
data_reformed = pd.concat([data.drop(columns=['ID']),
                           data_region, data_weekday, res1, res2, res3], axis=1)

Seed = 40

split = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=Seed)
for train_index, test_index in split.split(data_reformed, data_reformed['Accept']):
    train = data_reformed.loc[train_index]
    test = data_reformed.loc[test_index]
train_data = train.loc[:, train.columns != 'Accept']
train_label = train.loc[:, train.columns == 'Accept']
test_data = test.loc[:, test.columns != 'Accept']
test_label = test.loc[:, test.columns == 'Accept']
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)


weights = (len(train_label)-np.sum(train_label))/np.sum(train_label)

'''
# Training
# Build xgboost classifier
# Tuning number of estimators
param_test1 = {'n_estimators': range(50, 300, 25)}
gsearch1 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=150,
                                                max_depth=5,
                                                min_child_weight=3,
                                                gamma=0,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test1,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch1.fit(train_data, train_label.ravel())
print(gsearch1.best_params_, gsearch1.best_score_)
# optimal parameter: {'n_estimators': 50} 0.7874907842576959

xgb1 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=50,
                     max_depth=5,
                     min_child_weight=3,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0, reg_lambda=1,
                     objective='binary:logistic', nthread=4, seed=Seed)
xgb1.fit(train_data, train_label.ravel())
predictions1 = xgb1.predict(test_data)
accuracy1 = accuracy_score(test_label, predictions1)
recall1 = recall_score(test_label, predictions1)
f1_1 = f1_score(test_label, predictions1)
auc1 = roc_auc_score(test_label, xgb1.predict_proba(test_data)[:, 1])
auc_train1 = roc_auc_score(train_label, xgb1.predict_proba(train_data)[:, 1])
print(accuracy1, recall1, f1_1, auc1, auc_train1)
# get 0.6916 0.8387176325524044 0.5952047602380119 0.7917022507448125 0.8172005660072216

# Tuning max_depth and min_child_weight
param_test2 = {
    'max_depth': range(3, 9, 2),
    'min_child_weight': range(1, 6, 2)
}

gsearch2 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=50,
                                                max_depth=5,
                                                min_child_weight=1,
                                                gamma=0,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test2,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch2.fit(train_data, train_label.ravel())
print(gsearch2.best_params_, gsearch2.best_score_)
# optimal parameters: {'max_depth': 5, 'min_child_weight': 5} 0.7876668323414941

xgb2 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=50,
                     max_depth=5,
                     min_child_weight=5,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0, reg_lambda=1,
                     objective='binary:logistic', nthread=4, seed=Seed)
xgb2.fit(train_data, train_label.ravel())
predictions2 = xgb2.predict(test_data)
accuracy2 = accuracy_score(test_label, predictions2)
recall2 = recall_score(test_label, predictions2)
f1_2 = f1_score(test_label, predictions2)
auc2 = roc_auc_score(test_label, xgb2.predict_proba(test_data)[:, 1])
auc_train2 = roc_auc_score(train_label, xgb2.predict_proba(train_data)[:, 1])
print(accuracy2, recall2, f1_2, auc2, auc_train2)
# get 0.6908666666666666 0.8394574599260173 0.5948449104412408 0.792500052104486 0.8172330792692573

# Tuning gamma
param_test3 = {'gamma': [i/10.0 for i in range(0, 5)]}
gsearch3 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=50,
                                                max_depth=5,
                                                min_child_weight=5,
                                                gamma=0,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test3,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch3.fit(train_data, train_label.ravel())
print(gsearch3.best_params_, gsearch3.best_score_)
# optimal parameter: {'gamma': 0.1} 0.7877064625861381

xgb3 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=50,
                     max_depth=5,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0, reg_lambda=1,
                     objective='binary:logistic', nthread=4, seed=Seed)
xgb3.fit(train_data, train_label.ravel())
predictions3 = xgb3.predict(test_data)
accuracy3 = accuracy_score(test_label, predictions3)
recall3 = recall_score(test_label, predictions3)
f1_3 = f1_score(test_label, predictions3)
auc3 = roc_auc_score(test_label, xgb3.predict_proba(test_data)[:, 1])
auc_train3 = roc_auc_score(train_label, xgb3.predict_proba(train_data)[:, 1])
print(accuracy3, recall3, f1_3, auc3, auc_train3)
# get 0.691 0.8394574599260173 0.5949488770427336 0.792080748997763 0.8171704200549978

# Tuning subsample and colsample_bytree
param_test4 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}
gsearch4 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=50,
                                                max_depth=5,
                                                min_child_weight=5,
                                                gamma=0.1,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test4,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch4.fit(train_data, train_label.ravel())
print(gsearch4.best_params_, gsearch4.best_score_)
# optimal paramters: {'colsample_bytree': 0.6, 'subsample': 0.7} 0.7891815629881815

xgb4 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=50,
                     max_depth=5,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.7,
                     colsample_bytree=0.6,
                     reg_alpha=0, reg_lambda=1,
                     objective='binary:logistic', nthread=4, seed=Seed)
xgb4.fit(train_data, train_label.ravel())
predictions4 = xgb4.predict(test_data)
accuracy4 = accuracy_score(test_label, predictions4)
recall4 = recall_score(test_label, predictions4)
f1_4 = f1_score(test_label, predictions4)
auc4 = roc_auc_score(test_label, xgb4.predict_proba(test_data)[:, 1])
auc_train4 = roc_auc_score(train_label, xgb4.predict_proba(train_data)[:, 1])
print(accuracy4, recall4, f1_4, auc4, auc_train4)
# get 0.6942 0.8369913686806412 0.5967472527472527 0.7919680793835787 0.81779747365746

# Tuning 'reg_alpha' and 'reg_lambda'
param_test5 = {
    'reg_alpha': [0, 0.05, 0.1, 1, 2, 3],
    'reg_lambda': [0.05, 0.1, 1, 2, 3]
}
gsearch5 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=50,
                                                max_depth=5,
                                                min_child_weight=5,
                                                gamma=0.1,
                                                subsample=0.7,
                                                colsample_bytree=0.6,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test5,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch5.fit(train_data, train_label.ravel())
print(gsearch5.best_params_, gsearch5.best_score_)
# optimal parameters: {'reg_alpha': 0, 'reg_lambda': 1} 0.7891815629881815

xgb5 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=50,
                     max_depth=5,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.7,
                     colsample_bytree=0.6,
                     reg_alpha=0, reg_lambda=1,
                     objective='binary:logistic', nthread=4, seed=Seed)
xgb5.fit(train_data, train_label.ravel())
predictions5 = xgb5.predict(test_data)
accuracy5 = accuracy_score(test_label, predictions5)
recall5 = recall_score(test_label, predictions5)
f1_5 = f1_score(test_label, predictions5)
auc5 = roc_auc_score(test_label, xgb5.predict_proba(test_data)[:, 1])
auc_train5 = roc_auc_score(train_label, xgb5.predict_proba(train_data)[:, 1])
print(accuracy5, recall5, f1_5, auc5, auc_train5)
# get 0.6942 0.8369913686806412 0.5967472527472527 0.7919680793835787 0.81779747365746

# Tuning learning_rate
param_test5 = {
 'learning_rate': [0.01, 0.05, 0.075, 0.1, 0.2, 0.5]
}
gsearch5 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=50,
                                                max_depth=5,
                                                min_child_weight=5,
                                                gamma=0.1,
                                                subsample=0.7,
                                                colsample_bytree=0.6,
                                                reg_alpha=0, reg_lambda=1,
                                                objective='binary:logistic', nthread=4, seed=Seed),
                        param_grid=param_test5,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch5.fit(train_data, train_label.ravel())
print(gsearch5.best_params_, gsearch5.best_score_)
# optimal parameter: {'learning_rate': 0.1} 0.7891815629881815
'''
xgb_model = XGBClassifier(scale_pos_weight=weights,
                          learning_rate=0.1,
                          n_estimators=50,
                          max_depth=5,
                          min_child_weight=5,
                          gamma=0.1,
                          subsample=0.7,
                          colsample_bytree=0.6,
                          reg_alpha=0, reg_lambda=1,
                          objective='binary:logistic', nthread=4, seed=Seed)
'''
xgb_model.fit(train_data, train_label.ravel())
predictions_xgb = xgb_model.predict(test_data)
accuracy_xgb = accuracy_score(test_label, predictions_xgb)
recall_xgb = recall_score(test_label, predictions_xgb)
f1_xgb = f1_score(test_label, predictions_xgb)
auc_xgb = roc_auc_score(test_label, xgb_model.predict_proba(test_data)[:, 1])
auc_train_xgb = roc_auc_score(train_label, xgb_model.predict_proba(train_data)[:, 1])
print(accuracy_xgb, recall_xgb, f1_xgb, auc_xgb, auc_train_xgb)
# final xgb_model get 0.6942 0.8369913686806412 0.5967472527472527 0.7919680793835787 0.81779747365746

# Build Random Forest classifier
# Tuning number of trees
param_test6 = {
    'n_estimators': range(20, 150, 20)
}
gsearch6 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=20,
                                                         min_samples_split=100,
                                                         min_samples_leaf=20,
                                                         max_depth=8,
                                                         max_features='sqrt',
                                                         random_state=Seed,
                                                         class_weight={0: 1, 1: weights},
                                                         oob_score=True),
                        param_grid=param_test6,
                        scoring='roc_auc',
                        cv=5)
gsearch6.fit(train_data, train_label.ravel())
print(gsearch6.best_params_, gsearch6.best_score_)
# optimal parameter: {'n_estimators': 140} 0.7835480908805582

rf1 = RandomForestClassifier(n_estimators=140,
                             min_samples_split=100,
                             min_samples_leaf=20,
                             max_depth=8,
                             max_features='sqrt',
                             random_state=Seed,
                             class_weight={0: 1, 1: weights},
                             oob_score=True)
rf1.fit(train_data, train_label.ravel())
predictions6 = rf1.predict(test_data)
accuracy6 = accuracy_score(test_label, predictions6)
recall6 = recall_score(test_label, predictions6)
f1_6 = f1_score(test_label, predictions6)
auc6 = roc_auc_score(test_label, rf1.predict_proba(test_data)[:, 1])
auc_train6 = roc_auc_score(train_label, rf1.predict_proba(train_data)[:, 1])
print(accuracy6, recall6, f1_6, auc6, auc_train6)
# get 0.6901333333333334 0.8293464858199754 0.5913486899947249 0.7865761043757066 0.8132190447475316

# Tuning max_depth and min_samples_split
param_test7 = {
    'max_depth': range(3, 10, 2),
    'min_samples_split': range(50, 201, 30)
}
gsearch7 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=140,
                                                         min_samples_split=100,
                                                         min_samples_leaf=20,
                                                         max_depth=8,
                                                         max_features='sqrt',
                                                         random_state=Seed,
                                                         class_weight={0: 1, 1: weights},
                                                         oob_score=True),
                        param_grid=param_test7,
                        scoring='roc_auc',
                        cv=5)
gsearch7.fit(train_data, train_label.ravel())
print(gsearch7.best_params_, gsearch7.best_score_)
# optimal parameters: {'max_depth': 9, 'min_samples_split': 110} 0.784252859999122

rf2 = RandomForestClassifier(n_estimators=140,
                             min_samples_split=110,
                             min_samples_leaf=20,
                             max_depth=9,
                             max_features='sqrt',
                             random_state=Seed,
                             class_weight={0: 1, 1: weights},
                             oob_score=True)
rf2.fit(train_data, train_label.ravel())
predictions7 = rf2.predict(test_data)
accuracy7 = accuracy_score(test_label, predictions7)
recall7 = recall_score(test_label, predictions7)
f1_7 = f1_score(test_label, predictions7)
auc7 = roc_auc_score(test_label, rf2.predict_proba(test_data)[:, 1])
auc_train7 = roc_auc_score(train_label, rf2.predict_proba(train_data)[:, 1])
print(accuracy7, recall7, f1_7, auc7, auc_train7)
# get 0.692 0.8254007398273736 0.5916563549584586 0.7874697103948167 0.8210995252678848

# Tuning min_samples_split and min_samples_leaf
param_test8 = {
    'min_samples_leaf':range(10, 60, 10),
    'min_samples_split': range(90, 130, 10)
}
gsearch8 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=140,
                                                         min_samples_split=110,
                                                         min_samples_leaf=20,
                                                         max_depth=9,
                                                         max_features='sqrt',
                                                         random_state=Seed,
                                                         class_weight={0: 1, 1: weights},
                                                         oob_score=True),
                        param_grid=param_test8,
                        scoring='roc_auc',
                        cv=5)
gsearch8.fit(train_data, train_label.ravel())
print(gsearch8.best_params_, gsearch8.best_score_)
# optimal parameters: {'min_samples_leaf': 20, 'min_samples_split': 100} 0.784360314372237

rf3 = RandomForestClassifier(n_estimators=120,
                             min_samples_split=100,
                             min_samples_leaf=20,
                             max_depth=9,
                             max_features='sqrt',
                             random_state=Seed,
                             class_weight={0: 1, 1: weights},
                             oob_score=True)
rf3.fit(train_data, train_label.ravel())
predictions8 = rf3.predict(test_data)
accuracy8 = accuracy_score(test_label, predictions8)
recall8 = recall_score(test_label, predictions8)
f1_8 = f1_score(test_label, predictions8)
auc8 = roc_auc_score(test_label, rf3.predict_proba(test_data)[:, 1])
auc_train8 = roc_auc_score(train_label, rf3.predict_proba(train_data)[:, 1])
print(accuracy8, recall8, f1_8, auc8, auc_train8)
# get 0.6921333333333334 0.8249075215782984 0.591616554651574 0.7873670786394702 0.822084826926828

# Tuning max_features
param_test9 = {'max_features': range(3, 13)}
gsearch9 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=120,
                                                         min_samples_split=100,
                                                         min_samples_leaf=20,
                                                         max_depth=9,
                                                         max_features='sqrt',
                                                         random_state=Seed,
                                                         class_weight={0: 1, 1: weights},
                                                         oob_score=True),
                        param_grid=param_test9,
                        scoring='roc_auc',
                        cv=5)
gsearch9.fit(train_data, train_label.ravel())
print(gsearch9.best_params_, gsearch9.best_score_)
# optimal parameter: {'max_features': 11} 0.7849360946226477
'''
rf_model = RandomForestClassifier(n_estimators=120,
                                  min_samples_split=100,
                                  min_samples_leaf=20,
                                  max_depth=9,
                                  max_features=11,
                                  random_state=Seed,
                                  class_weight={0: 1, 1: weights},
                                  oob_score=True)
'''
rf_model.fit(train_data, train_label.ravel())
predictions_rf = rf_model.predict(test_data)
accuracy_rf = accuracy_score(test_label, predictions_rf)
recall_rf = recall_score(test_label, predictions_rf)
f1_rf = f1_score(test_label, predictions_rf)
auc_rf = roc_auc_score(test_label, rf_model.predict_proba(test_data)[:, 1])
auc_train_rf = roc_auc_score(train_label, rf_model.predict_proba(train_data)[:, 1])
print(accuracy_rf, recall_rf, f1_rf, auc_rf, auc_train_rf)
# final RF model get 0.6936666666666667 0.8239210850801479 0.5925334752150395 0.7879292663294052 0.8254250679625883

# Build GradientBoostingClassifier
# Tuning n_estimators
param_test12 = {
    'n_estimators': range(50, 150, 20)
}
gsearch12 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=100,
                                                              learning_rate=0.1,
                                                              min_samples_split=300,
                                                              min_samples_leaf=20,
                                                              max_depth=8,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test12,
                         scoring='roc_auc',
                         cv=5)
gsearch12.fit(train_data, train_label.ravel())
print(gsearch12.best_params_, gsearch12.best_score_)
# optimal parameter: {'n_estimators': 50} 0.7849290054894699

gc1 = GradientBoostingClassifier(n_estimators=50,
                                 learning_rate=0.1,
                                 min_samples_split=300,
                                 min_samples_leaf=20,
                                 max_depth=8,
                                 max_features='sqrt',
                                 subsample=0.8,
                                 random_state=Seed)
gc1.fit(train_data, train_label.ravel())
predictions12 = gc1.predict(test_data)
accuracy12 = accuracy_score(test_label, predictions12)
recall12 = recall_score(test_label, predictions12)
f1_12 = f1_score(test_label, predictions12)
auc12 = roc_auc_score(test_label, gc1.predict_proba(test_data)[:, 1])
auc_train12 = roc_auc_score(train_label, gc1.predict_proba(train_data)[:, 1])
print(accuracy12, recall12, f1_12, auc12, auc_train12)
# get 0.7358666666666667 0.3336621454993835 0.40581883623275344 0.7883685437612905 0.8475484410666853

# Tuning max_depth and min_samples_split
param_test13 = {
    'max_depth': range(3, 9, 2),
    'min_samples_split': range(100, 801, 200)
}
gsearch13 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=50,
                                                              learning_rate=0.1,
                                                              min_samples_split=300,
                                                              min_samples_leaf=20,
                                                              max_depth=8,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test13,
                         scoring='roc_auc',
                         cv=5)
gsearch13.fit(train_data, train_label.ravel())
print(gsearch13.best_params_, gsearch13.best_score_)
# optimal parameters: {'max_depth': 7, 'min_samples_split': 700} 0.7867562046064023

gc2 = GradientBoostingClassifier(n_estimators=50,
                                 learning_rate=0.1,
                                 min_samples_split=700,
                                 min_samples_leaf=20,
                                 max_depth=7,
                                 max_features='sqrt',
                                 subsample=0.8,
                                 random_state=Seed)
gc2.fit(train_data, train_label.ravel())
predictions13 = gc2.predict(test_data)
accuracy13 = accuracy_score(test_label, predictions13)
recall13 = recall_score(test_label, predictions13)
f1_13 = f1_score(test_label, predictions13)
auc13 = roc_auc_score(test_label, gc2.predict_proba(test_data)[:, 1])
auc_train13 = roc_auc_score(train_label, gc2.predict_proba(train_data)[:, 1])
print(accuracy13, recall13, f1_13, auc13, auc_train13)
# get 0.7398666666666667 0.3376078914919852 0.4123493975903615 0.7893967765066787 0.8256476695806967

# Tuning min_samples_split and min_samples_leaf
param_test14 = {
    'min_samples_split': range(500, 900, 50),
    'min_samples_leaf': range(60, 101, 10)
}
gsearch14 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=50,
                                                              learning_rate=0.1,
                                                              min_samples_split=700,
                                                              min_samples_leaf=20,
                                                              max_depth=7,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test14,
                         scoring='roc_auc',
                         cv=5)
gsearch14.fit(train_data, train_label.ravel())
print(gsearch14.best_params_, gsearch14.best_score_)
# optimal parameters: {'min_samples_leaf': 60, 'min_samples_split': 850} 0.7876015094756067

gc3 = GradientBoostingClassifier(n_estimators=50,
                                 learning_rate=0.1,
                                 min_samples_split=850,
                                 min_samples_leaf=60,
                                 max_depth=7,
                                 max_features='sqrt',
                                 subsample=0.8,
                                 random_state=Seed)
gc3.fit(train_data, train_label.ravel())
predictions14 = gc3.predict(test_data)
accuracy14 = accuracy_score(test_label, predictions14)
recall14 = recall_score(test_label, predictions14)
f1_14 = f1_score(test_label, predictions14)
auc14 = roc_auc_score(test_label, gc3.predict_proba(test_data)[:, 1])
auc_train14 = roc_auc_score(train_label, gc3.predict_proba(train_data)[:, 1])
print(accuracy14, recall14, f1_14, auc14, auc_train14)
# get 0.7366 0.3316892725030826 0.40505947899412736 0.7896511815889222 0.8210174702539141

# Tuning max_features
param_test15 = {
    'max_features': range(3, 13, 2)
}
gsearch15 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=50,
                                                              learning_rate=0.1,
                                                              min_samples_split=850,
                                                              min_samples_leaf=60,
                                                              max_depth=7,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test15,
                         scoring='roc_auc',
                         cv=5)
gsearch15.fit(train_data, train_label.ravel())
print(gsearch15.best_params_, gsearch15.best_score_)
# optimal parameter: {'max_features': 7} 0.7876015094756067

gc4 = GradientBoostingClassifier(n_estimators=50,
                                 learning_rate=0.1,
                                 min_samples_split=850,
                                 min_samples_leaf=60,
                                 max_depth=7,
                                 max_features=7,
                                 subsample=0.8,
                                 random_state=Seed)
gc4.fit(train_data, train_label.ravel())
predictions15 = gc4.predict(test_data)
accuracy15 = accuracy_score(test_label, predictions15)
recall15 = recall_score(test_label, predictions15)
f1_15 = f1_score(test_label, predictions15)
auc15 = roc_auc_score(test_label, gc4.predict_proba(test_data)[:, 1])
auc_train15 = roc_auc_score(train_label, gc4.predict_proba(train_data)[:, 1])
print(accuracy15, recall15, f1_15, auc15, auc_train15)
# get 0.7366 0.3316892725030826 0.40505947899412736 0.7896511815889222 0.8210174702539141

# Tuning subsample
param_test16 = {
    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
}
gsearch16 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=50,
                                                              learning_rate=0.1,
                                                              min_samples_split=850,
                                                              min_samples_leaf=60,
                                                              max_depth=7,
                                                              max_features=7,
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test16,
                         scoring='roc_auc',
                         cv=5)
gsearch16.fit(train_data, train_label.ravel())
print(gsearch16.best_params_, gsearch16.best_score_)
# optimal parameter: {'subsample': 0.8} 0.7876015094756067, doesn't change.

# Tuning learning_rate
param_test17 = {
    'learning_rate': [2, 1, 0.5, 0.1, 0.05, 0.01]
}
gsearch17 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=50,
                                                              learning_rate=0.1,
                                                              min_samples_split=850,
                                                              min_samples_leaf=60,
                                                              max_depth=7,
                                                              max_features=7,
                                                              subsample=0.8,
                                                              random_state=Seed),
                         param_grid=param_test17,
                         scoring='roc_auc',
                         cv=5)
gsearch17.fit(train_data, train_label.ravel())
print(gsearch17.best_params_, gsearch17.best_score_)
# optimal parameter: {'learning_rate': 0.1} 0.7876015094756067, doesn't change.
'''
gc_model = GradientBoostingClassifier(n_estimators=50,
                                      learning_rate=0.1,
                                      min_samples_split=850,
                                      min_samples_leaf=60,
                                      max_depth=7,
                                      max_features=7,
                                      subsample=0.8,
                                      random_state=Seed)
'''
gc_model.fit(train_data, train_label.ravel())
predictions_gc = gc_model.predict(test_data)
accuracy_gc = accuracy_score(test_label, predictions_gc)
recall_gc = recall_score(test_label, predictions_gc)
f1_gc = f1_score(test_label, predictions_gc)
auc_gc = roc_auc_score(test_label, gc_model.predict_proba(test_data)[:, 1])
auc_train_gc = roc_auc_score(train_label, gc_model.predict_proba(train_data)[:, 1])
print(accuracy_gc, recall_gc, f1_gc, auc_gc, auc_train_gc)
# Final GradientBoostingClassifier get 0.7366 0.3316892725030826 0.40505947899412736 0.7896511815889222 0.8210174702539141


# Stacking Classifier

lr = LogisticRegression(class_weight={0: 1, 1: weights})
xgb_model.fit(train_data, train_label.ravel())
rf_model.fit(train_data, train_label.ravel())
gc_model.fit(train_data, train_label.ravel())
log_clf = LogisticRegression()
ac = AdaBoostClassifier()
log_clf.fit(train_data, train_label.ravel())
ac.fit(train_data, train_label.ravel())
sclf = StackingClassifier(classifiers=[xgb_model, rf_model, gc_model, ac, log_clf],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
base_predictions_train = pd.DataFrame( {'xgboost': xgb_model.predict_proba(train_data)[:, 1],
      'GradientBoost': gc_model.predict_proba(train_data)[:, 1],
    'RandomForest': rf_model.predict_proba(train_data)[:, 1],
    'logistic': log_clf.predict_proba(train_data)[:, 1],
    'adaboost': ac.predict_proba(train_data)[:, 1]
})
lr.fit(base_predictions_train, train_label.ravel())
sclf.fit(train_data, train_label)
predictions = sclf.predict(test_data)
accuracy = accuracy_score(test_label, predictions)
recall = recall_score(test_label, predictions)
f1 = f1_score(test_label, predictions)
auc = roc_auc_score(test_label, sclf.predict_proba(test_data)[:, 1])
auc_train = roc_auc_score(train_label, sclf.predict_proba(train_data)[:, 1])
print(accuracy, recall, f1, auc, auc_train, lr.coef_)
'''
# Voting Classifier
log_clf = LogisticRegression(class_weight={0: 1, 1: weights})
ac = AdaBoostClassifier()
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model), ('gc', gc_model), ('log_clf', log_clf),
                                          ('ac', ac)],
                              voting='soft')
voting_clf.fit(train_data, train_label.ravel())
predictions = voting_clf.predict(test_data)
accuracy = accuracy_score(test_label, predictions)
recall = recall_score(test_label, predictions)
f1 = f1_score(test_label, predictions)
auc = roc_auc_score(test_label, voting_clf.predict_proba(test_data)[:, 1])
auc_train = roc_auc_score(train_label, voting_clf.predict_proba(train_data)[:, 1])
print(accuracy, recall, f1, auc, auc_train)

data1 = pd.read_csv('/Users/xiujiayang/Downloads/kaggle/venv/test.csv')
data1['p/picqual'] = data1['Price']/data1['Pic Quality']
data1['p/Rev'] = data1['Price']/data1['Review']
data1['p/Beds'] = data1['Price']/data1['Beds']
for i in range(1, 11):
    data1.loc[data1['Region'] == i, 'expensive than average region'] = data1.loc[data1['Region'] == i, 'Price'] - \
                                                             data1.loc[data1['Region'] == i, 'Price'].mean()
for i in range(1, 8):
    data1.loc[data1['Weekday'] == i, 'expensive than average weekday'] = data1.loc[data1['Weekday'] == i, 'Price'] - \
                                                                      data1.loc[data1['Weekday'] == i, 'Price'].mean()
for i in range(1, 366):
    data1.loc[data1['Date'] == i, 'expensive than average date'] = data1.loc[data1['Date'] == i, 'Price'] - \
                                                                      data1.loc[data1['Date'] == i, 'Price'].mean()
for i in range(2):
    data1.loc[data1['Apartment'] == i, 'expensive than average apartment'] = data1.loc[data1['Apartment'] == i, 'Price'] - \
                                                                      data1.loc[data1['Apartment'] == i, 'Price'].mean()
for i in range(1, 5):
    data1.loc[data1['Beds'] == i, 'expensive than average bed'] = data1.loc[data1['Beds'] == i, 'Price'] - \
                                                                data1.loc[data1['Beds'] == i, 'Price'].mean()
threshold1 = Binarizer(threshold=3.0)
Res1 = pd.DataFrame(threshold1.transform(data1['Review'].values.reshape(-1, 1)))
threshold2 = Binarizer(threshold=80)
Res2 = pd.DataFrame(threshold2.transform(data1['Price'].values.reshape(-1, 1)))
pf = PolynomialFeatures(degree=2, interaction_only=True,
                        include_bias=False)

Res3 = pd.DataFrame(pf.fit_transform(data1[['Apartment', 'Beds', 'Review', 'Pic Quality', 'Price']]))


encoder = OneHotEncoder()
data1_region1hot = encoder.fit_transform(data1['Region'].values.reshape(-1, 1))
data1_region = pd.DataFrame(data1_region1hot.toarray())
data1_weekday1hot = encoder.fit_transform(data1['Weekday'].values.reshape(-1, 1))
data1_weekday = pd.DataFrame(data1_weekday1hot.toarray())
data1_reformed = pd.concat([data1.drop(columns=['ID']),
                           data1_region, data1_weekday, Res1, Res2, Res3], axis=1)
output_test = np.array(data1_reformed)
output = pd.DataFrame(voting_clf.predict_proba(output_test)[:, 1], index=data1['ID'], columns=['Probability of 1'])
output.to_csv('/Users/xiujiayang/Downloads/output.csv')
'''
# Error analysis
dn = []
for i in range(len(test_data)):
    if predictions[i] != test_label[i]:
        df = pd.concat([pd.DataFrame(test_data[i]),pd.DataFrame(test_label[i])], axis=1)
        dn.append(df)
dn = pd.concat(dn, axis=1)
dn.T.to_csv('/Users/xiujiayang/Downloads/kaggle/venv/error.csv')
data_error =pd.read_csv('/Users/xiujiayang/Downloads/kaggle/venv/error.csv')
data_error.dropna(axis=0, inplace=True)
data_error.plot(kind='scatter', x='5',y='7')
data['Region'].hist()
data_error['0'].hist()
'''
