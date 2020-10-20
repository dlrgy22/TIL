from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
import xgboost
import data_processing


model = RandomForestClassifier(n_estimators=100,
                                            random_state=1234,
                                            min_samples_split=4,
                                            min_samples_leaf=32
                                            )
path = './trainset.csv'
x_data, y_data, name = data_processing.get_data(path)
x_data = data_processing.std_scale(x_data)
x_data = data_processing.pca(x_data,30)
sfs1 = SFS(model,k_features=15, verbose=2,scoring='accuracy', cv=5, n_jobs=-1)
sfs1.fit(x_data, y_data)
print(sfs1.subsets_)
print(sfs1.k_feature_idx_)
print(sfs1.k_feature_names_)