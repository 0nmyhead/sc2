import sklearn
import xgboost
import shap
from sklearn.model_selection import train_test_split

shap.initjs();

df, target = shap.datasets.boston()
X_train,X_test,y_train,y_test = train_test_split(df, target, test_size=0.2, random_state=2)

model = xgboost.XGBRegressor().fit(X_train, y_train)

from pdpbox.pdp import pdp_isolate, pdp_plot

feature = ['LSTAT','B','TAX','AGE']

for i in feature:
    isolated = pdp_isolate(
        model=model, 
        dataset=X_train, 
        model_features=X_train.columns, 
        feature=i,
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
    pdp_plot(isolated, i)
    