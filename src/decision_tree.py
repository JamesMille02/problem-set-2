'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
from sklearn.model_selection import  GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import GridSearchCV

def decision_tree(df_arrests_train, df_arrests_test):
    """initializes decision tree model and makes predictions on the test set

    Args:
        df_arrests_train(dataframe): The training dataframe.
        df_arrests_test(dataFrame): The testing dataframe.

    Returns:
        df_arrests_test(dataframe): The testing dataframe 
        gs_cv_dt(GridSearchCV):  GridSearchCV object 
    """
    # creates a parameter grid for tree depth
    param_grid_dt = {'max_depth': [3, 5, 7]}
    
    #initializes  the DTC
    dt_model = DTC()
    
    #initialize gridsearchcv
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy')
    
    #features created
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    
    gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])
    
    #identifies the  value for max_depth
    max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"The optimal value for max_depth is: {max_depth}")
    
    if max_depth == 3:
        regularization = "most regularization"
    elif max_depth == 7:
        regularization = "least regularization"
    else:
        regularization = "in the middle"
    
    print(f"Did it have the most or least regularization? Or in the middle? {regularization}")
    
    #predicts the test set
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[features])
    
    #return test dataframe and predictions
    return df_arrests_test, gs_cv_dt

