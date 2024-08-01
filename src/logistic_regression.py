'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import classification_report

def logistic_regression(df_arrests):
    """performs logistic regression to predict felony arrests

    Args:
        df_arrests (df):dataframe containing arrest information

    Returns:
        df_arrests_train(df):  training dataframe
        df_arrests_test(df): The testing dataframe
        gs_cv(object): GridSearchCV object 
    """
    
    #creates train and test sets
    df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'], random_state=42)
    
    #create a list of features
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    
    #creates parameter grid
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    
    #initialize the Logistic Regression model
    lr_model = lr(solver='liblinear')
    
    #initializes GridSearch
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')
    
    gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])
    
    #identify the optimal C
    optimal_C = gs_cv.best_params_['C']
    print(f"The optimal value for C is: {optimal_C}")
    
    if optimal_C == 0.01:
        regularization = "most regularization"
    elif optimal_C == 100.0:
        regularization = "least regularization"
    else:
        regularization = "in the middle"
    
    print(f"Did it have the most or least regularization? Or in the middle? {regularization}")
    
    #predict for the test set
    df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])
    
    #prints classification report for the test predictions
    print(classification_report(df_arrests_test['y'], df_arrests_test['pred_lr']))
    

    return df_arrests_train, df_arrests_test, gs_cv




