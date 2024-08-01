'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot



# Call functions / instanciate objects from the .py files
def main():
    etl.etl()
    df_arrests = preprocessing.preprocessing()
    df_arrests_train, df_arrests_test, gs_cv_lr = logistic_regression.logistic_regression(df_arrests)
    df_arrests_test, gs_cv_dt = decision_tree.decision_tree(df_arrests_train, df_arrests_test)
    calibration_plot.evaluate_models(df_arrests_test)


if __name__ == "__main__":
    main()