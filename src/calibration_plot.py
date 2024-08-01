'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def evaluate_models(df_arrests_test):
    """
    Evaluates the logistic regression and decision tree models with calibration plots,
    ppv, and auc metrics.

    Parameters:
        df_arrests_test(DataFrame): The test dataset 

    Returns:
        None
    """
    y_true = df_arrests_test['y']
    y_prob_lr = df_arrests_test['pred_lr']
    y_prob_dt = df_arrests_test['pred_dt']
    

    calibration_plot(y_true, y_prob_lr, n_bins=5)

    calibration_plot(y_true, y_prob_dt, n_bins=5)
    
    print("The calibration plot for the dt plot is more calabrated than the lr plot")

    #ppv for Logistic Regression model for arrestees in the top 50 predicted risk
    top_50_logreg = df_arrests_test.nlargest(50, 'pred_lr')
    ppv_lr = precision_score(top_50_logreg['y'], top_50_logreg['pred_lr'] > 0.5)
    print(f"PPV for Logistic Regression model for top 50 arrestees: {ppv_lr:.2f}")
    
    #ppv for Decision Tree model for arrestees in the top 50 predicted risk
    top_50_dt = df_arrests_test.nlargest(50, 'pred_dt')
    ppv_dt = precision_score(top_50_dt['y'], top_50_dt['pred_dt'] > 0.5)
    print(f"PPV for Decision Tree model for top 50 arrestees: {ppv_dt:.2f}")
    
    #auc for Logistic Regression model
    auc_lr = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_lr'])
    print(f"AUC for Logistic Regression model: {auc_lr:.2f}")
    
    #auc for Decision Tree model
    auc_dt = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_dt'])
    print(f"AUC for Decision Tree model: {auc_dt:.2f}")
    
    print("Do both metrics agree that one model is more accurate than the other?")
    print("The models do not agree, AUC has each model as equal while ppv does not")