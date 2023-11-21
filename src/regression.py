from sklearn.linear_model import LogisticRegression
import pandas as pd

def create_model(train_file, cols):
    """This function create a model with specific selected features and train the model with the trainning data

    Args:
        train_file (DataFrame): file with the trainning data
        cols (list): list of selected features

    Returns:
        LogisticRegression: logistic regression model
    """
    
    X = train_file[cols]
    y = train_file['Survived']
    
    model = LogisticRegression() #create model
    model.fit(X,y)  #train model
    
    return model

def predict(model, test_file, cols):
    """ This function predict the survival of the passengers in the test file

    Args:
        model (LogisticRegression): logistic regression model
        test_file (DataFrame): file with the test data
        cols (list): list of selected features

    Returns:
        list: list of prediction
    """
    
    X = test_file[cols]
    
    y_pred = model.predict(X) #predict
    
    return y_pred

def similarity_percentage(y_pred, y_true):
    """This function compare the prediction with the gender_submission file and return the percentage of similarity

    Args:
        y_pred (list): list of prediction
        y_true (list): list of true awnser


    Raises:
        ValueError: if the length of the list is different from the length of the test file

    Returns:
        float: percentage of similarity between the prediction and the test file / Accuracy of the model
    """
    
    if len(y_pred) != len(y_true):
        raise ValueError("The length of lists are different.") # Check if the length of the list is different from the length of the test file
    
    matching_elements = sum(x == y for x, y in zip(y_pred, y_true)) # loop to compare the prediction with the test file
    
    percentage = round((matching_elements / len(y_pred)) * 100,2) # Calculate the percentage of similarity between the prediction and the test file / Accuracy of the model
    
    return percentage