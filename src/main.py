from work_data import * 
from regression import *

if __name__ == '__main__':
    
    #------------------------------------------------------------------------------------------------
    #                                      LOGISTIC REGRESSION MODEL
    #------------------------------------------------------------------------------------------------
    
    # Load the data
    
    test_file, train_file = import_data()    
    
    # Clean and normalize the data
    clean_data_train(train_file)
    
    train_nomalized = normalize_data(train_file)
    test_nomalized = normalize_data(test_file)
    clean_data_test_normalized(test_nomalized)
    

    # Create the model and predict
    
    
    cols = ["TravelAlone","Sex_male","IsMinor"] # Selected Features      
    
    """ 
    ["Age","Fare","TravelAlone","Pclass_1","Pclass_2", "Pclass_3", "Embarked_C","Embarked_S","Sex_male","IsMinor"] : accurency = 97.85% 
    ["Age","Fare","TravelAlone","Embarked_C","Embarked_S", "Sex_male","IsMinor"] :  accurency = 98.33% 
    ["TravelAlone","Sex_male","IsMinor"] : accurency = 100% 
    
    """
      
    model = create_model(train_nomalized, cols)
    y_pred = predict(model, test_nomalized, cols)
    
    
    sub = pd.read_csv('./data/gender_submission.csv') # Load the submission file
    y_true = sub['Survived']
    
    result = similarity_percentage(y_pred, y_true)
    
    # Calculate the accuracy of the model
    print(f"Accurency  : {result}%")
    
    