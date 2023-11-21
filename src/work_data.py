import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def import_data():
    """This function import the test file and the train file

    Returns:
        DataFrame, DataFrame: content of the test file and the train file
    """
    
    test_file = pd.read_csv('./data/test.csv')
    train_file = pd.read_csv('./data/train.csv')
    
    return test_file, train_file

    

def explore_data(train_file):
    """This function plot the data to explore it and see the correlation between the variables to select the best features

    Args:
        train_file (DataFrame): file with the trainning data
    """

    variables = train_file.select_dtypes(include=['int64', 'float64', 'object', 'bool']).columns # Select the variables to plot
    
    variables = variables[1:] # Remove the Survived variable
    
    # Setting the number of columns and rows for the plot
    num_variables = len(variables)
    num_cols = 3  
    num_rows = (num_variables + num_cols - 1) // num_cols
    plt.figure(figsize=(15.5, 2.5 * num_rows))
   
    # Plotting the variables function of the Survived variable
    for i, variable in enumerate(variables, 1):
        
        plt.subplot(num_rows, num_cols, i)
        
        if train_file[variable].dtype in ['int64', 'float64']:
            sns.kdeplot(data=train_file, x=variable, hue='Survived', fill=True, common_norm=False, alpha=0.5)
        
        else : 
            sns.countplot(data=train_file, x=variable, hue='Survived') 
        
        plt.title(f'Densité de survie et de décès en fonction de {variable}')
        plt.xlabel(variable)
        plt.ylabel('Densité')
        plt.legend(['Décès', 'Survie'])
    
    plt.tight_layout() # To avoid overlapping
    plt.show()



def clean_data_train(train_file):
    """This function clean the train file by filling the missing values and drop the useless variables

    Args:
        train_file (DataFrame): file with the trainning data
    """
    
    if __name__ == '__main__':
        
        print(train_file.isnull().sum()) # Check the number of missing values in each variable
        
        # Interpretation of the results
        print('\nAge : 177 --> ≈ 20% change by the mean') 
        print('Cabin : 687 --> ≈ 70% drop beacause no impact on the result')
        print('Embarked : 2 --> ≈ 0.2% drop\n')
    
    #train_file.drop('Cabin', axis=1, inplace=True) do in the normalize_data function
    
    train_file['Age'].fillna(train_file['Age'].mean(), inplace=True) # Fill the missing age by the mean
    train_file['Embarked'].fillna(train_file['Embarked'].mode()[0], inplace=True) # Fill the missing embarked by the most popular
        
def clean_data_test_normalized(test_file):
    """This function clean the test file by filling the missing values and drop the useless variables

    Args:
        test_file (DataFrame): file with the test data
    """
    
    if __name__ == '__main__':
        print(test_file.isnull().sum())# Check the number of missing values in each variable
        
        # Interpretation of the results
        print('\nAge : 86 -->  change by the mean') 
        print('Fare : 1 -->  change by the mean')
        
    test_file['Age'].fillna(test_file['Age'].mean(), inplace=True)
    test_file['Fare'].fillna(test_file['Fare'].mean(), inplace=True)
        
    
def normalize_data(train_file):
    """This function normalize the train file by creating new variables and drop the useless variables

    Args:
        train_file (DataFrame): file with the trainning data

    Returns:
        DataFrame: file with the trainning data normalized
    """
    
    normalize_data = train_file.copy() # Copy the train file to avoid modifying the original file
    normalize_data['TravelAlone']=np.where((normalize_data["SibSp"]+normalize_data["Parch"])>0, 0, 1) # Create a new variable to know if the passenger travel alone or not
        
    normalize_data = pd.get_dummies(normalize_data, columns=["Pclass","Embarked","Sex"]) # Create dummy variables
     
    normalize_data['IsMinor'] = np.where(normalize_data['Age']<=16, 1, 0) # Create a new variable to know if the passenger is a minor or not

    
    # Drop the useless variables
    normalize_data.drop('SibSp', axis=1, inplace=True)
    normalize_data.drop('Parch', axis=1, inplace=True)
    normalize_data.drop('Sex_female', axis=1, inplace=True)
    normalize_data.drop('PassengerId', axis=1, inplace=True)
    normalize_data.drop('Name', axis=1, inplace=True)
    normalize_data.drop('Ticket', axis=1, inplace=True)
    normalize_data.drop('Cabin', axis=1, inplace=True)
    
    # Convert boolean to int
    boolean_columns = normalize_data.select_dtypes(include=bool).columns
    normalize_data[boolean_columns] = normalize_data[boolean_columns].astype(int)
    
    return normalize_data
    
    

if __name__ == '__main__':  
    
    
    test_file, train_file = import_data()

    #explore_data(train_file)
    
    """ 
    With this first exploration, we can see that the variables that have the less impact on the survival of the passengers are:
        - PassengerId
        - Name
        - Ticket
        - Cabin
    So we can drop them.
    """
    
    clean_data_train(train_file)
    
    train_nomalized = normalize_data(train_file)
    print(train_nomalized.head())
    
    test_nomalized = normalize_data(test_file)
    clean_data_test_normalized(test_nomalized)
    print(test_nomalized.head())
        
    #explore_data(train_nomalized)
    
    """ 
    With this second exploration, we can see that the variables that have the most impact on the survival of the passengers are:
        - IsMinor
        - Sex_male
        - TravelAlone
    
    Other variables can have an impact on the survival of the passengers, we need to make test to know if we add them or not.
    """