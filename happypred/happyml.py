import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

def hwtrain(X_csv: str, y_csv: str, model: str = 'lm') -> str:
    """ Read the feature matrix and label vector from training data and fit a
        machine learning model. The model is saved in pickle format. 
    
        Parameters
        ----------
        X_csv
            The path to the feature matrix in CSV format
        y_csv
            The path to the label vector in CSV format
        model
            The type of machine learning model
        Returns
        -------
        pickled_model_path
            Path to the pickled model
    """
    # Read the X and y CSV files
    X_train_df = pd.read_csv(X_csv)
    y_train_df = pd.read_csv(y_csv)
    # Fit the model
    if model == 'lm':
        estimator = LinearRegression()
    else:
        raise ValueError('The only available model is "lm"')
    estimator.fit(X_train_df, y_train_df)
    # Save the model
    pickled_model_path = model + '_model.pkl'
    with open(pickled_model_path, 'wb') as model_file:
        pickle.dump(estimator, model_file)
    return pickled_model_path

def hwpredict(log_gdp: float, social: float, life_exp: float, freedom: float,
              generosity: float, corruption: float, model: str = 'lm') -> float:
    """ Predict the happiness of a country from the provided features. 
    
        Parameters
        ----------
        model
            The type of machine learning model
        log_gdp
            The Log GDP per capita
        social
            Social support
        life_exp
            Healthy life expectancy at birth
        freedom
            Freedom to make life choices
        generosity
            Generosity
        corruption
            Perceptions of corruption
        Returns
        -------
        life_ladder
            The happiness score measured as a "Life Ladder"
    """
    # Load the model
    pickled_model_path = model + '_model.pkl'
    with open(pickled_model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    # Predict the outcome
    life_ladder = model.predict([[log_gdp, social, life_exp, freedom, generosity, corruption]])[0][0]
    return life_ladder

def test_hwpredict():
    assert 0 <= test_hwpredict(5, 0.5, 50, 0.5, 0.5,	0.5) <= 10