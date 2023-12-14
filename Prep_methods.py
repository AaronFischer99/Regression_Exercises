#Splitting with a 20% test data ratio for test purposes to a 33% test data ratio for vlidation purposes
def train_test_validate2(df, test_size=(0.2, 0.30), random_state=123):
    train, test = train_test_split(df, test_size=test_size[0], random_state=random_state)
    train_validate, test_validate = train_test_split(train, test_size=test_size[1], random_state=random_state)
    
    return train, test, train_validate, test_validate

#Splitting with a 20% test data ratio for test purposes to a 10% test data ratio for vlidation purposes
def train_test_validate4(df, test_size=(0.2, 0.1), random_state=123):
    train, test = train_test_split(df, test_size=test_size[0], random_state=random_state)
    train_validate, test_validate = train_test_split(train, test_size=test_size[1], random_state=random_state)
    
    return train, test, train_validate, test_validate




def train_test_validate3(df, target_col, test_size=(0.2, 0.3), random_state=123):
    # Splitting into train and test while stratifying on the target variable
    train, test = train_test_split(df, test_size=test_size[0], stratify=df[target_col], random_state=random_state)
    
    # Splitting train further into train and validate with stratification
    train_val, test_validate = train_test_split(train, test_size=test_size[1], stratify=train[target_col], random_state=random_state)
    
    return train, validate, test





from sklearn.preprocessing import OneHotEncoder 
  
def one_hot_encode_columns(df, categorical_columns): 
    """ 
    Perform one-hot encoding on selected categorical columns in a DataFrame. 
  
    Parameters: 
    - df (DataFrame): The input DataFrame. 
    - categorical_columns (list): A list of column names to be one-hot encoded. 
  
    Returns: 
    - DataFrame: A new DataFrame with one-hot encoded columns. 
    """ 
    # Select the specified categorical columns 
    categorical_data = df[categorical_columns] 
  
    # Initialize the OneHotEncoder 
    ohe = OneHotEncoder(categories='auto') 
  
    # Perform one-hot encoding and convert to a NumPy array 
    feature_arr = ohe.fit_transform(categorical_data).toarray() 
  
    # Get the feature names after one-hot encoding 
    ohe_labels = ohe.get_feature_names_out(categorical_columns) 
 
 
    # Create a new DataFrame with one-hot encoded columns 
    features = pd.DataFrame(feature_arr, columns=ohe_labels) 
  
    return features 


def drop_columns(dataframe, columns_to_drop):
    """
    Drop specified columns from a DataFrame.

    Parameters:
    - dataframe: The input DataFrame.
    - columns_to_drop: List of column names to be dropped.

    Returns:
    - DataFrame: A new DataFrame with specified columns removed.
    """
    modified_dataframe = dataframe.drop(columns=columns_to_drop, axis=1)
    return modified_dataframe

# df_telco = drop_columns(df_telco, ['payment_type_id', 'contract_type_id', 'internet_service_type_id'])




def calculate_baseline_percentage(df, target_column):
    """
    Calculate the baseline percentage for the minority class in the specified target column.

    Parameters:
    - df: DataFrame
    - target_column: str, the column representing the target variable

    Returns:
    - min_baseline_percentage: float, the baseline percentage for the minority class
    - df_with_baseline: DataFrame, the original DataFrame with a new 'baseline_prediction' column
    """

    # Identify the minority class
    minority_class = df[target_column].unique()[1]

    # Calculate the baseline percentage
    min_baseline_percentage = df[target_column].value_counts(normalize=True).get(minority_class, 0) * 100

    # Create a baseline prediction column with the majority class
    df_with_baseline = df.copy()
    df_with_baseline['baseline_prediction'] = minority_class

    # Display the baseline percentage
    print(f"Baseline Percentage: {min_baseline_percentage:.2f}%")

    return min_baseline_percentage, df_with_baseline

# Example usage:
# Replace 'df_telco' and 'Churn' with your actual DataFrame and target column
baseline_percentage, df_with_baseline = calculate_baseline_percentage(df_telco, 'Churn')

# baseline_percentage, df_with_baseline = calculate_baseline_percentage(df_telco, 'Churn')





import pandas as pd
import numpy as np

def convert_column_to_float(df, column_name):
    """
    Convert a specified column in the DataFrame to float type.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - column_name: str
        The name of the column to be converted to float.

    Returns:
    - DataFrame
        The DataFrame with the specified column converted to float.
    """

    # Replace empty strings with NaN
    df[column_name] = df[column_name].replace(' ', np.nan)

    # Convert the column to float
    dataframe[column_name] = dataframe[column_name].astype(float)

    return df

# Example usage:
# Replace 'your_dataframe' with the actual DataFrame variable and 'your_column' with the actual column name
your_dataframe = pd.DataFrame(...)  # Replace ... with your actual data
your_dataframe = convert_column_to_float(your_dataframe, 'your_column')

# Check the data type after conversion
print(your_dataframe['your_column'].dtype)

# your_dataframe = convert_column_to_float(your_dataframe, 'your_column')
# Check the data type after conversion
# print(your_dataframe['your_column'].dtype)



import pandas as pd
from scipy import stats

from scipy.stats import chi2_contingency

def perform_chi_square_test(dataframe, variable1_name, variable2_name, alpha=0.05):
    """
    Perform a chi-square test of independence between two categorical variables.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - variable1_name (str): Name of the first categorical variable.
    - variable2_name (str): Name of the second categorical variable.
    - alpha (float): Significance level (default is 0.05).

    Returns:
    - tuple: Tuple containing the chi-square statistic, p-value, degrees of freedom, and the expected frequencies.
    - str: Result of the significance test.
    """

    # Create a contingency table
    contingency_table = pd.crosstab(dataframe[variable1_name], dataframe[variable2_name])

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Check if the p-value is less than alpha
    if p_value < alpha:
        result = f"There is a significant association (p-value: {p_value:.4f}, alpha: {alpha})"
    else:
        result = f"There is no significant association (p-value: {p_value:.4f}, alpha: {alpha})"

    return (chi2_stat, p_value, dof, expected), result

# Example usage:
# Assuming 'your_dataframe' is your actual DataFrame and #'variable1'/'variable2' are your column names
#result, test_result = perform_chi_square_test(your_dataframe, 'variable1', #'variable2')

# Display the results
#print(f"Chi-square Statistic: {result[0]:.4f}")




import statsmodels.api as sm
from statsmodels.formula.api import ols

def perform_anova(data, target_column, feature_columns, alpha=0.05):
    """
    Perform ANOVA (Analysis of Variance) for the specified feature columns against the target column.

    Parameters:
    - data: DataFrame, the input data
    - target_column: str, the column representing the target variable (dependent variable)
    - feature_columns: list, a list of strings representing the feature variables (independent variables)
    - alpha: float, the significance level for hypothesis testing (default is 0.05)

    Returns:
    - anova_table: DataFrame, ANOVA table
    - conclusion: str, conclusion based on the statistical test result
    """

    # Create a formula for the ANOVA model
    formula = f"{target_column} ~ {' + '.join(feature_columns)}"

    # Fit the ANOVA model
    model = ols(formula, data=data).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Check the p-value and draw a conclusion
    p_value = anova_table['PR(>F)'][0]

    if p_value < alpha:
        conclusion = 'There is a relationship between the specified features and the target variable.'
    else:
        conclusion = 'We fail to reject the null hypothesis; there is no significant relationship.'

    return anova_table, conclusion

# Example usage:
# Replace 'your_dataframe', 'Churn', and ['online_security', 'online_backup', 'device_protection'] with your actual data and column names
#data_telco = your_dataframe
#result, conclusion = perform_anova(data_telco, 'Churn',  ['online_security', 'online_backup', 'device_protection'])
#print(result)
#print(conclusion)




import pandas as pd

def drop_nulls_for_column(df, column_name):
    """
    Drops rows with null values for a specific column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which null values will be dropped.

    Returns:
    - pd.DataFrame: DataFrame with null values dropped for the specified column.
    """
    # Drop null values for the specified column
    df.dropna(subset=[column_name], inplace=True)
    
    return df

# Example usage:
# df_telco = drop_nulls_for_column(df_telco, 'Total_Charges')




import pandas as pd

def convert_to_binary(df, target_column):
    """
    Convert the target variable in a DataFrame to binary.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - target_column: str
        The name of the target variable column.

    Returns:
    - pd.Series
        The binary representation of the target variable.
    """
    # Extract the target variable column
    y = df[target_column]

    # Convert to binary (1 for values greater than 0, 0 otherwise)
    y_binary = (y > 0).astype(int)

    return y_binary

# Example usage:
# Assuming df is your DataFrame and 'target' is the name of your target variable column
#y_binary_train = convert_to_binary(df, 'target_train')
#y_binary_test = convert_to_binary(df, 'target_test')





import pandas as pd
from sklearn.preprocessing import StandardScaler

def scaling_dataframe(df, columns_to_scale, fill_value=0):
    """
    Preprocess a DataFrame by scaling specified columns and filling/replacing NaN values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns_to_scale (list): List of column names to be scaled.
    - fill_value (int or float): Value to fill NaN entries in the DataFrame.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """

    # Scaling Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns_to_scale])

    # Create a DataFrame from the scaled data with appropriate column labels
    df_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale)

    # Filling/Replacing NaN values with the specified fill value
    df_scaled = df_scaled.fillna(fill_value)

    return df_scaled



# Assuming df_train and df_test are your DataFrames
#columns_to_scale = ['column1', 'column2', 'column3']  # Replace with actual column names

# Preprocess training data
#X_train_scaled_df = scaling_dataframe(df_train, columns_to_scale)

# Preprocess test data
#X_test_scaled_df = scaling_dataframe(df_test, columns_to_scale)



from sklearn.model_selection import KFold, cross_val_score

def cross_val(model, X_train, y_train, n_splits=3, shuffle=False):
    """
    Perform k-fold cross-validation for a given model.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X_train: The training data.
    - y_train: The target labels.
    - n_splits: Number of folds for cross-validation.
    - shuffle: Whether to shuffle the data before splitting.

    Returns:
    - The mean cross-validation score.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    scores = cross_val_score(model, X_train, y_train, cv=kf)
    return round(scores.mean(), 3)

# Example usage with your provided lines of code
# Replace svc, X_train_scaled_df, and y_train_binary with your actual model and data
# For example, model = RandomForestClassifier(), X_train = your_actual_X_train, y_train = your_actual_y_train
# mean_score = cross_val(model, X_train, y_train)
# print(mean_score)




import pandas as pd

def df_csv(X_test, y_test, y_pred):
    """
    Create a DataFrame with X_test, y_true, and y_pred.

    Parameters:
    - X_test: Features DataFrame (assuming X has column names)
    - y_test: True labels
    - y_pred: Predicted labels

    Returns:
    - result_df: DataFrame containing X_test, y_true, and y_pred
    """
    # Assuming X_test is a DataFrame with column names
    X_test_df = pd.DataFrame(X_test, columns=X_test.columns)

    # Reset the index of y_test
    y_test = y_test.reset_index(drop=True)

    # Create DataFrames for y_true and y_pred
    y_true_df = pd.DataFrame({'y_true': y_test})
    y_pred_df = pd.DataFrame({'y_pred': y_pred})

    # Concatenate the DataFrames horizontally
    result_df = pd.concat([X_test_df, y_true_df, y_pred_df], axis=1)

    return result_df

# Example Usage:
# result_df2 = df_csv(X_test_scaled_df, y_test, y_pred)
# print(result_df2)



import pandas as pd
from sklearn.preprocessing import StandardScaler

def scaling_df(df, fill_value=0):
    """
    Preprocess a DataFrame by scaling all numeric columns and filling/replacing NaN values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - fill_value (int or float): Value to fill NaN entries in the DataFrame.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """

    # Identify numeric columns for scaling
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Check if there are any numeric columns to scale
    if not numeric_columns.empty:
        # Scaling Data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_columns])

        # Create a DataFrame from the scaled data with appropriate column labels
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_columns)

        # Filling/Replacing NaN values with the specified fill value
        df_scaled = df_scaled.fillna(fill_value)

        return df_scaled
    else:
        # If no numeric columns, return the original DataFrame
        return df

# Example usage:
# Assuming 'your_dataframe' is your actual DataFrame and 'your_fill_value' is the fill value.
#your_dataframe_scaled = scaling_dataframe(your_dataframe, #fill_value=your_fill_value)



from scipy.stats import pearsonr

def perform_pearson_correlation_test(dataframe, variable1_name, variable2_name, alpha=0.05):
    """
    Perform a Pearson correlation test between two continuous variables.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - variable1_name (str): Name of the first continuous variable.
    - variable2_name (str): Name of the second continuous variable.
    - alpha (float): Significance level (default is 0.05).

    Returns:
    - tuple: Tuple containing the correlation coefficient and the p-value.
    - str: Result of the significance test.
    """

    # Extract the variables from the DataFrame
    variable1 = dataframe[variable1_name]
    variable2 = dataframe[variable2_name]

    # Perform Pearson correlation test
    correlation_coefficient, p_value = pearsonr(variable1, variable2)

    # Check if the p-value is less than alpha
    if p_value < alpha:
        result = f"There is a significant correlation (p-value: {p_value:.4f}, alpha: {alpha})"
    else:
        result = f"There is no significant correlation (p-value: {p_value:.4f}, alpha: {alpha})"

    return (correlation_coefficient, p_value), result

# Example usage:
# Assuming 'your_dataframe' is your actual DataFrame and 'variable1'/'variable2' are your column names
#result, test_result = perform_pearson_correlation_test(your_dataframe, 'variable1', 'variable2')

#Display the results
#print(f"Correlation Coefficient: {result[0]:.4f}")

