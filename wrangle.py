import pandas as pd
from prepare import train_test_validate2

def wrangle_zillow():
    """
    Perform data wrangling on Zillow dataset:
    1. Query relevant columns from the database.
    2. Save the data to a CSV file.
    3. Read the CSV file into a DataFrame.
    4. Check for and handle null values.
    5. Split the data into train, test, train_val, and test_val sets.

    Returns:
    - tuple: (train, test, train_val, test_val)
    """

    # Step 1: Query relevant columns from the database
    zillow_query = """
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
           taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM propertylandusetype
    LEFT JOIN properties_2017 USING(propertylandusetypeid)
    WHERE propertylandusedesc = "Single Family Residential";
    """

    # Reading the data
    data = pd.read_sql(zillow_query, env4.get_db_url('zillow'))
    
    # Step 2: Save the data to a CSV file
    data.to_csv('zillow.csv')

    # Step 3: Read the CSV file into a DataFrame
    data = pd.read_csv('zillow.csv', index_col=0)

    # Step 4: Check for and handle null values
    data = data.dropna()

    # Step 5: Split the data into train, test, train_val, and test_val sets
    train, test, train_val, test_val = train_test_validate2(data)

    return train, test, train_val, test_val



import pandas as pd

def convert_to_integer(dataframe, columns):
    """
    Convert specified columns in a DataFrame to integers.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - columns (str or list): Column(s) to convert to integers.

    Returns:
    - pd.DataFrame: DataFrame with specified columns converted to integers.
    """

    # Ensure 'columns' is a list
    if not isinstance(columns, list):
        columns = [columns]

    # Convert specified columns to integers
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce', downcast='integer')

    return dataframe


