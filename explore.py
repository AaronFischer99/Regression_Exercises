import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df):
    """
    Plot all pairwise relationships along with the regression line for each pair.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - None (displays plots).
    """

    # Create a pair plot with regression lines
    sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})

    # Display the plots
    plt.show()
    
    
    
    

import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_vars(df, cat_columns):
    """
    Plot three different visualizations for each categorical variable with continuous variables.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cat_columns (list): List of categorical column names.

    Returns:
    - None (displays plots).
    """

    # Set the style for seaborn plots
    sns.set(style="whitegrid")

    # Iterate through each categorical column
    for cat_column in cat_columns:

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # Plot 1: Boxplot
        sns.boxplot(x=cat_column, y='calculatedfinishedsquarefeet', data=df, ax=axes[0])
        axes[0].set_title(f'Boxplot of {cat_column} vs. Calculated Finished Square Feet')
        axes[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Plot 2: Violinplot
        sns.violinplot(x=cat_column, y='taxvaluedollarcnt', data=df, ax=axes[1])
        axes[1].set_title(f'Violinplot of {cat_column} vs. Tax Value Dollar Count')
        axes[1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Plot 3: Scatterplot
        sns.scatterplot(x=cat_column, y='taxamount', data=df, ax=axes[2])
        axes[2].set_title(f'Scatterplot of {cat_column} vs. Tax Amount')
        axes[2].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Adjust layout
        plt.tight_layout()
        
        

        # Display the plots
        plt.show()


        
        
        
        
import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_and_continuous_vars(df, cat_columns, cont_columns):
    """
    Plot three different plots for visualizing the relationship between a categorical variable and a continuous variable.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cat_columns (list): List of categorical column names.
    - cont_columns (list): List of continuous column names.

    Returns:
    - None (displays plots).
    """

    # Loop through each categorical column
    for cat_col in cat_columns:

        # Loop through each continuous column
        for cont_col in cont_columns:

            # Create a box plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=cat_col, y=cont_col, data=df)
            plt.title(f'{cont_col} by {cat_col}')
            plt.show()

            # Create a violin plot
            plt.figure(figsize=(12, 6))
            sns.violinplot(x=cat_col, y=cont_col, data=df)
            plt.title(f'{cont_col} by {cat_col}')
            plt.show()

            # Create a swarm plot
            plt.figure(figsize=(12, 6))
            sns.swarmplot(x=cat_col, y=cont_col, data=df)
            plt.title(f'{cont_col} by {cat_col}')
            plt.show()



            
            
import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_vars2(df, cat_columns):
    """
    Plot three different visualizations for each categorical variable with continuous variables.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cat_columns (list): List of categorical column names.

    Returns:
    - None (displays plots).
    """

    # Set the style for seaborn plots
    sns.set(style="whitegrid")

    # Iterate through each categorical column
    for cat_column in cat_columns:

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # Plot 1: Boxplot
        sns.boxplot(x=cat_column, y='calculatedfinishedsquarefeet', data=df, ax=axes[0])
        axes[0].set_title(f'Boxplot of {cat_column} vs. Calculated Finished Square Feet')
        axes[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Plot 2: Violinplot
        sns.violinplot(x=cat_column, y='taxvaluedollarcnt', data=df, ax=axes[1])
        axes[1].set_title(f'Violinplot of {cat_column} vs. Tax Value Dollar Count')
        axes[1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Plot 3: Scatterplot
        sns.scatterplot(x=cat_column, y='taxamount', data=df, ax=axes[2])
        axes[2].set_title(f'Scatterplot of {cat_column} vs. Tax Amount')
        axes[2].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Adjust layout
        plt.tight_layout()
        
        

        # Display the plots
        plt.show()

# Example usage:
plot_categorical_vars(train.sample(n=1000), cat_columns)




import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df):
    """
    Plot all pairwise relationships along with the regression line for each pair.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - None (displays plots).
    """

    # Create a pair plot with regression lines
    sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'red'}}, corner=True)

    # Display the plots
    plt.show()

