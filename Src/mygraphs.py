import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


def boxplots(df, num_columns):
    warnings.filterwarnings('ignore', module='matplotlib.*')
    graph_df = df.select_dtypes(exclude=['object']).copy()
    num_plots = len(graph_df.columns)
    num_rows = int(np.ceil(num_plots / num_columns))

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows*5))

    # Reshape axs to 2D if num_rows or num_columns is 1
    if num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    for i, column in enumerate(graph_df.columns):
        if graph_df[column].dtype != 'object':
            row = i // num_columns
            col = i % num_columns
            axs[row, col].boxplot(graph_df[column])
            axs[row, col].set_title(f'Boxplot of {column}')

    plt.tight_layout()
    plt.show()


def histograms(df, num_columns):
    warnings.filterwarnings('ignore', module='matplotlib.*')
    graph_df = df.select_dtypes(exclude=['object']).copy()
    num_plots = len(graph_df.columns)
    num_rows = int(np.ceil(num_plots / num_columns))

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows*5))

    # Reshape axs to 2D if num_rows or num_columns is 1
    if num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    for i, column in enumerate(graph_df.columns):
        if True: #graph_df[column].dtype != 'object':
            row = i // num_columns
            col = i % num_columns
            axs[row, col].hist(graph_df[column], bins=30, color='skyblue', edgecolor='black')
            axs[row, col].set_title(f'Histogram of {column}')

    plt.tight_layout()
    plt.show()


def graphs(df, num_columns):
    warnings.filterwarnings('ignore', module='matplotlib.*')
    graph_df = df.copy()
    num_plots = len(graph_df.columns)
    num_rows = int(np.ceil(num_plots / num_columns))

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows*5))

    # Reshape axs to 2D if num_rows or num_columns is 1
    if num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    for i, column in enumerate(graph_df.columns):
        row = i // num_columns
        col = i % num_columns
        if np.issubdtype(graph_df[column].dtype, np.number):
            # If the column is numeric, create a histogram
            axs[row, col].hist(graph_df[column], bins=30, color='skyblue', edgecolor='black')
            axs[row, col].set_title(f'Histogram of {column}')
        else:
            # If the column is not numeric (i.e., it's an object), create a bar graph
            graph_df[column].value_counts().plot(kind='bar', ax=axs[row, col])
            axs[row, col].set_title(f'Bar Graph of {column}')

    plt.tight_layout()
    plt.show()

def boxplots2(df, num_columns):
    warnings.filterwarnings('ignore', module='matplotlib.*')
    numeric_df = df.select_dtypes(include=[np.number])
    object_df = df.select_dtypes(include=['object'])
    
    num_plots = len(numeric_df.columns) * len(object_df.columns)
    num_rows = int(np.ceil(num_plots / num_columns))

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows*5))

    # Reshape axs to 2D if num_rows or num_columns is 1
    if num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    plot_index = 0
    for num_column in numeric_df.columns:
        for obj_column in object_df.columns:
            row = plot_index // num_columns
            col = plot_index % num_columns
            sns.boxplot(x=obj_column, y=num_column, data=df, ax=axs[row, col])
            axs[row, col].set_title(f'Boxplot of {num_column} by {obj_column}')
            plot_index += 1

    plt.tight_layout()
    plt.show()


def histogram_top_n_tokens(df_tokenized, n, text_col_name, target_col_name):
    '''
    Parameters: freq_dist - a FreqDist object
                n - an integer
                target_col_name - a string
    Returns: None

    Create a bar plot of the top n words in freq_dist
    '''
    from nltk import FreqDist
    # create a list of unique classes in the target column
    classes = df_tokenized[target_col_name].unique()

    # Create a FreqDist object for each class
    freq_dists = {cls: FreqDist(df_tokenized[df_tokenized[target_col_name] == 
                                             cls][text_col_name].explode()) for cls in classes}
    
    # Create a list of the top n words for each class
    top_n = {cls: list(zip(*freq_dists[cls].most_common(n))) for cls in classes}

    # Create a separate plot for each class, tightly packed together
    fig, ax = plt.subplots(1, len(classes), figsize=(15, 5), sharey=True)
    for i, cls in enumerate(classes):
        # Create a bar plot for the class
        ax[i].bar(top_n[cls][0], top_n[cls][1])
        # Add a title to the plot
        ax[i].set_title(f'Top {n} Tokens for Target Class={cls}')
        # Add labels to the plot
        ax[i].set_xlabel('Word', labelpad=20)
        ax[i].set_ylabel('Count')
        # Rotate the x-axis labels
        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust the layout to make room for the x-axis labels
    # plt.subplots_adjust(bottom=0.5)
    plt.show()
    

