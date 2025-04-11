import pandas as pd

def data_cleaning_single(df, type, inp_int, target):
    input_df = pd.DataFrame(df)
    if type == "mean":
        # Fill missing values with mean
        input_df[target].fillna(input_df[target].mean(), inplace=True)
    elif type == "median":
        # Fill missing values with median
        input_df[target].fillna(input_df[target].median(), inplace=True)
    elif type == "mode":
        # Fill missing values with mode
        input_df[target].fillna(input_df[target].mode().iloc[0], inplace=True)
    elif type == "drop":
        # Drop rows with missing values
        input_df[target].dropna(inplace=True)
    elif type == "custom":
        # Custom logic for filling missing values
        input_df[target].fillna(inp_int, inplace=True)
    elif type == "zero":
        # Fill missing values with zero
        input_df[target].fillna(0, inplace=True)

    df[target] = input_df[target]
    return df
