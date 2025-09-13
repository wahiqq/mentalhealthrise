import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Stress_Dataset.csv')

# Show basic info
def basic_info(df):
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('Missing values per column:')
    print(df.isnull().sum())
    print('\nSample rows:')
    print(df.head())

# Encode categorical variables
def encode_categoricals(df):
    df_encoded = df.copy()
    # Example: Encode Gender
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender'] = df_encoded['Gender'].astype('category').cat.codes
    # Encode target variable
    if 'Which type of stress do you primarily experience?' in df_encoded.columns:
        df_encoded['StressType'] = df_encoded['Which type of stress do you primarily experience?'].astype('category').cat.codes
    return df_encoded

# Visualize target distribution
def plot_target_distribution(df):
    plt.figure(figsize=(8,4))
    sns.countplot(y='Which type of stress do you primarily experience?', data=df)
    plt.title('Target Distribution')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    basic_info(df)
    plot_target_distribution(df)
    df_encoded = encode_categoricals(df)
    print('\nEncoded sample:')
    print(df_encoded.head())
