import pandas as pd
from sklearn.preprocessing import StandardScaler
from linear_regression import compute_regression
from pca import compute_pca

pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 50)

if __name__ == "__main__":
    df = pd.read_csv('./candy-data.csv')
    binary_features = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
    continuous_features = ['sugarpercent', 'pricepercent']

    features = binary_features + continuous_features
    std_values = StandardScaler().fit_transform(df.loc[:, continuous_features].values)
    std_frame = pd.DataFrame(data=std_values, columns=continuous_features)
    x = pd.concat([df.loc[:, binary_features], std_frame], axis=1)
    x_binary_only = df.loc[:, binary_features]
    y = df.loc[:, ['winpercent']].values

    compute_pca(x, None, y)
    compute_pca(x_binary_only, None, y)
    compute_pca(x, 2, y)

    compute_regression(x, y)

