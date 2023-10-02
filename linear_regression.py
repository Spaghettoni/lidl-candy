from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def compute_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    importance = model.coef_[0]
    print(importance)
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks([i for i in range(len(x.columns))], x.columns, rotation=90)
    plt.tight_layout()
    plt.show()