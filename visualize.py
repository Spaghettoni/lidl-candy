import matplotlib.pyplot as plt


def visualize_pca(data_frame, title, targets, colors, target_name):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title)

    for target, color in zip(targets, colors):
        indicesToKeep = data_frame[target_name] == target
        ax.scatter(data_frame.loc[indicesToKeep, 'principal component 1']
                   , data_frame.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.tight_layout()
    plt.show()