import matplotlib.pyplot as plt
from sklearn import manifold
Bshape = {"0": "L", "1": "Z", "2": "O", "3": "Y", "4": "I",
          "5": "T", "6": "E", "7": "U", "8": "F", "9": "H"}
BshapeXY = {"0": [[], []], "1": [[], []], "2": [[], []], "3": [[], []], "4": [[], []],
            "5": [[], []], "6": [[], []], "7": [[], []], "8": [[], []], "9": [[], []]}

def TSE_visual(encoding, labels):
    print("--- T-SNE ---")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(encoding)
    print("Org data dimension is {}. Embedded data dimension is {}".format(encoding.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    for i in range(X_norm.shape[0]):
        BshapeXY[str(labels[i])][0].append(X_norm[i, 0])
        BshapeXY[str(labels[i])][1].append(X_norm[i, 1])

    plt.figure(figsize=(8, 8))
    for key in BshapeXY:
        index = int(key)
        x = BshapeXY[key][0]
        y = BshapeXY[key][1]
        plt.plot(x[0], y[0], 'o', color=plt.cm.tab10(index), label=Bshape[key])
        for i in range(1, len(x)):
            plt.plot(x[i], y[i], 'o', color=plt.cm.tab10(index))

    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()