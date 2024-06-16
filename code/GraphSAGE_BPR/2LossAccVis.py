import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

if __name__ == '__main__':
    KK = 1
    Epoch = 401
    Epochs = np.linspace(1, Epoch - 1, Epoch - 1)

    # train_losses1 = np.loadtxt(f"GraphSAGE_TrainLoss_1.txt", dtype=np.float32)
    train_losses2 = np.loadtxt(f"GraphSAGE_TrainLoss_2.txt", dtype=np.float32)
    train_losses3 = np.loadtxt(f"GraphSAGE_TrainLoss_3.txt", dtype=np.float32)
    train_losses4 = np.loadtxt(f"GraphSAGE_TrainLoss_4.txt", dtype=np.float32)
    train_losses5 = np.loadtxt(f"GraphSAGE_TrainLoss_5.txt", dtype=np.float32)
    train_losses6 = np.loadtxt(f"GraphSAGE_TrainLoss_6.txt", dtype=np.float32)
    # plt.plot(Epochs, train_losses1, c="b", label="N=1")
    plt.plot(Epochs, train_losses2, c="g", label="N=2")
    plt.plot(Epochs, train_losses3, c="r", label="N=3")
    plt.plot(Epochs, train_losses4, c="b", label="N=4")
    plt.plot(Epochs, train_losses5, c="m", label="N=5")
    plt.plot(Epochs, train_losses6, c="y", label="N=6")
    plt.ylabel('Train Loss')
    plt.xlabel('Epochs')
    plt.grid(linestyle=":", color='gray')
    plt.legend()
    plt.show()

    # val_accs1 = np.loadtxt(f"GraphSAGE_ValAcc_1.txt", dtype=np.float32)
    val_accs2 = np.loadtxt(f"GraphSAGE_ValAcc_2.txt", dtype=np.float32)
    val_accs3 = np.loadtxt(f"GraphSAGE_ValAcc_3.txt", dtype=np.float32)
    val_accs4 = np.loadtxt(f"GraphSAGE_ValAcc_4.txt", dtype=np.float32)
    val_accs5 = np.loadtxt(f"GraphSAGE_ValAcc_5.txt", dtype=np.float32)
    val_accs6 = np.loadtxt(f"GraphSAGE_ValAcc_6.txt", dtype=np.float32)
    # plt.plot(Epochs, val_accs1, c="b", label="N=1")
    plt.plot(Epochs, val_accs2, c="g", label="N=2")
    plt.plot(Epochs, val_accs3, c="r", label="N=3")
    plt.plot(Epochs, val_accs4, c="b", label="N=4")
    plt.plot(Epochs, val_accs5, c="m", label="N=5")
    plt.plot(Epochs, val_accs6, c="y", label="N=6")
    plt.ylabel('Val Acc')
    plt.xlabel('Epochs')
    plt.grid(linestyle=":", color='gray')
    plt.legend()
    plt.show()