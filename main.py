import numpy as np
import matplotlib.pyplot as plt
from LojisticRegression import LogRes
from CleaningData import Data

def show(X, y, classifier):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    hs = .02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hs), np.arange(y_min, y_max, hs))
    Z = classifier.test(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z,cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("Logistic regression")
    plt.xlabel('1 st exam')
    plt.ylabel('2 nd exam')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == '__main__':
    exam_data = Data('PassingTheExam.csv')
    exam_data.CleaningData()
    x_data, y_data, admitted, not_admitted = exam_data.ReturnData()

    train_shape = (x_data.shape[0] * 70) // 100  # the first 70 data will be trained
    x_train, y_train = x_data[:train_shape], y_data[:train_shape]

    x_test, y_test = x_data[train_shape:], y_data[train_shape:]
    classifier = LogRes(x_train.shape[1])

    x_train, y_train = classifier.prepare_data(x_train, y_train)
    x_test, y_test = classifier.prepare_data(x_test, y_test)

    classifier.train(x_train, y_train, n_epochs=1000000)
    predictions_test = classifier.test(x_test)

    acc_score = (predictions_test == y_test).sum() / len(predictions_test)
    print("Lojistic score-->", acc_score)

    show(x_data, y_data, classifier)