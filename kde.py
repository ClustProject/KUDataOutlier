import os
import argparse
from utils import *
from sklearn.neighbors import KernelDensity


# Kernel Density Estimation
def kde(X_train, X_test):
    # KDE 기반 분포 추정
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde_model.fit(X_train.values)

    kde_test = - 1.0 * kde_model.score_samples(X_test.values)
    kde_test = pd.DataFrame(kde_test, index=X_test.index, columns=['score'])
    return kde_test


def main(train_data_path, test_data_path, save_root_path):
    os.makedirs(save_root_path, exist_ok=True)

    X_train = pd.read_csv(train_data_path, index_col=0)
    X_train.index = pd.to_datetime(X_train.index)

    X_test = pd.read_csv(test_data_path, index_col=0)
    X_test.index = pd.to_datetime(X_test.index)

    test_scores = kde(X_train, X_test)
    test_scores.to_csv(os.path.join(save_root_path, 'anomaly_score.csv'))

    draw_plot(test_scores, save_path=os.path.join(save_root_path, 'anomaly_score_plot.jpg'))


if __name__ == '__main__':
    
    train_data_path= './data/nasa_bearing_train.csv'
    test_data_path= './data/nasa_bearing_test.csv'
    save_root_path= './result/kde'

    main(train_data_path= train_data_path, test_data_path=test_data_path, save_root_path=save_root_path)
