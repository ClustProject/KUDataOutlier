import os
import argparse
from utils import *
from sklearn.ensemble import IsolationForest


# Isolation Forest
def iforest(X_train, X_test):
    # IForest 모델 적합
    if_model = IsolationForest(random_state=42)
    if_model.fit(X_train.values)

    if_test = - 1.0 * if_model.score_samples(X_test.values)
    if_test = pd.DataFrame(if_test, index=X_test.index, columns=['score'])
    return if_test


def main(train_data_path, test_data_path, save_root_path):
    os.makedirs(save_root_path, exist_ok=True)

    X_train = pd.read_csv(train_data_path, index_col=0)
    X_train.index = pd.to_datetime(X_train.index)

    X_test = pd.read_csv(test_data_path, index_col=0)
    X_test.index = pd.to_datetime(X_test.index)

    test_scores = iforest(X_train, X_test)
    test_scores.to_csv(os.path.join(save_root_path, 'anomaly_score.csv'))

    draw_plot(test_scores, save_path=os.path.join(save_root_path, 'anomaly_score_plot.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=str, default='./data/nasa_bearing_train.csv')
    parser.add_argument('--test-data-path', type=str, default='./data/nasa_bearing_test.csv')
    parser.add_argument('--save-root-path', type=str, default='./result/iforest')

    args, _ = parser.parse_known_args()

    main(train_data_path=args.train_data_path,
         test_data_path=args.test_data_path,
         save_root_path=args.save_root_path)
