import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Standard scaling (mean=0, variance=1) 
def normalization(X_train, X_test):
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled


# anomaly score plot
def draw_plot(scores, save_path):
    plt.figure(figsize=(12, 5))
    plt.scatter(scores.index, scores['score'], c='blue', s=3)

    plt.xlabel('Date')
    plt.ylabel('Anomaly Score')
    plt.xlim(scores.index[0], scores.index[-1])
    
    plt.savefig(save_path)
    plt.close()
