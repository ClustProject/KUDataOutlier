# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:50:22 2021

@author: korea
"""

import numpy as np
import matplotlib.pyplot as plt
import sranodec as anom
import argparse

class Main():
    
    def __init__(self,config):
        self.test_signal = np.concatenate(
            [np.random.normal(0.7, 0.05, 300), 
             np.random.normal(0.8, 0.05, 300),
             np.random.normal(0.6, 0.05, 300),
             np.random.normal(0.9, 0.05, 300)])
        
        self.window = config['window']
        self.score_window = config['score_window']
                
        
    def draw_plot(self) :
        plt.plot(self.test_signal, alpha=0.5, label="observation")
        
    def spectral(self,window,score_window,signal) :
                
        spec = anom.Silency(window, window, score_window)
        score = spec.generate_anomaly_score(signal)
        
        return score
    
    def run(self) :
        score = self.spectral(self.window,self.score_window,self.test_signal)
        plt.plot(self.test_signal, alpha=0.5, label="observation")
        index_changes = np.where(score > np.percentile(score, 99))[0]
        plt.scatter(index_changes, self.test_signal[index_changes], c='green', label="change point")
        plt.savefig('./anomaly_detected_signal')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-window', help='spectral window', type=int, default=24)
    parser.add_argument('-score_window', help='score window', type=int, default=100)
    args = parser.parse_args()
    
    config = {'window' : args.window,
              'score_window' : args.score_window}
    
    main = Main(config)
    main.run()
