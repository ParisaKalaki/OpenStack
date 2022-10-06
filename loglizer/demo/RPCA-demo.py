#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

from loglizer.models.RPCA import R_pca
from loglizer import dataloader, preprocessing


       
struct_log = '../data/OpenStack/20k+4k destroy+10k undefined +4k dhcp.log_structured.csv' # The structured log file
label_file = '../data/OpenStack/20k+4k destroy + 10k undefiend+4k dhcp_label.csv' # The anomaly label file


if __name__ == '__main__':
     
    (x_train, y_train), (x_test, y_test), DataFrame = dataloader.load_OpenStack(struct_log,
                                                            label_file=label_file,
                                                            window='session', 
                                                            train_ratio=0.5,
                                                            split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf'
                                              )
    x_test = feature_extractor.transform(x_test)
   
    model = R_pca(x_train, threshold=0.8)
    L, S = model.fit2(max_iter=600, iter_print=100)
   
    print('Train validation:')
    precision, recall, f1, accuracy, y_pred = model.evaluate1(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1, accuracy, y_pred = model.evaluate2(x_test, y_test)
  
