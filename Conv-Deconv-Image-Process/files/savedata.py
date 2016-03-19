'''
Created on Mar 10, 2016

@author: Wuga
'''
import files as F
import cPickle
import os

def saveDataPickle():
    print('... Loading Training Images')
    train_set_x,train_rotate = F.loadImage('data/images/train')
    print('... Loading Training Tags')
    #train_set_y_flat=F.loadGroundTruth('data/groundTruth/train_label_flat.txt')
    #train_set_y=F.reshapeLables(train_set_y_flat, train_rotate)
    train_set_y,_=F.loadCSV('data/groundTruth/train')
    
    print('... Loading Training Images')
    valid_set_x,valid_rotate = F.loadImage('data/images/val')
    print('... Loading Training Tags')
    #valid_set_y_flat=F.loadGroundTruth('data/groundTruth/val_label_flat.txt')
    #valid_set_y=F.reshapeLables(valid_set_y_flat, valid_rotate)
    valid_set_y,_=F.loadCSV('data/groundTruth/val')
    
    print('... Loading Training Images')
    test_set_x,test_rotate = F.loadImage('data/images/test')
    print('... Loading Training Tags')
    #test_set_y_flat=F.loadGroundTruth('data/groundTruth/test_label_flat.txt')
    #test_set_y=F.reshapeLables(test_set_y_flat, test_rotate)
    test_set_y,_=F.loadCSV('data/groundTruth/test')
    
    datapath='data/data.save'
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir,'..',datapath)
    f = open(abs_data_path, 'wb')
    cPickle.dump(((train_set_x,train_set_y), (valid_set_x,valid_set_y), (test_set_x,test_set_y)), f, -1)
    f.close()
    
saveDataPickle()
    
    