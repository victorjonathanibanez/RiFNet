#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# RiFNet testing code 
# Author: Victor Ibanez, University of Zurich, Institute of Forensic Medicine
# Contact: victor.ibanez@uzh.ch
# -------------------------------------------------------------------------------------

# import libraries

import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# On mac you need to shut this down

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# create statistic lists

acc_list = []
prec_list = []
rec_list = []
F1_list = []

# start iteration

for it in range (1,6):
	path = 'models_CV_' + str(it) + '/'
	for fold in range(1,6):
		name = path + 'RiFNet_CV_' + str(it) '_run_' + str(fold) + '.h5'

		#load model

		loaded_model = tf.keras.models.load_model(name)
		loaded_model.layers[0].input_shape 

		#input dimensions
		# batch, height, width, depth

		batch,rows,cols,depth = loaded_model.layers[0].input_shape

		#define img directory

		img_dir = 'test_CV_' + str(it) + '/'

		#find all .jpg files and store them randomly in list

		file_folder = []
		for filename in glob.iglob(img_dir + '/*/*.jpg', recursive=True):
		    file_folder.append(filename)
		    random.shuffle(file_folder)

		#extract batch length

		batch_length = len(file_folder)

		#create batch holder

		batch_holder = np.zeros((batch_length, rows, cols, depth))

		#iterate through list and fill in batch holder and real label

		real_label_holder = []
		for i in range(len(file_folder)):
		    file_path = file_folder[i]
		    img = image.load_img(file_path)
		    real_label_holder.append(os.path.basename(file_path)[:-4])
		    batch_holder[i, :] = img

		real_label = np.zeros((len(real_label_holder),1), dtype=int)
		for i in range(len(real_label_holder)):
		  if real_label_holder[i][0] == 'n':
		    real_label[i:i+1] = 0
		  else:
		    real_label[i:i+1] = 1

		#feed data in to model, make prediction

		result=loaded_model.predict_classes(batch_holder)

		#get label name

		def get_label_name(number):
		  if number == 0:
		      return 'no_fracture'
		  else:
		      return 'fracture'

		#append confusions items

		cnt_f = 0
		cnt_t = 0
		confusion_items = []
		confusion = np.equal(real_label, result)
		for i in range(len(confusion)):
		    if confusion[i] == False:
		        cnt_f += 1
		        confusion_items.append(i)
		    else:
		        cnt_t += 1
		
		# calculate accuracy, precision, recall and F1-score values

		TP = 0
		TN = 0
		FP = 0
		FN = 0
		for i, j in np.nditer([real_label, result]):
			if i == j and i == 1:
				TP += 1
			if i == j and i == 0:
				TN += 1
			if i != j and i == 1:
				FN += 1
			elif i != j and i == 0:
				FP += 1

		acc = (TP+TN)/(TP+FP+FN+TN)
		prec = TP/(TP+FP)
		rec = TP/(TP+FN)
		F1 = (2*(rec*prec))/(rec+prec)

		print('accuracy: ', acc, 'precision: ', prec, 'recall: ', rec, 'F1-score: ', F1)
		
		print('wrong classification:',  cnt_f, '/', len(real_label), 'CA:', round(cnt_t/len(real_label),4)*100, '%')
		

		# plot confusion images

		fig = plt.figure(figsize=(20,10))
		for i in range(len(confusion_items)):
		    fig.add_subplot(np.ceil(len(confusion_items)/2),np.ceil(len(confusion_items)/2), i+1)
		    plt.title(real_label_holder[confusion_items[i]] + ' class: ' + get_label_name(result[confusion_items[i]][0]))
		    fig.suptitle('wrong classification: ' +  str(cnt_f) + '/' + str(len(real_label)) + ' // CA: ' + str(round(cnt_t/len(real_label),4)) + '%', fontsize=16)
		    plt.axis('off')
		    plt.imshow(batch_holder[confusion_items[i]]/256.)

		conf_name = img_dir + 'plt_confusion_' + 'CV_' + str(it) + '_' + str(fold) + '.png'
		plt.savefig(conf_name)
		plt.close()


		# append values to corresponding statistic lists

		acc_list.append(round(acc,4)*100)
		prec_list.append(round(prec,4)*100)
		rec_list.append(round(rec,4)*100)
		F1_list.append(round(F1,4)*100)

print('mean: ', np.mean(acc_list), 'std: ', np.std(acc_list))

# write a text file with statistics

cnt = 0
f = open("pred_stats", "w+")
f.write("Prediction statistics" + '\n' + '\n')
for s,t,u,v in zip(acc_list, prec_list, rec_list, F1_list):
	cnt += 1
	f.write('\n' + 'run_# ' + str(cnt) + '\n' + 'acc: ' + str(s) + '%' + '\n' + 'precision: ' + str(t) + '%' + '\n' + 'recall: ' + str(u) + '%' + '\n' + 'F1-score: ' + str(v) + '%' + '\n')
f.write('\n' + "mean_prediction_accuracy: " + str(np.mean(acc_list)) + '%' + '\n')
f.write('\n' + "mean_precision: " + str(np.mean(prec_list)) + '%' + '\n')
f.write('\n' + "mean_recall: " + str(np.mean(rec_list)) + '%' + '\n')
f.write('\n' + "mean_F1-score: " + str(np.mean(F1_list)) + '%' + '\n')
f.write('\n' + "std_accuracy: " + str(np.std(acc_list)) + '%' + '\n')
f.write('\n' + "std_precision: " + str(np.std(prec_list)) + '%' + '\n')
f.write('\n' + "std_recall: " + str(np.std(rec_list)) + '%' + '\n')
f.write('\n' + "std_F1-score: " + str(np.std(F1_list)) + '%' + '\n')
f.close()


