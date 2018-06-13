import numpy as np
import pickle
import tensorflow as tf
import sys
from dataloaderforcnn import combine_alldata


'''
Three files need to be shuffled
Load all of them and shuffle them, then choose 3134 from each 
And write back
wordRep, ontRep, truths
Need to choose 1880 samples from each for 85:15 split
'''
def shuffle_data():
	#lens = [12533, 12533, 12534, 12537]

	for i in range(1,5):
		

		batchword_data = pickle.load(open("../dumps/wordRep" + str(i) + ".pkl","rb"),encoding='latin1')
		truth_data = pickle.load(open("../dumps/truth" + str(i) + ".pkl","rb"),encoding='latin1')
		ontrep_data = pickle.load(open("../dumps/ontRep" + str(i) + ".pkl","rb"),encoding='latin1')

		x_i = np.arange(batchword_data.shape[0])
		np.random.shuffle(x_i)

		batchword_data = batchword_data[x_i]
		truth_data = truth_data[x_i]
		ontrep_data = ontrep_data[x_i]

		batchword_dataf = open("../dumps/wordRep" + str(i) + ".pkl","wb")
		truth_dataf = open("../dumps/truth" + str(i) + ".pkl","wb")
		ontrep_dataf = open("../dumps/ontRep" + str(i) + ".pkl","wb")

		pickle.dump(batchword_data, batchword_dataf)
		pickle.dump(truth_data,truth_dataf)
		pickle.dump(ontrep_data,ontrep_dataf)

		truth_dataf.close()
		batchword_dataf.close()
		ontrep_dataf.close()

		print(" Shuffled for ", i)

'''
Three files need to be shuffled
Load all of them and shuffle them, then choose 3134 from each 
And write back
wordRep, ontRep, truths
Need to choose 1880 samples from each for 85:15 split
'''
def pick_data(file_type):
	batches=[[],[],[],[]]
	file_name = ""
	if file_type == "sent":
		file_name="wordRep"
	elif file_type == "ont":
		file_name = "ontRep"
	elif file_type == "truth":
		file_name = "truth"
	#When i = 4, you just divide 12537/4
	for i in range(1,5):
		batch_data = pickle.load(open("../dumps/" + file_name +  str(i) + ".pkl","rb"),encoding='latin1')
		if i == 1:
			batches[0] = batch_data[0:3133]
			batches[1] = batch_data[3133:2*3133]
			batches[2] = batch_data[2*3133:3*3133]
			batches[3] = batch_data[3*3133:]
		elif i < 4:
			batches[0] = np.concatenate((batches[0],batch_data[0:3133]))
			batches[1] = np.concatenate((batches[1],batch_data[3133:2*3133]))
			batches[2] = np.concatenate((batches[2],batch_data[2*3133:3*3133]))
			batches[3] = np.concatenate((batches[3],batch_data[3*3133:]))
		elif i == 4:
			batches[0] = np.concatenate((batches[0],batch_data[0:3134]))
			batches[1] = np.concatenate((batches[1],batch_data[3134:2*3134]))
			batches[2] = np.concatenate((batches[2],batch_data[2*3134:3*3134]))
			batches[3] = np.concatenate((batches[3],batch_data[3*3134:]))
		lens = [len(batches[i]) for i in range(4)]
		print("Lens ",lens)
	for j in range(len(batches)):
		batch_f = open("../dumps/" + file_name + str(j+1) + ".pkl","wb")
		print("Dumping split data for file type ",file_type," for batch ", (j+1), " with size ", batches[j].shape)
		pickle.dump(batches[j],batch_f)
		batch_f.close()

'''
Three files need to be split
wordRep, ontRep, truths
Need to choose 1880 samples from each for 85:15 split
'''
def generate_data():
	test_y = []
	test_worddata  = []
	test_ontdata = []
	(all_sentdata,all_ontdata,all_truthdata) = combine_alldata()
	all_i = np.arange(all_sentdata.shape[0])
	np.random.shuffle(all_i)

	all_sentdata = all_sentdata[all_i]
	all_ontdata = all_ontdata[all_i]
	all_truthdata = all_truthdata[all_i]
	division_seed = int(all_sentdata.shape[0]/4)


	print(" Shuffled sent ", all_sentdata.shape, " Shuffled ont ",all_ontdata.shape, " Shuffled truth ", all_truthdata.shape, "Division seed ", division_seed)

	#Go through the  4 batches, that are induced inside
	for i in range(0,4):
		if i == 3:
			batchword_data = all_sentdata[i*division_seed:]
			truth_data = all_truthdata[i*division_seed:]
			ontrep_data = all_ontdata[i*division_seed:]
		else:
			batchword_data = all_sentdata[i*division_seed:(i+1)*division_seed]
			truth_data = all_truthdata[i*division_seed:(i+1)*division_seed]
			ontrep_data =all_ontdata[i*division_seed:(i+1)*division_seed]		
		#Normalising data
		batchword_data -= np.mean(batchword_data, axis=(1, 2), keepdims=True)
		batchword_data /= np.std(batchword_data, axis=(1,2), keepdims=True)

		ontrep_data -= np.mean(ontrep_data, axis=(1, 2), keepdims=True)
		ontrep_data /= np.std(ontrep_data, axis=(1,2), keepdims=True)

		print("Sent length ", batchword_data.shape, " Ont length ",ontrep_data.shape, " Truth length ",truth_data.shape)

		#Shuffling the data for the train and test, and writing them to separate files.
		len_batch = batchword_data.shape[0]
		len_testbatch = int(0.15*len_batch)
		shuffle=np.random.choice(len_batch, len_testbatch,replace=False)
		train_left = list(set([i for i in range(len_batch)]) - set(shuffle))
		#Defining the splits per batch
		train_wordbdata = batchword_data[train_left]
		train_y = truth_data[train_left]
		train_ontdata = ontrep_data[train_left]

		if i == 0:
			test_worddata = batchword_data[shuffle]
			test_y = truth_data[shuffle]
			test_ontdata = ontrep_data[shuffle]
		else:
			test_worddata = np.concatenate((test_worddata, batchword_data[shuffle]))
			test_y = np.concatenate((test_y,truth_data[shuffle]))
			test_ontdata = np.concatenate((test_ontdata, ontrep_data[shuffle]))


		with open("../dumps/wordRep" + str(i) + "train.pkl","wb") as wordf:
			pickle.dump(train_wordbdata, wordf)
			print(" Batch data shape for word ", train_wordbdata.shape)

		with open("../dumps/ontRep" + str(i) + "train.pkl","wb") as ontf:
			pickle.dump(train_ontdata, ontf)
			print(" Batch data shape for ont ", train_ontdata.shape)

		with open("../dumps/truth" + str(i) + "train.pkl","wb") as truthf:
			pickle.dump(train_y, truthf)
			print(" Batch data shape for truth ", train_y.shape)

	with open("../dumps/wordReptest.pkl","wb") as wordt:
		pickle.dump(test_worddata, wordt)
		print(" Test data shape for word ", test_worddata.shape)

	with open("../dumps/ontReptest.pkl","wb") as ontt:
		pickle.dump(test_ontdata, ontt)
		print(" Test data shape for ont ", test_ontdata.shape)

	with open("../dumps/truthtest.pkl","wb") as truthf:
		pickle.dump(test_y, truthf)
		print(" Test data shape for truth ", test_y.shape)

# shuffle_data()
# pick_data("sent")
# pick_data("ont")
# pick_data("truth")
generate_data()




