import numpy as np
import pickle
import tensorflow as tf
import sys


'''
Three files need to be shuffled
Load all of them and shuffle them, then choose 3134 from each 
And write back
wordRep, ontRep, truths
Need to choose 1880 samples from each for 85:15 split
'''
def shuffle_data():
	lens = [12534, 12534, 12534, 12537]

	for i in range(1,5):
		x_i = np.arange(lens[i - 1])
		np.random.shuffle(x_i)

		batchword_data = pickle.load(open("../dumps/wordRep" + str(i) + ".pkl","rb"),encoding='latin1')
		truth_data = pickle.load(open("../dumps/truth" + str(i) + ".pkl","rb"),encoding='latin1')
		ontrep_data = pickle.load(open("../dumps/ontRep" + str(i) + ".pkl","rb"),encoding='latin1')

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
	lens = [12533, 12533, 12533, 12540]
	#Go through the 4 batches
	for i in range(1,5):
		batchword_data = pickle.load(open("../dumps/wordRep" + str(i) + ".pkl","rb"),encoding='latin1')
		truth_data = pickle.load(open("../dumps/truth" + str(i) + ".pkl","rb"),encoding='latin1')
		ontrep_data = pickle.load(open("../dumps/ontRep" + str(i) + ".pkl","rb"),encoding='latin1')
		#Normalising data
		batchword_data -= np.mean(batchword_data, axis=(1, 2), keepdims=True)
		batchword_data /= np.std(batchword_data, axis=(1,2), keepdims=True)

		ontrep_data -= np.mean(ontrep_data, axis=(1, 2), keepdims=True)
		ontrep_data /= np.std(ontrep_data, axis=(1,2), keepdims=True)

		#Shuffling the data for the train and test, and writing them to separate files.
		shuffle=np.arange(1880)
		np.random.shuffle(shuffle)
		train_left = list(set([i for i in range(lens[i-1])]) - set(shuffle))
		#Defining the splits per batch
		train_wordbdata = batchword_data[train_left]
		train_y = truth_data[train_left]
		train_ontdata = ontrep_data[train_left]

		if i == 1:
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
			pickle.dump(truth_data, truthf)
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




