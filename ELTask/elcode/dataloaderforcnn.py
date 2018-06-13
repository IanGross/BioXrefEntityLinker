import pickle
import numpy as np
import nltk

def combine_data():
	all_sentdata = []
	all_ontdata = []
	all_truthdata = []
	#Over different batch files
	for batch_i in range(1,5):    
		sent_bf = open("../dumps/wordRep" + str(batch_i) + "train.pkl","rb")
		sent_batchdata = pickle.load(sent_bf,encoding='latin1')
		print("Sent shape ",sent_batchdata.shape)
		sent_bf.close()
		ont_bf = open("../dumps/ontRep" + str(batch_i) + "train.pkl","rb")
		ont_batchdata = pickle.load(ont_bf, encoding = 'latin1')
		ont_bf.close()
		truth_bf = open("../dumps/truth" + str(batch_i) + "train.pkl","rb")
		truth_batchdata = pickle.load(truth_bf)
		print("Truth shape ",truth_batchdata.shape)
		truth_bf.close()
		if batch_i == 1:
			all_sentdata = sent_batchdata
			all_ontdata = ont_batchdata
			all_truthdata = truth_batchdata
		else:
			all_sentdata = np.concatenate((all_sentdata, sent_batchdata))
			all_ontdata = np.concatenate((all_ontdata, ont_batchdata))
			all_truthdata = np.concatenate((all_truthdata, truth_batchdata))

	print(" Shape of sentence data ", all_sentdata.shape)
	print(" Shape of ont data ",all_ontdata.shape)
	print(" Shape of truth data ",all_truthdata.shape)
	return (all_sentdata, all_ontdata, all_truthdata)

def combine_alldata():
	all_sentdata = []
	all_ontdata = []
	all_truthdata = []
	#Over different batch files
	for batch_i in range(1,5):    
		sent_bf = open("../dumps/wordRep" + str(batch_i) + ".pkl","rb")
		sent_batchdata = pickle.load(sent_bf,encoding='latin1')
		print("Sent shape ",sent_batchdata.shape)
		sent_bf.close()
		ont_bf = open("../dumps/ontRep" + str(batch_i) + ".pkl","rb")
		ont_batchdata = pickle.load(ont_bf, encoding = 'latin1')
		ont_bf.close()
		truth_bf = open("../dumps/truth" + str(batch_i) + ".pkl","rb")
		truth_batchdata = pickle.load(truth_bf)
		print("Truth shape ",truth_batchdata.shape)
		truth_bf.close()
		if batch_i == 1:
			all_sentdata = sent_batchdata
			all_ontdata = ont_batchdata
			all_truthdata = truth_batchdata
		else:
			all_sentdata = np.concatenate((all_sentdata, sent_batchdata))
			all_ontdata = np.concatenate((all_ontdata, ont_batchdata))
			all_truthdata = np.concatenate((all_truthdata, truth_batchdata))

	print(" Shape of sentence data ", all_sentdata.shape)
	print(" Shape of ont data ",all_ontdata.shape)
	print(" Shape of truth data ",all_truthdata.shape)
	return (all_sentdata, all_ontdata, all_truthdata)
