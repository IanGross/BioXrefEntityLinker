import pickle
from importlib import reload
import json
import numpy as np

with open("../dumps/labelsEmbedding.pkl", 'rb') as f:
    ontEmbDict = pickle.load(f, encoding='latin1') 

file_name = "testSet"
idLabelDict = json.load(open("../dumps/IDLabelMapping.json","rb"))
truths = pickle.load(open("../dumps/ground_truthstestSet.pkl","rb"))

emb_list = []

truth_np = []

#print("Keys ", idLabelDict.keys())
for truth in truths:
	associatedLabel = idLabelDict[str(truth)]
	associatedEmbedding = ontEmbDict[associatedLabel]
	emb_list.append(associatedEmbedding)

	truth_np1 =np.zeros(shape=(1827,1))
	truth_np1[truth - 1] = 1
	truth_np.append(truth_np1)	

truth_np = np.stack(truth_np,axis=0)
truth_np = truth_np.astype('float32')
ontMatrix = np.stack(emb_list,axis=0)
print(" Shape of ontMatrix ", ontMatrix.shape)

#Distribute if it is the entire batch, else just generate the test set
if file_name != "testSet":
	batch1 = ontMatrix[0:12534]
	batch2 = ontMatrix[12534:2*12534]
	batch3 = ontMatrix[12534*2:12534*3]
	batch4 = ontMatrix[12534*3:]

	with open("../dumps/ontRep1.pkl","wb") as b1:
		pickle.dump(batch1, b1)
		print(" Batch size b1 ",batch1.shape)

	with open("../dumps/ontRep2.pkl","wb") as b2:
		pickle.dump(batch2, b2)
		print(" Batch size b2 ",batch2.shape)

	with open("../dumps/ontRep3.pkl","wb") as b3:
		pickle.dump(batch3, b3)
		print(" Batch size b3 ",batch3.shape)

	with open("../dumps/ontRep4.pkl","wb") as b4:
		pickle.dump(batch4, b4)
		print(" Batch size b4 ",batch4.shape)

	batcht1 = truth_np[0:12534]
	batcht2 = truth_np[12534:2*12534]
	batcht3 = truth_np[12534*2:12534*3]
	batcht4 = truth_np[12534*3:]

	with open("../dumps/truth1.pkl","wb") as bt1:
		pickle.dump(batcht1, bt1)
		print(" Batch size b1 ",batcht1.shape)

	with open("../dumps/truth2.pkl","wb") as bt2:
		pickle.dump(batcht2, bt2)
		print(" Batch size b2 ",batcht2.shape)

	with open("../dumps/truth3.pkl","wb") as bt3:
		pickle.dump(batcht3, bt3)
		print(" Batch size b3 ",batcht3.shape)

	with open("../dumps/truth4.pkl","wb") as bt4:
		pickle.dump(batcht4, bt4)
		print(" Batch size b4 ",batcht4.shape)
else:
	with open("../dumps/ontRep" + file_name + ".pkl","wb") as tf:
		pickle.dump(ontMatrix, tf)
		print(" Batch size  ",ontMatrix.shape)
	with open("../dumps/truth" + file_name + ".pkl","wb") as ttf:
		pickle.dump(truth_np, ttf)
		print(" Batch size  ",truth_np.shape)

