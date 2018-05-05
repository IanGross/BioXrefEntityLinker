import pickle
import numpy as np
import json

ont_content = pickle.load(open("../dumps/unique_truthscont.pkl","rb"))

essential_tags = ['name', 'synonym', 'comment', 'def']
lens = []
max_len = 0
max_sen = ""
max_truth = ""
labels_ctr = 0
labels_mapping = {}
labelsID_mapping = {}
print("Len ",len(ont_content))
#Split the comments
for keys in ont_content.keys():
	#Uncomment for the first run after extracting from obo
	if 'comment' in ont_content[keys].keys():
		ont_content[keys]['comment'] = ".".join(ont_content[keys]['comment'].split('.')[:1])
	ont_content[keys]['consolidated'] = ""
	labels_ctr += 1
	term_keys = ont_content[keys].keys()
	term_value = ont_content[keys]
	if "name" in term_keys:
		ont_content[keys]['consolidated'] += term_value['name'] + " "
	if 'synonym' in term_keys:
		ont_content[keys]['consolidated'] += ", ".join(term_value['synonym']) + " "	
	if 'def' in term_keys:
		if len(" ".join(term_value['def'].split(".")).split(" ")) >= 60:
			ont_content[keys]['consolidated'] += " ".join(" ".join(term_value['def'].split(".")).split(" ")[:60]) + " " 
		else:
			ont_content[keys]['consolidated'] += " ".join(" ".join(term_value['def'].split(".")).split(" ")) + " "
	existing_consolidated = ont_content[keys]['consolidated']
	#Only use synonyms and comments if there isn't already too much context
	if len(existing_consolidated.split(" ")) < 40 and 'synonym' in term_keys:
		ont_content[keys]['consolidated'] += ", ".join(term_value['synonym']) + " "
	if len(existing_consolidated.split(" ")) < 40 and 'comment' in term_keys:
		if len(" ".join(term_value['comment'].split(".")).split(" ")) >= 35:
			ont_content[keys]['consolidated'] += " ".join(" ".join(term_value['comment'].split(".")).split(" ")[:35]) + " " 
		else:
			ont_content[keys]['consolidated'] += " ".join(" ".join(term_value['comment'].split(".")).split(" ")) + " "
	
	lens.append(len(ont_content[keys]['consolidated'].split(" ")))
	if len(ont_content[keys]['consolidated'].split(" ")) > max_len:
		max_len = len(ont_content[keys]['consolidated'].split(" "))
		max_sen = ont_content[keys]['consolidated']
		max_truth = keys
	labels_mapping[labels_ctr] = keys
	labelsID_mapping[keys] = labels_ctr

#Write to the file - wb
ont_write = open("../dumps/unique_truthscont.pkl","wb")
pickle.dump(ont_content, ont_write,protocol=2)
ont_write.close()

#Maintain a mapping file with the IDs against the labels, this will be used for lookup by the training and test data to assign label classes
with open('../dumps/IDLabelMapping.json', 'w') as outfile:
    json.dump(labels_mapping, outfile)

with open('../dumps/LabelIDMapping.json', 'w') as outfile:
    json.dump(labelsID_mapping, outfile)


#print(" Lens ",lens)
print("Min len ",min(lens)," Mean len ",np.mean(lens)," Max len ",max(lens), " GT ",max_truth)
print(" Max sen ",max_sen)