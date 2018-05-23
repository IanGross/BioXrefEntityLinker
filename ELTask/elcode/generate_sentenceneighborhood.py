import os
from bs4 import BeautifulSoup
import pickle
import re
import shutil
from pattern.en import singularize
from readerModified import *
import random
import json


'''
A word is in some depth in AMR, if it is not present use entire sentence
Else if the node is a parent, use two of its children and its adjacent nodes
Otherwise use adjacent nodes alone
'''
def generateNeighborhood(amrCont, word):
	word = word.lower()	
	nodeKeys = list(amrCont.amr_nodes.keys())
	'''
	Have a dictionary of name against first occurence in path, use this to find the nodes directly connected to it
	'''
	nodeNames= []
	nodeNameFullNameMapping = {}
	for nodeKey in amrCont.amr_nodes.keys():
	    '''
	    self.name = name               # Node name (acronym)
	        self.ful_name = ful_name       # Full name of the node
	        self.next_nodes = next_nodes   # Next nodes (list)
	    '''
	    nodeAMR = amrCont.amr_nodes[nodeKey]
	    cont = nodeAMR.content
	    full_name = re.sub('-[\d]+','',nodeAMR.ful_name)
	    nodeNames.append((nodeAMR.name,full_name.lower()))
	    nodeNameFullNameMapping[nodeAMR.name] = full_name

	paths = amrCont.graph
	#Name against first occurence
	allNamesFirst = {}
	'''
	As a node is consumed add the first occurence to the tuple
	'''
	for trav in range(len(paths)):
	    tuple_i = paths[trav]
	    vals = (tuple_i[0],tuple_i[1])
	    #@ is root node
	    if vals[0] != '@' and vals[0] not in allNamesFirst.keys():
	        allNamesFirst[vals[0]] = trav

	    if vals[1] not in allNamesFirst.keys():
	        allNamesFirst[vals[1]] = trav

	all_FullNames = [ele[1] for ele in nodeNames]
	nodeNameFirstOccurence = 0
	associatedNodeName = ""	
	if word in all_FullNames:
		firstMatch = all_FullNames.index(word)
		associatedNodeName = nodeNames[firstMatch][0]
		#It is not in the AMR, use a random term then?
		if associatedNodeName in allNamesFirst.keys():
			nodeNameFirstOccurence = allNamesFirst[associatedNodeName]
		else:	
			print("Using a random pos ")			
			firstMatch = random.randint(0, len(allNamesFirst) -1)
			associatedNodeName = list(allNamesFirst.keys())[firstMatch]
			nodeNameFirstOccurence = allNamesFirst[associatedNodeName]
	else:
		#Try singularize and see if this form exists, solves something like mice - mouse
		singular = singularize(word)
		if singular in all_FullNames:
			firstMatch = all_FullNames.index(singular)
			associatedNodeName = nodeNames[firstMatch][0]
			#It is not in the AMR, use a random term then?
			if associatedNodeName in allNamesFirst.keys():
				nodeNameFirstOccurence = allNamesFirst[associatedNodeName]
			else:		
				print("Using a random pos ")		
				firstMatch = random.randint(0, len(allNamesFirst) - 1)
				associatedNodeName = list(allNamesFirst.keys())[firstMatch]
				nodeNameFirstOccurence = allNamesFirst[associatedNodeName]
		else:
			max_matchpos = -1
			max_matchlen = 0
			word_list = list(word)
			for possible_nodei in range(len(all_FullNames)):
				node_list = list(all_FullNames[possible_nodei])
				#Find maximum number of intersection and weight them by the number of repititions, eg cell-> cellular and not visceral
				counts = [node_list.count(chari) for chari in word_list]
				intersection_len = sum(counts)
				if intersection_len > max_matchlen:
					max_matchlen = intersection_len
					max_matchpos = possible_nodei
			firstMatch = max_matchpos
			associatedNodeName = nodeNames[firstMatch][0]
			#It is not in the AMR, use a random term then?
			if associatedNodeName in allNamesFirst.keys():
				nodeNameFirstOccurence = allNamesFirst[associatedNodeName]
			else:
				print("Using a random pos ")				
				firstMatch = random.randint(0, len(allNamesFirst) -1)
				associatedNodeName = list(allNamesFirst.keys())[firstMatch]
				nodeNameFirstOccurence = allNamesFirst[associatedNodeName]

		print(" Fuzzy match for ", word, " in ",all_FullNames[firstMatch])

	candidates = []
	#Extract six words, three ahead and three behind 
	if nodeNameFirstOccurence == 0:
		if len(paths) < 7:
			candidates = paths[1:]
		else:
			candidates = paths[1:7]
	elif nodeNameFirstOccurence == 1:
		candidates.append(paths[0])
		if len(paths) >= 8:
			candidates += paths[2:7]
		else:
			candidates = paths[2:]
	#2nd occurence
	elif nodeNameFirstOccurence == 2:
		candidates += paths[0:2]
		if len(paths) >= 8:
			candidates += paths[3:7]
		else:
			candidates += paths[3:]
	#Last occurence
	elif nodeNameFirstOccurence == len(paths) - 1:
		if len(paths) > 7:
			candidates = paths[len(paths)-2:len(paths)-6:-1]
		else:
			candidates = paths[::-1]	
	#3rd or second last
	elif nodeNameFirstOccurence >= (len(paths) - 3):
		candidates += paths[nodeNameFirstOccurence + 1:]
		need_consum = (6 - len(candidates))
		#Consume all
		if (len(paths) - len(candidates)) - 1 < need_consum: 
			candidates += paths[0:nodeNameFirstOccurence]
		else:
			candidates += paths[nodeNameFirstOccurence - 1:nodeNameFirstOccurence - need_consum:-1]
	#Anywhere else
	else:
		#Might need to do exception handling here
		candidates += paths[nodeNameFirstOccurence - 3:nodeNameFirstOccurence] + paths[nodeNameFirstOccurence + 1: nodeNameFirstOccurence + 4]

	amr_neighbors=[]
	amr_neighbor_names =[]
	if associatedNodeName != "" and associatedNodeName in nodeNameFullNameMapping.keys():
		amr_neighbors.append(nodeNameFullNameMapping[associatedNodeName])
		amr_neighbor_names.append(associatedNodeName)
	else:
		amr_neighbors.append(word)
	for vals in candidates:
		if vals[0] != "@" and vals[0] not in amr_neighbor_names:
			amr_neighbors.append(nodeNameFullNameMapping[vals[0]])
			amr_neighbor_names.append(vals[0])
		if vals[1] not in amr_neighbor_names:
			amr_neighbors.append(nodeNameFullNameMapping[vals[1]])
			amr_neighbor_names.append(vals[1])
	#print(" AMR ", amrCont.raw_amr)
	print(" Neighborhood words are "," ".join(amr_neighbors), " Pos ", nodeNameFirstOccurence, " Path lengths ",len(paths))
	return " ".join(amr_neighbors)

'''
Performs the neighborhood generation for a particular file alone
Need AMR file, term file, and normal text
'''
def loadSentences(ind_file):
	unmatched_truths = pickle.load(open("../dumps/unmatchedalways_truths.pkl","rb"))
	labelIDmapping = json.load(open("../dumps/LabelIDmapping.json","rb"))
	word_truths = []
	utruthlist = []
	groundtruthslist = []
	amr_content = {}
	actual_unmatched = set()
	#Term, amr and normal file
	ind_path =  "test/" + ind_file + ".txt.xml"
	amr_path = "test/" + ind_file + ".amr.txt"
	normal_path ="test/" + ind_file + ".txt"
	if ind_path.endswith("txt.xml"):
			#path
			with open(ind_path,"r", encoding='utf-8', errors='ignore') as ind_fileo:
				print("File name ",ind_file)
				#actualFileName = re.match('([\d]+).txt.xml',ind_file).groups()[0]
				amrCont = open(amr_path,'r').read()
				#Fetch amrCont of file if not already populated
				if ind_file not in amr_content.keys():
					print("Fetching amr for ", ind_file)
					amr_content[ind_file] = returnAMR(amrCont)
					print("File name ", ind_file)

				#print(ind_fileo)
				for line in ind_fileo:
					#line = str(line.strip())
					#line = str(line)
					line = line.replace("termsem", "term sem").replace("\"> ", "\">").replace(" </term>", "</term>").replace("</term> - ","</term>-").replace("- <term","-<term").replace("<?xmlversion=\"1.0\"encoding=\"UTF-8\"?> ", "")
					mod_line = line
					#Remove the space in hyphen
					#line = line.replace(" - ","-")
							
					#Non-empty line
					if line != "" and "term" in line:
						#Change this to just remove the terms from the sentence
						mod_sent = re.sub('<.*?>', '', line)
						sentAmrFound = False
						amr_sent = None
						#Find the amr_sent cont
						try:
							amr_sent = amr_content[ind_file][mod_sent.strip()]
							sentAmrFound = True
							#generateNeighborhood(amr_sent)								
						except KeyError:
							#print(" No match for ", mod_sent, " Mod line ",mod_line)
							orig_sent = set(mod_sent.strip().split(" "))
							max_intersection = 0
							max_sent = ""
							#Find sentence with max intersection and use that instead
							for sent in amr_content[ind_file].keys():
								sent_set = set(sent.strip().split(" "))
								intersection_len = len(orig_sent.intersection(sent_set))
								if intersection_len > max_intersection:
									max_intersection = intersection_len
									max_sent = sent
								try:
									amr_sent = amr_content[ind_file][max_sent.strip()]										
									#print(" Found match in ",max_sent)
									sentAmrFound = True
								except KeyError:
									print(" No hope for ",mod_sent.strip())
									actual_unmatched.update([mod_sent.strip()])
						#Only if AMR found pursue the term mentions
						if sentAmrFound:
							bsOut = BeautifulSoup(line, 'lxml')
							sent_groundtemplate = {"sentence":"","word_truths":[]}								
							term_mentions = bsOut.findAll("term")

							for i in range(len(term_mentions)):
								ontology_mapping = term_mentions[i]["sem"]
								word = term_mentions[i].text
								#Don't choose the unmatched terms
								if ontology_mapping not in unmatched_truths:
									#Add individual mappings for 1-Hot
									if ontology_mapping not in utruthlist:
										#print(sentenceleveltruths)
										#Will need to add an AMR portion here, pos will be used to retrieve AMR candidates if word not found
										amr_neighbors = generateNeighborhood(amr_sent,word)	
										word_truths.append({"entity":word,"truth":ontology_mapping,"sentence":mod_sent.strip(),"consolidated":amr_neighbors})
										groundtruthslist.append(labelIDmapping[ontology_mapping])			
	print(" unmatched ", len(actual_unmatched))
	print("Finished adding truths for ",ind_file)		
	

	#Store the entity neighborhoods and ground truths
	word_contextsf = open("../dumps/word_contexts" + ind_file + ".pkl","wb")
	ground_truthsf = open("../dumps/ground_truths" + ind_file + ".pkl","wb")

	pickle.dump(word_truths,word_contextsf,protocol=2)
	word_contextsf.close()

	pickle.dump(groundtruthslist,ground_truthsf,protocol=2)
	ground_truthsf.close()

	unmatched_f = open("../dumps/unmatcheddump" + ind_file + ".pkl","wb")
	pickle.dump(actual_unmatched,unmatched_f)
	unmatched_f.close()

	print(" Total number of terms ", len(word_truths))


'''
Main function to run - 
reads annotated and AMR files from the directories
Makes a call to the generateNeighborhood function and retrieves the neighborhood words for an entity mention
Outputs unique_truths, entity contexts and ground_truths(used to generate the y's)
'''
def sentenceXMLtoMapping():
	path = "../SentenceOutputFiles/"
	#TextInputFiles
	#path = "../craft-2.0/genia-xml/term/"
	all_subdirs = os.listdir(path)

	unmatched_truths = pickle.load(open("../dumps/unmatchedalways_truths.pkl","rb"))
	labelIDmapping = json.load(open("../dumps/LabelIDmapping.json","rb"))

	#print("All sub_dir",all_subdirs)
	#ground_truths = open("../dumps/word_truths.pkl","wb")
	#Will be used for the indexing
	
	amr_path = '../Sentence-Text_AMR/PubmedAMR/'
	word_truths = []
	utruthlist = []
	groundtruthslist = []
	amr_content = {}
	actual_unmatched = set()
	for ind_file in all_subdirs:
		if ind_file.endswith("txt.xml") and not ind_file.startswith("entrezgene"):
				#path
				with open(path + "/" + ind_file,"r", encoding='utf-8', errors='ignore') as ind_fileo:
					print("File name ",ind_file)
					actualFileName = re.match('.*?_([\d]+).txt.xml',ind_file).groups()[0]
					#actualFileName = re.match('([\d]+).txt.xml',ind_file).groups()[0]
					amrCont = open(amr_path + actualFileName + ".txt",'r').read()
					#Fetch amrCont of file if not already populated
					if actualFileName not in amr_content.keys():
						print("Fetching amr for ",actualFileName)
						amr_content[actualFileName] = returnAMR(amrCont)
					print("File name ", ind_file)

					#print(ind_fileo)
					for line in ind_fileo:
						#line = str(line.strip())
						#line = str(line)
						line = line.replace("termsem", "term sem").replace("\"> ", "\">").replace(" </term>", "</term>").replace("</term> - ","</term>-").replace("- <term","-<term").replace("<?xmlversion=\"1.0\"encoding=\"UTF-8\"?> ", "")
						mod_line = line
						#Remove the space in hyphen
						#line = line.replace(" - ","-")
							
						#Non-empty line
						if line != "" and "term" in line:
							#Change this to just remove the terms from the sentence
							mod_sent = re.sub('<.*?>', '', line)
							sentAmrFound = False
							amr_sent = None
							#Find the amr_sent cont
							try:
								amr_sent = amr_content[actualFileName][mod_sent.strip()]
								sentAmrFound = True
								#generateNeighborhood(amr_sent)								
							except KeyError:
								#print(" No match for ", mod_sent, " Mod line ",mod_line)
								orig_sent = set(mod_sent.strip().split(" "))
								max_intersection = 0
								max_sent = ""
								#Find sentence with max intersection and use that instead
								for sent in amr_content[actualFileName].keys():
									sent_set = set(sent.strip().split(" "))
									intersection_len = len(orig_sent.intersection(sent_set))
									if intersection_len > max_intersection:
										max_intersection = intersection_len
										max_sent = sent
								try:
									amr_sent = amr_content[actualFileName][max_sent.strip()]										
									#print(" Found match in ",max_sent)
									sentAmrFound = True
								except KeyError:
									print(" No hope for ",mod_sent.strip())
									actual_unmatched.update([mod_sent.strip()])
							#Only if AMR found pursue the term mentions
							if sentAmrFound:
								bsOut = BeautifulSoup(line, 'lxml')
								sent_groundtemplate = {"sentence":"","word_truths":[]}								
								term_mentions = bsOut.findAll("term")

								for i in range(len(term_mentions)):
									ontology_mapping = term_mentions[i]["sem"]
									word = term_mentions[i].text
									#Don't choose the unmatched terms
									if ontology_mapping not in unmatched_truths:
										#Add individual mappings for 1-Hot
										if ontology_mapping not in utruthlist:
											utruthlist.append(ontology_mapping)
											#print(sentenceleveltruths)
										#Will need to add an AMR portion here, pos will be used to retrieve AMR candidates if word not found
										amr_neighbors = generateNeighborhood(amr_sent,word)	
										word_truths.append({"truth":ontology_mapping,"sentence":mod_sent.strip(),"consolidated":amr_neighbors})
										groundtruthslist.append(labelIDmapping[ontology_mapping])			
					print(len(utruthlist))
					print(" unmatched ", len(actual_unmatched))
					print("Finished adding truths for ",ind_file)
	# for sub_dir in all_subdirs:
	# 	#Ignore the italics ontology
	# 	if sub_dir != "sections-and-typography" or sub_dir != "entrezgene":
	# 		#Append sub_dir to path and extract all files in the sub dir
	# 		sub_dirpath = path + sub_dir
	# 		files_subdir = os.listdir(sub_dirpath)
	# 		#all_subdirs

	unique_truths = open("../dumps/unique_truths.pkl","wb")
	word_contextsf = open("../dumps/word_contexts.pkl","wb")
	ground_truthsf = open("../dumps/ground_truths.pkl","wb")

	pickle.dump(word_truths,word_contextsf,protocol=2)
	word_contextsf.close()

	pickle.dump(groundtruthslist,ground_truthsf,protocol=2)
	ground_truthsf.close()

	pickle.dump(utruthlist, unique_truths)
	unique_truths.close()

	unmatched_f = open("unmatcheddump.pkl","wb")
	pickle.dump(actual_unmatched,unmatched_f)
	unmatched_f.close()

	print("Number of unique_truths is ",len(utruthlist))
	print(" Total number of terms ", len(word_truths))

def moveRenamefiles():
	path = "/mnt/c/Users/grossi2/Documents/RPI Work Files/Current Semester/Natural Language Processing - CSCI 4130/TermProject/craft-2.0/genia-xml/term"
	#path = "/Users/shruthichari/Documents/MS/Courses/NLP/Assignments/ELTask/craft-2.0/genia-xml/term"
	textpath = "/mnt/c/Users/grossi2/Documents/RPI Work Files/Current Semester/Natural Language Processing - CSCI 4130/TermProject/TextInputFiles/"
	
	all_subdirs = os.listdir(path)

	for sub_dir in all_subdirs:
		#Ignore the italics ontology
		if sub_dir != "sections-and-typography":
			print(sub_dir)
			#Append sub_dir to path and extract all files in the sub dir
			sub_dirpath = path + "/" + sub_dir
			files_subdir = os.listdir(sub_dirpath)
			for ind_file in files_subdir:
				#Only consider the sentences from files with proper annotations
				if ind_file.endswith("txt.xml"):
					xmlFile = sub_dirpath + "/" + ind_file
					shutil.copy2(xmlFile, textpath + sub_dir + "_" + ind_file)

	


#Run this first
#moveRenamefiles()
#Next, run the sentence parser (NOTE: currently in a tokenized form, but can be very easily modified to be untokenized)
#Finally, run this function (Still need to generate the AMR based on these sentences)
#sentenceXMLtoMapping()
loadSentences("testSet")



