import sys
import gensim 
import numpy as np
import pickle
from gensim.models import KeyedVectors


from func2cleanASentence import * ## import functions needed to clean setences 
from func2getSimOf2GoTerms import * ## import functions needed to compare go terms 
from SentenceSimilarity import infoContentOfWord ## import functions needed to compare go terms 

def keepOnlyWordsInGeneralModel (textInFile,model): ## "model" is from w2v model 
	# l2 = []
	# for l22 in textInFile.split():
		# l22 = l22.strip()
		# if l22 in model.vocab: 
			# l2.append(l22)
	l2 = map(lambda x: x.strip(), textInFile.split())	### textInFile: is a string like 'abcdef'
	l2 = filter(lambda x: x in model.wv.vocab, l2)
	return l2 

#Normalise the word embedding by tf-idf
def infoContentOfWord (w1,model): 
	# print w1
	# return -1*np.log ( model.vocab[w1].count*1.0 / model.corpus_count )
	a = np.log(model.wv.vocab[w1].count*1.0)
	b = np.log(460155)
	return 1.0 - a/b 

lens=[]
zero_lens = []
sen_emb = {}

def generateSentenceEmbedding(model, ont_consolidated):
	#print "Orig sent " + ont_consolidated
	cleaned_desc = cleanASentence(ont_consolidated)
	s1=keepOnlyWordsInModel (cleaned_desc,model)
	len_modelwords = len(s1)
	#print("Len ",len_modelwords)
	lens.append(len_modelwords)

	if len_modelwords == 0:
		#This might need to be run through the other trained word2vec model
		zero_lens.append(cleaned_desc)
		print "Orig sent ",ont_consolidated, "Cleaned sent ", cleaned_desc
		#Generate some random noise as these samples were not seen
		mat_sent = np.random.randn(*(12,250))
		return mat_sent
	#print "Changed sent ",s1
	#Create a matrix for holding the word embeddigns of the sentence set, and each embedding is of len 250
	mat_sent = np.zeros((12, 250))
	#Cut off words after 50, or randomly choose 50 words?
	if len_modelwords > 12:
		rand_indices12 = np.random.choice(len_modelwords,12,replace=False)
		s1 = [s1[i] for i in rand_indices12]
	for i in range(len(s1)):
		w1 = s1[i]
		#print "Word w1 ",w1
		#Obtain word embedding
		mat_sent[i] = infoContentOfWord(w1,model)
	mat_sent = mat_sent.astype('float32')
	return mat_sent

print "loading gensim library, and the already trained word model from 15GB of Pubmed open access articles (may take 1-2 minutes)"
model = gensim.models.Word2Vec.load('modelWord2Vec')

#Model trained for the second assignment, it was trained on a portion of Wikipedia
#generalModel = KeyedVectors.load_word2vec_format('model/enwiki_emb.txt')

print "finished loading."
emb_list = []
file_word = "testSet"
#Read values from ont file, and assign embedings
term_list = pickle.load(open("../dumps/word_contexts" + file_word + ".pkl","rb"))

for word_term in term_list:
	cont = word_term['consolidated']
	print "Generating embedding for ",word_term['consolidated'], " with truth ",word_term['truth']
	embedding = generateSentenceEmbedding(model, cont)
	emb_list.append(embedding)

wordMatrix = np.stack(emb_list,axis=0)


#Distribute data for normal rep
if file_word != "testSet":
	#Distribute the matrix, unable to write it fully
	batch1 = wordMatrix[0:12534]
	batch2 = wordMatrix[12534:2*12534]
	batch3 = wordMatrix[12534*2:12534*3]
	batch4 = wordMatrix[12534*3:]
	with open("../dumps/wordRep1.pkl","wb") as b1:
		pickle.dump(batch1, b1)
		print(" Batch size b1 ",batch1.shape)

	with open("../dumps/wordRep2.pkl","wb") as b2:
		pickle.dump(batch2, b2)
		print(" Batch size b2 ",batch2.shape)

	with open("../dumps/wordRep3.pkl","wb") as b3:
		pickle.dump(batch3, b3)
		print(" Batch size b3 ",batch3.shape)

	with open("../dumps/wordRep4.pkl","wb") as b4:
		pickle.dump(batch4, b4)
		print(" Batch size b4 ",batch4.shape)
else:
	with open("../dumps/wordRep" + file_word + ".pkl","wb") as wr:
		pickle.dump(wordMatrix, wr)
		print(" Batch size ",wordMatrix.shape)


print " The overall word embedings length is ",wordMatrix.shape
print "Zero lenghts ",zero_lens
print("Max length ",max(lens),"Min len",min(lens)," Mean len ",np.median(lens))
#dist = hausdorffDistModWted (s1,s2,model)

