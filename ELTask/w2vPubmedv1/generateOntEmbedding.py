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
		mat_sent = np.random.randn(*(50,250))
		return mat_sent
	#print "Changed sent ",s1
	#Create a matrix for holding the word embeddigns of the sentence set, and each embedding is of len 250
	mat_sent = np.zeros((50, 250))
	#Cut off words after 50, or randomly choose 50 words?
	if len_modelwords > 50:
		rand_indices50 = np.random.choice(len_modelwords,50,replace=False)
		s1 = [s1[i] for i in rand_indices50]
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
#Read values from ont file, and assign embedings
ont_dict = pickle.load(open("../dumps/unique_truthscont.pkl","rb"))

for ont_term in ont_dict.keys():
	print "Generating embedding for ",ont_term
	embedding = generateSentenceEmbedding(model, ont_dict[ont_term]['consolidated'])
	sen_emb[ont_term] = embedding
	emb_list.append(embedding)

ontMatrix = np.stack(emb_list,axis=0)


print " The overall ont embedings length is ",ontMatrix.shape
print "Zero lenghts ",zero_lens
mat_file = open("../dumps/ontEmbeddings.pkl","wb")
pickle.dump(ontMatrix,mat_file)
mat_file.close()

labelEmbeddingFile = open("../dumps/labelsEmbedding.pkl","wb")
pickle.dump(sen_emb, labelEmbeddingFile)
labelEmbeddingFile.close()

print("Max length ",max(lens),"Min len",min(lens)," Mean len ",np.median(lens))
#dist = hausdorffDistModWted (s1,s2,model)

