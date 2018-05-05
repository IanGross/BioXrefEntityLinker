'''
Retrieve the unique truths mappings and find the obonet contents for the same
'''
import obonet
import pickle
import networkx
import re

truths = pickle.load(open("../dumps/unique_truths.pkl","rb"))
path = "/Users/shruthichari/Documents/MS/Courses/NLP/Assignments/BioEventXrefEntityLinker/ELTask/craft-2.0/ontologies/"

#Defining the obonet graphs
PR_obo = obonet.read_obo(open(path + 'PR.obo','r'))
print("Finished loading graph for PR ")
GO_obo = obonet.read_obo('http://www.geneontology.org/ontology/go.obo')
print("Finished loading graph for GO ")

CL_obo = obonet.read_obo('https://raw.githubusercontent.com/obophenotype/cell-ontology/master/cl-basic.obo')
print("Finished loading graph for CL ")

NCBI_obo = obonet.read_obo('http://ontologies.berkeleybop.org/ncbitaxon.obo')
print("Finished loading graph for NCBI ")

SO_obo = obonet.read_obo('https://raw.githubusercontent.com/The-Sequence-Ontology/SO-Ontologies/master/so.obo')
print("Finished loading graph for SO ")
#CHecbi doesn't work need to check
CHEBI_obo = obonet.read_obo('ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo')
print("Finished loading graph for CHEBI ")
print("Finished loading the obo graphs ")

'''
Given the alt-ids, and ontology corpus fetch the names
'''
def get_altidnames(alt_ids, corpus_graph):
	id_to_name = {id_: data['name'] for id_, data in corpus_graph.nodes(data=True)}
	names = []
	for ids in alt_ids:
		present = ids in list(id_to_name.keys())
		print("Ids ",ids," List ", present)
		if present:
			print("Found a name match for ",ids)
			names.append(id_to_name[ids])
	return names


#For each truth extract the value from obonet
#Remove the RELATED [] in synonyms - 
'''
Preprocessing:
syn_patt='"(.*?)" [A-Z]+ \[.*\]'
def_patt = '"(.*?)" \[.*\]'
comment: " ".join(mod_def['comment'].split(" ")[:30])
'''

syn_patt='"(.*?)".*?\[.*\]'
def_patt = '"(.*?)" \[.*\]'

contents ={}
unmatched = []
unique_keysnode = set()
essential_tags = ['name', 'synonym', 'comment', 'def']

for truth in truths:
	try:
		corpus_graph = None
		content = {}
		if truth.startswith("PR"):
			content = PR_obo.node[truth]
			corpus_graph = PR_obo
		elif truth.startswith("GO"):
			content = GO_obo.node[truth]
			corpus_graph = GO_obo
		elif truth.startswith("NCBI"):
			content = NCBI_obo.node[truth]
			corpus_graph = NCBI_obo
		elif truth.startswith("SO"):
			content = SO_obo.node[truth]
			corpus_graph = SO_obo
		elif truth.startswith("CL"):
			content = CL_obo.node[truth]
			corpus_graph = CL_obo
		elif truth.startswith("CHEBI"):
			content = CHEBI_obo.node[truth]
			corpus_graph = CHEBI_obo	
		cont_dict = dict((key,content[key]) for key in content.keys() if key in essential_tags)
		#Could not find obo for EntrezGene, this will be future work	
		if not truth.startswith("EG") and len(cont_dict) > 0:
			#Retain  only those values in the essential tags
			contents[truth] = cont_dict
			if 'def' in contents[truth].keys():
				#Limit to 3 sentences for description
				defi = re.match(def_patt, contents[truth]['def']).groups()[0]			
				contents[truth]['def'] = ".".join(defi.split(".")[:2])
			#Limit it to 5 syns, and remove the []
			#print(" Value ", truth)
			if 'synonym' in contents[truth].keys():
				contents[truth]['synonym'] = [re.match(syn_patt,syn).groups()[0] for syn in contents[truth]['synonym'][:5]]
			#Limit comments to a sentence
			if 'comment' in contents[truth].keys():
				comm = ".".join(contents[truth]['comment'].split(".")[:1])
				contents['truth']['comment'] = comm
		elif len(cont_dict) == 0:
			unmatched.append(truth)
		unique_keysnode |= set(content.keys())

	except KeyError:
		print(" No match for ", truth)
		unmatched.append(truth)
		continue

print(" Keys ",unique_keysnode)
truths_dump = open("../dumps/unique_truthscont.pkl","wb")
pickle.dump(contents,truths_dump,protocol=2)
truths_dump.close()

unmatched_fil = open("../dumps/unmatched_truths.pkl","wb")
pickle.dump(unmatched, unmatched_fil,protocol=2)
unmatched_fil.close()

print("Number of unmatched ",len(unmatched))



