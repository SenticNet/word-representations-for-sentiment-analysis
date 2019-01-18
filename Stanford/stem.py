from nltk import SnowballStemmer
def stem(word):
	'''danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian',
		 'porter', 'portuguese", 'romanian',  'russian', 'spanish', 'swedish')'''
	stemmer = SnowballStemmer("english")
	try:
		word=stemmer.stem(word).encode('utf-8')
	except Exception,e:
		word=word
	return word
lines=open('datasetSentences.txt','r').readlines()
fw=open('datasetSentences_stem.txt','a')
for line in lines:
	words=line.strip().split('\t')[1].strip().split(' ')
	for word in words:
		wordstem=stem(word)
		fw.write(wordstem)
		if words.index(word)==len(words)-1:
			fw.write('\n')
		else:
			fw.write(' ')


