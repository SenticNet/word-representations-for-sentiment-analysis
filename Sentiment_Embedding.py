# -*- coding: utf-8 -*
from __future__ import division
import math
from nltk import SnowballStemmer
import numpy as np
import random
from time import time,clock
import cPickle as pickle
import os,sys
import threading
import Queue
from Similarity_Eva import distanceCalculate
import itertools
from multiprocessing import Pool, freeze_support
import pathos.multiprocessing as mp
import inspect
import ctypes 
# np.seterr(all='warn')
# old_settings = np.seterr(all='ignore')  #seterr to known value
def com(word1,word2):
	return word1+';'+word2


class Build_Dict(object):
	"""docstring for Build_Dict"""
	def __init__(self):
		super(Build_Dict, self).__init__()
		self.increment=False
		self.write_params=True
		self.file_corpus='../Stanford/datasetSentences.txt'
		self.file_corpus_lable='../Stanford/datasetSplit_label.txt'
		self.dict_index={}  ##word:index
		self.dict={}
		self.document_level=True
		self.dict_file='./cooccur.pkl'
		self.dict_cooccur={} ## {word1;word2:value} ## 共现矩阵 
		self.dict_cosenti={} ## {word: [0,0,0,0,0]} 词word的情感分布 情感-词矩阵 
	def load_dict(self):
		
		dict_name='dict.txt'

		dicts={}
		dicts_index={}
		dicts_index_re={}
		try:
			print "loading the dict..."
			for lin in open(dict_name,'r').readlines(): #初始化字典，将写入文件的字典读到self.dict
				key=lin.strip().split('\t')[0]
				value=lin.strip().split('\t')[1]
				value_index=lin.strip().split('\t')[2]
				dicts[key]=int(value)
				dicts_index[key]=int(value_index)
				dicts_index_re[int(value_index)]=key 
		except Exception,e:
			pass
		self.dict=dicts
		self.dict_index=dicts_index
		self.dict_index_re=dicts_index_re
		return self.dict,self.dict_index,self.dict_index_re
	# @staticmethod
	def build_corpus_upgratly(self, word): #增量式更新现有字典，最后将文件字典进行重写操作，保存最新的字典value
		word=stem(word)
		if self.dict.has_key(word):
			self.dict[word]+=1
		else:
			if len(self.dict)>0:
				self.dict[word]=1
				self.dict_index[word]=max(self.dict_index.values())+1
			else:
				self.dict[word]=1
				self.dict_index[word]=1
		# print self.dict
		if self.increment==True:

			file_update=open('dict.txt','w')

			for i in range(len(self.dict.items())):
				file_update.write(self.dict.items()[i][0]+'\t'+str(self.dict.items()[i][1])+'\t')
				file_update.write(str(self.dict_index.items()[i][1])+'\n')

	def read_corpus(self):
		
		file_lines=self.file_corpus
		index=0
		lines=open(file_lines,'r').readlines()
		self.load_dict()
		print "reading the corpus..."
		for lin in lines:
			
			words=lin.strip().split('\t')[1].strip().split(' ')
			index+=1			
			index_w=0
			# print index
			for word in words:
				index_w+=1
				if index==len(lines) and index_w==len(words):#保证读到增量语句的最后一个词进行字典的更新
					self.increment=True
					print "writing the dic.txt..."

				self.build_corpus_upgratly(word)

	## catch the sentence pattern by frequency ## 
	def sequence_pattern_mining(self):
		##
		file_name=self.file_corpus
		# for lin in open(file_name,'r').readlines():
	## 构造共现矩阵，最原始的方式二次遍历
	def sentic_lable(self):
		file_lable=self.file_corpus_lable
		sentence_lable=[]
		sentence_type=[]
		lines=open(file_lable,'r').readlines()
		for lin in lines:
			sentence_lable.append(int(lin.strip().split(',')[2]))
			sentence_type.append(int(lin.strip().split(',')[1]))
		return sentence_lable,sentence_type

	def context_matrix(self):
		if os.path.exists('./dict.txt'):
			self.load_dict()
		else: 
			self.read_corpus()
		windows=1
		row_len=len(self.dict)

		sentence_lable,sentence_type=self.sentic_lable()


		symmetric=True	##滑动窗口是否是对称的
		start_time=time()
		file_corpus=self.file_corpus
		lin_index=0
		lines=open(file_corpus,'r').readlines()
		lin_num=len(lines)
		cosenti_margin_sum=[0,0,0,0,0]
		cooccur_margin_sum={}     
		for lin in lines:
			words=lin.strip().split('\t')[1].strip().split(' ')
			lin_index+=1
			if lin_index%100==0:
				print "it is finished %f%% by using %f seconds"%(lin_index/lin_num*100,time()-start_time)
			for ind in range(len(words)):
				word=stem(words[ind])

				## document_level means 在当前文档的情感倾向情况下，词的统计
				if self.document_level:#self.dict_index[word]-1] 该量索引从1开始
					cosenti_margin_sum[sentence_lable[lin_index-1]]+=1
					if cooccur_margin_sum.has_key(word):
						cooccur_margin_sum[word]+=1
					else:
						cooccur_margin_sum[word]=1

					if self.dict_cosenti.has_key(word):
						self.dict_cosenti[word][sentence_lable[lin_index-1]]+=1
					else:
						self.dict_cosenti[word]=[0.,0.,0.,0.,0.]
						self.dict_cosenti[word][sentence_lable[lin_index-1]]=1
					# conSentic[self.dict_index[word]-1][sentence_lable[lin_index-1]]+=1
				# elif self.word_level:


				if symmetric:
					for i in range(-windows,windows+1):
						if ind+i>=0 and ind+i <len(words) and stem(words[ind])==stem(words[ind+i]):
							pass
						elif ind+i>=0 and ind+i <len(words) and stem(words[ind])!=stem(words[ind+i]):
							new_key=com(stem(words[ind]),stem(words[ind+i]))
							if self.dict_cooccur.has_key(new_key):
								self.dict_cooccur[new_key]+=1
							else:
								self.dict_cooccur[new_key]=1
							# conM_row=self.dict_index[stem(words[ind])]-1
							# conM_col=self.dict_index[stem(words[ind+i])]-1
							# conMatrix[conM_row][conM_col]+=1

				else:
					for i in range(windows):
						if ind-i>=0 and stem(words[ind])==stem(words[ind-i]):
							pass
						elif ind-i>=0 and stem(words[ind])!=stem(words[ind-i]):
							if self.dict_cooccur.has_key(new_key):
								self.dict_cooccur[new_key]+=1
							else:
								self.dict_cooccur[new_key]=1

							# conM_row=self.dict_index[stem(words[ind])]-1
							# conM_col=self.dict_index[stem(words[ind-i])]-1
							# conMatrix[conM_row][conM_col]+=1
					# if self.dict.has_key(word):## 加判断的目的是，将来可以把频数小于某阈值的word去掉
		stop_time=time()
		## make the conMatrix to the real-value format which represents the P_ij=P(j|i)=X_ij/X_i##
		print "It has finished the loading and get %d words.."%(len(self.dict_cosenti))
		# ij=0
		# for com_key in self.dict_cooccur:
			# ij+=1
			# if ij%100==0:
			# 	print "It is rescaling %f%%"%(ij/len(self.dict_cooccur.keys()))
			# def decoms(word):
			# 	word=word.strip().split(';')
			# 	return word[0],word[1]
			# word1,word2=decoms(com_key)

			# self.dict_cooccur[com_key]/=cooccur_margin_sum[word1]

		# for word in self.dict_cosenti:
		# 	for iik in range(5):
		# 		self.dict_cosenti[word][iik]/=cosenti_margin_sum[iik]



		stop_time1=time()
		print "the duration is",stop_time1-start_time

		if self.write_params==True:
			fwM=open(self.dict_file,'wb')
			pickle.dump(self.dict_cooccur,fwM)
			pickle.dump(self.dict_cosenti,fwM)
 

		return self.dict_cooccur,self.dict_cosenti


##  词干化处理  ###
def stem(word):
	'''danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian',
		 'porter', 'portuguese", 'romanian',  'russian', 'spanish', 'swedish')'''
	stemmer = SnowballStemmer("english")
	try:
		word=stemmer.stem(word).encode('utf-8')
	except Exception,e:
		word=word
	return word

class Train_SGlove(object):
	"""docstring for Train_SGlove"""
	def __init__(self):
		super(Train_SGlove, self).__init__()
 
		build_dict=Build_Dict()
		dicts,dict_index,dict_index_re=build_dict.load_dict()
		self.dict=dicts
		self.dict_index=dict_index
		self.dict_index_re=dict_index_re
		if os.path.exists('./cooccur.pkl'):
			conSentic_file=open('./cooccur.pkl','rb')
			self.conMatrix=pickle.load(conSentic_file)
			self.conSentic=pickle.load(conSentic_file)
			
		else:
			self.conMatrix,self.conSentic=build_dict.context_matrix()
		self.vocab_size=len(self.dict)
		''' hyper_params start'''
		self.alpha=0.75
		self.thread_num=1
		self.eta=0.5
		self.vector_size=50
		self.x_max=100
		self.iternum=50
		''' hyper_params end '''

		self.Wf='./Wparams_senti_depart_30_new_word1_075_1_750.pkl' 
		self.bs='./Wparams_bs.pkl'
		self.W={}
		self.S={}
		self.b_i={}
		self.b_k={}
		self.gradsq_w={}
		self.gradsq_s={}
		self.gradsq_b_i={}
		self.gradsq_b_k={}
		self.res=[]
		self.target_center=True
		self.document_level_2=False #document_level by using the sentence senitment fraction
		self.document_level_1=False # word_level with word senitment distribution in the corpus...
		self.from_original=True
		self.word_level_1=True #word_level but without sentiment parameters.
		self.final_vector=True  #store the {word:vector} into the pkl
		self.Alpha=0.75
		if self.from_original:
			for i in range(self.vocab_size):
				self.W[i]=np.array([0.]*self.vector_size,dtype='float64')
				self.S[i]=np.array([0.]*5,dtype='float64')
				self.gradsq_w[i]=np.array([2.]*self.vector_size,dtype='float64')
				self.gradsq_b_i[i]=2.0
				self.gradsq_b_k[i]=2.0
				self.b_i[i]=random.uniform(-0.5,0.5)/self.vector_size
				self.b_k[i]=random.uniform(-0.5,0.5)/self.vector_size
				for jw in range(self.vector_size):
					self.W[i][jw]=random.uniform(-0.5,0.5)/self.vector_size
			# for i in range(self.vocab_size):
				for js in range(5):
					self.S[i][js]=random.uniform(0,0.5)/self.vector_size
				self.gradsq_s[i]=np.array([2.]*5,dtype='float64')
		else:
			W_file=open(self.Wf,'rb')
			self.W=pickle.load(W_file)
			self.S=pickle.load(W_file)
			W_b_file=open(self.bs,'rb')
			self.b_i=pickle.load(W_b_file)
			self.b_k=pickle.load(W_b_file)
			for i in range(self.vocab_size):
				self.gradsq_w[i]=np.array([2.]*self.vector_size,dtype='float64')
				self.gradsq_b_i[i]=2.
				self.gradsq_b_k[i]=2.
				self.gradsq_s[i]=np.array([2.]*5,dtype='float64')		
		self.chunk_size=math.ceil(len(self.conMatrix)/self.thread_num)
		


	def train_thread(self,conMatrix_chunk):
		# print "training...",thread_num
		cost=0
		indx=-1
		t1=time()
		def decoms(word):
			word=word.strip().split(';')
			return word[0],word[1]
		def word_Sentic():
			wordSentic={}
			for line in open('dict_sentiment_urban1.txt','r').readlines():
				wordS=line.strip().split('\t')[0]
				word_senti=int(line.strip().split('\t')[3])
				wordSentic[wordS]=word_senti
			return wordSentic
		if self.word_level_1==True:
			self.wordSentic=word_Sentic()
		for com_key in conMatrix_chunk:
			word1,word2=decoms(com_key)
			conValue=self.conMatrix[com_key]
			t3=clock()
			# print "there are %s seconds before training start"%(t3-t1)
			indx+=1
			if self.target_center==True:
				word_index=word1
			else:
				word_index=word2
			# print word1,word2,com_key,indx
			if conValue>0:
				# conValue/=200
				# print indx,conValue
				l1=self.dict_index[word1]-1 ## courrent word location
				l2=self.dict_index[word2]-1 #+self.vocab_size ## context word location
				# S_l2=self.dict_index[word2]-1 ## context word location
				
				if self.document_level_1: ## document-level sum_{s=1}^{S}(w_i*w_k*s+b_i*s+b_k*s-log(X_ik)-log(M_is))^2
					conSentic=np.array(self.conSentic[word_index])
					conSentic+=1
					# if sum(conSentic)>0:
					conSentic/=sum(conSentic)
					conSentic/=1
					Count_value=conSentic*conValue
					Wij=np.inner(self.W[l1],self.W[l2])

					if self.target_center==True:
						diff=Wij*self.S[l1]+(self.b_i[l1]+self.b_k[l2])*self.S[l1]-np.log10(Count_value) 
					else:
						diff=Wij*self.S[l2]+(self.b_i[l1]+self.b_k[l2])*self.S[l2]-np.log10(Count_value)
					# print np.sqrt(diff.dot(diff)),np.log10(Count_value)
				elif self.document_level_2:## document-level (w_i*w_k+b_i+b_k-log(X_ik)-log(M_is))^2
					Count_value=self.conSentic[word_index]
					pos=Count_value[3]+Count_value[4]
					neu=Count_value[2]
					neg=Count_value[0]+Count_value[1]
					# print pos

					if neu>pos and neu>neg:
						Senti_value=1.0

					else:
						if neg==0.0:
							if pos==0.0:
								Senti_value=1
							else:
								Senti_value=pos
						else:
							if pos==0.0:
								Senti_value=1
							else:
								Senti_value=pos/neg
					if Senti_value<1:
						Senti_value/=(1/self.Alpha)
					elif Senti_value>1:
						Senti_value*=(1/self.Alpha)
						# print "%f,%f,%f"%(Senti_value,pos,neg)
					Wij=np.inner(self.W[l1],self.W[l2])
					diff=Wij+self.b_i[l1]+self.b_k[l2]-np.log10(conValue*Senti_value)


				elif self.word_level_1:   ###using the $J=f_ij*(w_i*w_k+b_i+b_k-log(X_ij*s_i))$ a simple of glove with word sentiment
					Wij=np.inner(self.W[l1],self.W[l2])
					try:
						Senti_value=self.wordSentic[word_index]
					except Exception,e:
						Senti_value=0
				
					if Senti_value==-1:
						Senti_value=self.Alpha
					elif Senti_value==0:
						Senti_value=1
					elif Senti_value==1:
						Senti_value=(1/self.Alpha)
					diff=Wij+self.b_i[l1]+self.b_k[l2]-np.log10(conValue*Senti_value)			 

				if conValue>self.x_max:
					fdiff=diff
					fX_ij=1
				else:
					fdiff=diff*((conValue/self.x_max)**self.alpha)
					fX_ij=(conValue/self.x_max)**self.alpha
				if self.document_level_1:
					cost+=0.5*fdiff.dot(diff) ## loss function 
				elif self.word_level_1 or self.document_level_2:
					cost+=0.5*fdiff*diff

				t4=time()
				'''updating the params using adaptive gradient'''
				fdiff *= self.eta 
				if self.document_level_1:
					inner_WW=self.W[l1].dot(self.W[l2])
					tmpS=(fdiff*(inner_WW+self.b_i[l1]+self.b_k[l2]))
					if tmpS.dot(tmpS)>10:
						tmpS/=tmpS.dot(tmpS)				
					if self.target_center==True:
						self.S[l1]-=tmpS/np.sqrt(self.gradsq_s[l1])
						self.gradsq_s[l1]+=tmpS*tmpS

					else:
						self.S[l2]-=tmpS/np.sqrt(self.gradsq_s[l2])
						self.gradsq_s[l2]+=tmpS*tmpS
					# if np.isnan(tmpS[0]):
					# 	print tmpS,self.gradsq_s[l1]

					if self.target_center:
						dot_F_S_1=fdiff.dot(self.S[l1])
					else:
						dot_F_S_1=fdiff.dot(self.S[l2])
					tmp1=(dot_F_S_1*self.W[l2])				
					tmp2=(dot_F_S_1*self.W[l1])
					if tmp1.dot(tmp1)>10:
						tmp1/=tmp1.dot(tmp1)
					if tmp2.dot(tmp2)>10:
						tmp2/=tmp2.dot(tmp2)
					self.W[l1]-=(tmp1/np.sqrt(self.gradsq_w[l1]))
					self.W[l2]-=(tmp2/np.sqrt(self.gradsq_w[l2]))
					# print "%f\t%f\n"%(abs(np.mean(self.W[l2])),abs(np.mean(self.W[l1])))
					# if l1==0:
					# 	print self.W[l1],l2,tmp1.dot(tmp1)
					self.gradsq_w[l1]+=(tmp1*tmp1)
					self.gradsq_w[l2]+=(tmp2*tmp2)
					'''#update for the sentiment distribution value# '''
					if self.target_center==True:
						self.b_i[l1]-=fdiff.dot(self.S[l1])/np.sqrt(self.gradsq_b_i[l1])
						self.b_k[l2]-=fdiff.dot(self.S[l1])/np.sqrt(self.gradsq_b_k[l2])
					else:
						self.b_i[l1]-=fdiff.dot(self.S[l2])/np.sqrt(self.gradsq_b_i[l1])
						self.b_k[l2]-=fdiff.dot(self.S[l2])/np.sqrt(self.gradsq_b_k[l2])					
					self.gradsq_b_i[l1]+=(diff.dot(diff))
					self.gradsq_b_k[l2]+=(diff.dot(diff))
					# if l1==0:
					# 	print Wij,l2,dot_F_S_1,fdiff,self.S[l1],self.S[l2]
				elif self.word_level_1 or self.document_level_2:
					tmp1=fdiff*self.W[l2]
					tmp2=fdiff*self.W[l1]
					self.W[l1]-=tmp1/np.sqrt(self.gradsq_w[l1])
					self.W[l2]-=tmp2/np.sqrt(self.gradsq_w[l2])
					self.gradsq_w[l1]+=tmp1**2
					self.gradsq_w[l2]+=tmp2**2
					self.b_i[l1]-=fdiff/np.sqrt(self.gradsq_b_i[l1])
					self.b_k[l2]-=fdiff/np.sqrt(self.gradsq_b_k[l2])
					self.gradsq_b_i[l1]+=fdiff**2
					self.gradsq_b_k[l2]+=fdiff**2

				  




			t5=clock()
			# print "each word using %f\n"%(t5-t3)
		# t5=time()
		# print "training thread using %s senconds"%(t5-t1)
		self.res.append(cost)
		return cost
	def train_sglove(self):
		print "it is start training..."
		iter_num=self.iternum
		chunks=chunkize_serial(self.conMatrix, self.chunk_size)

		chunk_lists=[]
		for chunk in chunks:
			chunk_lists.append(chunk)
		print "number of chunk is %d, max_size is %d, min_size is %d"%(len(chunk_lists),len(chunk_lists[0]),len(chunk_lists[-1]))
		iter_size=len(chunk_lists)
		chunks=chunk_lists*iter_num
		threads=[]
		costs=[]
		

		# def log_result(result):
		# 	costs.append(result)
		for i in range(len(chunks)):
			threads.append(threading.Thread(target=self.train_thread,args=(chunks[i],)))	
		# for m in range(iter_num):
		costs=[]
		
		
			
		time_start=time()
		# for iter_i in range(len(chunks)):
		for tp in threads:
			iter_i=threads.index(tp)
			# print iter_i
			if iter_i!=0 and (iter_i+1)%iter_size==0:
			 	tp.setDaemon(True)
			 	tp.start()
				for tp in threads[iter_i-(self.thread_num-1):iter_i+1]:
					tp.join()
				total_cost=0
				ii=0
				for k in range(len(self.res)):
					total_cost+=self.res[k]
				time_end=time()		
			 	print "iter: %d, cost:%f, in %f senconds"%(int(iter_i/iter_size)+1,total_cost/len(self.conMatrix),time_end-time_start)			
			 	time_start=time_end
			 	self.res=[]
			else:
			 	tp.setDaemon(True)
			 	tp.start()				






			# cost=pools.apply_async(self.train_thread, (chunk,))
			# if i!=0 and i%iter_size==0:
			# 	costs.append(cost)
			# 	total_cost=0
			# 	for cost_value in costs:
			# 		costvalue=cost_value.get()
			# 		print "the cost is %f in %dth iter of number %d"%(costvalue,int(i/iter_size),i)
			# 		total_cost+=costvalue
			# 	costs=[]
			# else:
			# 	costs.append(cost)	

		# pools.close()
		# pools.join()
		print "it is finished the training..."

		if self.from_original:

	 		fw=open(self.Wf,'wb')
	 		pickle.dump(self.W,fw) 
	 		pickle.dump(self.S,fw)
	 		

	 		fw_para=open(self.bs,'wb')
	 		pickle.dump(self.b_i,fw_para)
	 		pickle.dump(self.b_k,fw_para)
	 	else:
	 		fw=open(self.Wf+'.succeed','wb')
	 		pickle.dump(self.W,fw) 
	 		pickle.dump(self.S,fw)	 		
	 	if self.final_vector:
			SGlove_dic={}
			for iw in range(len(self.W)):
				SGlove_dic[self.dict_index_re[iw+1]]=self.W[iw]	
			fwF=open('SGlove.pkl','wb')
			pickle.dump(SGlove_dic,fwF)		
 		print "it is finished dumping..."
	def evaluate(self):
		if os.path.exists(self.Wf):
			Wfile=open(self.Wf,'rb')
			self.W=pickle.load(Wfile)
			self.S=pickle.load(Wfile)
		else:
			self.train_thread()

		Simlarity=distanceCalculate()
		Corpus_length=len(self.dict_index)
		for indexi in range(Corpus_length):
			simsM=np.zeros((Corpus_length,Corpus_length))

			# sims=[0]*len(self.dict_index)
			for indexj in range(indexi,Corpus_length):
				if indexj>indexi:

					vec1=self.W[indexi]
					vec2=self.W[indexj]					
					sim=Simlarity.cosineDistance(vec1,vec2)
					simsM[indexi][indexj]=sim
					simsM[indexj][indexi]=sim
					# sims[indexj-1]=sim

		
			
			filew=open('./max_sim_10_senti_depart3.txt','a')
			# for indexii in range(Corpus_length):
			filew.write(str(self.dict_index_re[indexi+1])+'\t')
			b=list(simsM[indexi])
			# sim_max=[]
			for i in xrange(10):
				tmp=max(b)
				# sim_max.append(tmp)
				
				filew.write(str(self.dict_index_re[list(simsM[indexi]).index(tmp)+1]))
				b.remove(tmp)
				if i==9:
					filew.write('\n')
				else:
					filew.write('\t')



			# print np.array(vec1).dot(np.array(vec1))
			# print vec2
			 

def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    import numpy
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[numpy.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble, 
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
		
 



if __name__ == "__main__":
	# build=Build_Dict()
	# # build.read_corpus()
	# dicts,dict_index=build.load_dict()
	# print len(dicts)
	# build.context_matrix()
	# build.information_gain()

	# freeze_support()
	train=Train_SGlove()
	train.train_sglove()
	# train.evaluate()

	 
	# fw=open('test.pkl','wb')
	# a=np.zeros((13800,13800))
	# pickle.dump(a,fw)

