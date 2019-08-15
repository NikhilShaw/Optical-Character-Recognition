import cv2, os, random
import numpy as np
import scipy.io as sio
from keras.preprocessing.sequence import pad_sequences
from parameters import *

def labels_to_text(labels):     # letters index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text letter
    return list(map(lambda x: letters.index(x), text))

class text_image_generator:
	def __init__(self, dir_path):
		self.img_dirpath= dir_path
		self.img_dir= os.listdir(self.img_dirpath)
		self.img_w= img_w
		self.img_h= img_h
		self.channel= channel
		self.batch_size= batch_size
		self.max_text_len= max_text_len
		self.n= len(self.img_dir)
		self.indexes= list(range(self.n))
		self.cur_index= 0
		self.imgs= np.zeros((self.n , self.img_h, self.img_w, self.channel))
		self.image_path_2_label={}
		self.texts = []
		self.del_list=[]

	# loading mat file and extracting image tags
	def create_tags(self, mat_file, tags_file, data_key):
		if os.path.isfile(tags_file):
			return
		tags_fo = open(tags_file, "w")
		data_dir = sio.loadmat(mat_file)
		gts = data_dir[data_key][0]
		for i in range(gts.shape[0]):
			image_path = gts[i][0][0]
			gt = gts[i][1][0]
			tags_fo.write("{} {}\n".format(image_path, gt))
		tags_fo.close()

	# returns dict with  key: img path ;value: label
	def image_path_and_label(self, mat_file, data_key):
		img_path_2_label={}
		data_dir = sio.loadmat(mat_file)
		gts = data_dir[data_key][0]
		for i in range(gts.shape[0]):
			image_path = gts[i][0][0]
			gt = gts[i][1][0]
			img_path_2_label[self.img_dirpath+(image_path.split('/')[-1])]= gt
		return img_path_2_label

	def preprocess(self):
		if self.img_dirpath.split("/")[-1]=="train":
			self.create_tags(train_mat, train_tags, train_data_key)
			self.image_path_2_label = self.image_path_and_label(train_mat, train_data_key)
		elif self.img_dirpath.split("/")[-1]=="test":
			self.create_tags(test_mat, test_tags, test_data_key)
			self.image_path_2_label = self.image_path_and_label(test_mat, test_data_key)
		else:
			print("error[1]")

	def build_data(self):
		print("Loading and preprocessing " + str(self.n) +" images")
		self.preprocess()
		for i, img_file in enumerate(self.img_dir):
			img= cv2.imread(self.img_dirpath + "/" + img_file)
			img = cv2.resize(img, (self.img_w, self.img_h))
			img = img.astype(np.float32)
			img = (img / 255.0) * 2.0 - 1.0
			self.imgs[i, :, :, :] = img
			self.texts.append(self.image_path_2_label[self.img_dirpath + img_file])
		
		# if all the images were loaded then true
		if len(self.texts) == self.n:
			print(self.n, " Image Loading finish...")
		else:
			print(self.n, len(self.texts))

	# return random sample for a batch
	def next_sample(self):
		self.cur_index+=1
		if self.cur_index>=self.n:
			self.cur_index=0
			random.shuffle(self.indexes)
		r_index= self.indexes[self.cur_index]
		return self.imgs[r_index], self.texts[r_index]

	# returns a batch 
	def next_batch(self):
		while True:
			x_data= np.ones([self.batch_size, self.img_w, self.img_h, channel])
			y_data= np.ones([self.batch_size, self.max_text_len])
	
			input_length= np.ones((self.batch_size, 1))
			label_length= np.ones((self.batch_size, 1))
			for i in range(self.batch_size):
				img, text= self.next_sample()
				x_data[i]= img
				y_data[i]= pad_sequences([text_to_labels(text)], maxlen= max_text_len, padding='post', value=-1)
				label_length= len(y_data[i])
				# converting into numpy array
				x_data[i]= np.array(x_data[i])
				y_data[i]= np.array(y_data[i])
			
			inputs= {'the_input': x_data, 'the_labels': y_data, 'input_length': input_length, 'label_length': label_length}
			outputs= {'ctc': np.zeros([self.batch_size])}
			yield inputs, outputs