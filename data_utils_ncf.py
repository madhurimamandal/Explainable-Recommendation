import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data
import config


def load_all(test_num=100):
	train_data = pd.read_csv(
		config.train_rating, 
		usecols=['UserId', 'ProductId'], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['UserId'].max() + 1
	item_num = train_data['ProductId'].max() + 1

	train_data = train_data.values.tolist()

	
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	val_data = pd.read_csv(
		config.val_rating, 
		usecols=['UserId', 'ProductId'], dtype={0: np.int32, 1: np.int32})
	
	val_data = val_data.values.tolist()


	return train_data, val_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
	def __init__(self, features, train_mat=None, num_ng=0):
		super(NCFData, self).__init__()
		self.features_ps = features
		
		self.train_mat = train_mat
		self.num_item = self.train_mat.shape[1]
		self.num_ng = num_ng
		self.labels = [0 for _ in range(len(features))]

	
	def ng_sample(self):
		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng


	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)
	def __getitem__(self, idx):
		features = self.features_fill 
		labels = self.labels_fill 

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label


class NCFData_ng(data.Dataset):
	def __init__(self, features, train_mat=None, num_ng=0):
		super(NCFData, self).__init__()
		self.features_ps = features
		self.train_mat = train_mat
		self.num_item = self.train_mat.shape[1]
		self.num_ng = num_ng
		self.labels = [0 for _ in range(len(features))]

	
	def ng_sample(self):

		self.features_fill=[]
		self.labels_fill=[]

		for x in self.features_ps:

			self.features_ng = []
			self.features_fill.append(x)
			self.labels_fill.append(1)

			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])
				self.labels_fill.append(0)

			self.features_fill.extend(self.features_ng)


	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)
	def __getitem__(self, idx):
		features = self.features_fill 
		labels = self.labels_fill 

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label
		