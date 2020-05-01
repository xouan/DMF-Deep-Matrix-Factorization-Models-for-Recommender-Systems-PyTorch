
import numpy as np
from scipy.sparse import csr_matrix

class Dataset(object):

    def __init__(self, path):
        self.train_ratings, self.train_num_users, self.train_num_items = self.load_train_rating_file_as_list(path + ".train.rating")
        self.test_ratings, self.test_num_users, self.test_num_items = self.load_test_rating_file_as_list(path + ".test.rating")
        self.num_users = max(self.train_num_users, self.test_num_users)
        self.num_items = max(self.train_num_items, self.test_num_items)
        self.test_negative = self.load_negative_file(path + ".test.negative")
        self.user_item_rating_indices = self.get_user_item_matrix_indices()
        self.user_indices, self.item_incides, self.rating_data = self.user_item_rating_indices
        assert len(self.test_ratings) == len(self.test_negative)
        self.train_dict = self.get_train_dict()

    def load_test_rating_file_as_list(self, filename):
        test_ratings = []
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                test_ratings.append([user, item])
                line = f.readline()
            test_num_users = num_users + 1
            test_num_items = num_items + 1
        return test_ratings, test_num_users, test_num_items
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_train_rating_file_as_list(self, filename):
        '''
            return: [[user, item, rating]]
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            max_items = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        train_num_users = num_users + 1
        train_num_items = num_items + 1
        # Construct matrix
        train_ratings = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                train_ratings.append([user, item, rating])
                line = f.readline()    
        return train_ratings, train_num_users, train_num_items

    def get_user_item_matrix_indices(self):
        user_indices, item_indices, ratings = [], [], []
        for i in self.train_ratings:
            user_indices.append(i[0])
            item_indices.append(i[1])
            ratings.append(1)
        return [np.array(user_indices), np.array(item_indices), np.array(ratings)]

    def get_user_item_interact_list(self):
        user_item_interact = []
        user, item, rate = [], [], []
        user_idx = int(0)
        for i in self.train_ratings:
            print(i[0])
            if user_idx != i[0]:
                user_item_interact.append([user, item, rate])
                user_idx += 1
                user, item, rate = [], [], []
            else:
                user.append(i[0])
                item.append(i[1])
                rate.append(i[2])
        return user_item_interact

    def get_item_user_interact_list(self):
        item_user_interact = []
        user, item, rate = [], [], []
        item_idx = 0
        for i in self.train_ratings:
            if item_idx != i[1]:
                item_user_interact.append([user, item, rate])
                item_idx += 1
                user, item, rate = [], [], []
            else:
                user.append(i[0])
                item.append(i[1])
                rate.append(i[2])
        return item_user_interact

    def get_train_instances(self, num_negative):
        user, item, rate = [], [], []
        for i in self.train_ratings:
            user.append(i[0])
            item.append(i[1])
            rate.append(1)
            for t in range(num_negative):
                j = np.random.randint(self.num_items)
                while (i[0], j) in self.train_dict:
                    j = np.random.randint(self.num_items)
                user.append(i[0])
                item.append(j)
                rate.append(0)
        return [np.array(user), np.array(item), np.array(rate)]


    def get_user_and_item_matrix(self):
        rom = np.random.rand(1, 100)
        user_matrix = self.user_item_matrix
        item_matrix = self.user_item_matrix.T
        return user_matrix, item_matrix

    def get_train_dict(self):
        data_dict = {}
        for i in self.train_ratings:
            data_dict[(i[0], i[1])] = i[2]
        return data_dict

    def get_item_sparse_matrix(self):
        num_users, num_items = self.num_users, self.num_items
        user_indices, item_incides, rating_data = self.user_item_rating_indices
        item_sparse_matrix = csr_matrix((rating_data, (item_incides, user_indices)), shape=(num_items, num_users))
        return item_sparse_matrix

    def get_user_sparse_matrix(self):
        user_sparse_matrix = self.get_item_sparse_matrix().T
        return user_sparse_matrix
