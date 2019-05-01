import pickle

# =============================================================================
# my_list = [['Fred', 73],['Hello there', 81.9876e-13]]
# 
# pickle_file = open("my_pickled_list.pkl","wb")
# 
# pickle.dump(my_list, pickle_file)
# pickle_file.close()
# 
# 
# pkl_file = open('my_pickled_list.pkl', 'rb')
# 
# data1 = pickle.load(pkl_file)
# pprint.pprint(data1)
# 
# =============================================================================

class Pickle():
    def __init__(self, filename):
        self.filename = filename
    
    def write_pickle(self, write_content):
        pickle_file = open(self.filename, "wb")
        pickle.dump(write_content, pickle_file)
        pickle_file.close()
        
    def read_pickel(self):
        pickle_file = open(self.filename, "rb")
        data = pickle.load(pickle_file)
        pickle_file.close()
        return data
        