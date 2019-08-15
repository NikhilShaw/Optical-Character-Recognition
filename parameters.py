# Path 
dir_path="/content/IIIT5K/"
train_mat= dir_path+"trainCharBound.mat"
train_data_key= "trainCharBound"
test_data_key= "testCharBound"
test_mat= dir_path+"testCharBound.mat"
train_tags= "train.tags"
test_tags= "test.tags"
# img and label
CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters) + 1
img_w, img_h, channel = 197, 197, 3

# network 
batch_size = 128
val_batch_size = 16
max_text_len = 9