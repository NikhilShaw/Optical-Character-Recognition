import scipy.io as sio

train_mat = "IIIT5K/trainCharBound.mat"
test_mat = "IIIT5K/testCharBound.mat"
train_tags = "train.tags"
test_tags = "test.tags"

def create_tags(mat_file, tags_file, data_key):
    tags_fo = open(tags_file, "w")
    data_dir = sio.loadmat(mat_file)
    gts = data_dir[data_key][0]
    for i in range(gts.shape[0]):
        image_path = gts[i][0][0]
        gt = gts[i][1][0]
        tags_fo.write("{} {}\n".format(image_path, gt))

    tags_fo.close()

#create_tags(train_mat, train_tags, "trainCharBound")
#create_tags(test_mat, test_tags, "testCharBound")
unique_chars=set("")
def image_path_and_label(mat_file, data_key):
	global unique_chars
	img_path_and_label=[]
	data_dir = sio.loadmat(mat_file)
	gts = data_dir[data_key][0]
	for i in range(gts.shape[0]):
		global unique_chars
		image_path = gts[i][0][0]
		gt = gts[i][1][0]
		unique_chars= unique_chars.union(set(list(str(gt))))
		img_path_and_label.append((image_path, gt))
	return img_path_and_label
  
train_image_path_and_label = image_path_and_label(train_mat, "trainCharBound")
test_image_path_and_label = image_path_and_label(test_mat, "testCharBound")
print(len(train_path_image_and_label))
print(len(test_path_image_and_label))
#print(unique_chars)
#print(len(unique_chars))
# create dict for char to class 
char2class = {val: index for index, val in enumerate(unique_chars)}
# create dict for class to char
class2char = {index: val for index, val in enumerate(unique_chars)}
print(class2char)



