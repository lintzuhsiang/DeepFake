

# move images in /DeepFake00/DeepFake00 to /DeepFake00
import re
import os
import json
import random
from sklearn.utils import shuffle
import h5py
from PIL import Image
from numpy import asarray

for dir_ in os.listdir():
    if dir_ == '.DS_Store': continue
    if os.path.isdir(dir_):
        for file in os.listdir(dir_+'/'+dir_):
            shutil.move(dir_+'/'+dir_+ '/' + file,dir_)
        shutil.rmtree(dir_+'/'+dir_)
    else:
        newname = 'DeepFake' + re.sub("[^0-9]","",dir_) +'.json'
        os.rename(dir_,newname)

images = {}
images_R = []
images_F = []

for name in os.listdir():   
    if os.path.isdir(name) and name.startswith('DeepFake'):  #/DeepFake00
        images[name] = [pic for pic in os.listdir(name)]

for folder in images:
    with open(folder+'.json','r') as json_f:
        datas = json.load(json_f)
        for data in datas:
            name = data.split('.')[0] + '.jpg'
            if datas[data]['label'] == 'REAL':
                images_R.append(folder+"/"+name)
            else: images_F.append(folder+"/"+name)

random.shuffle(images_R)
random.shuffle(images_F)
images_F = images_F[:len(images_R)]

images_RL = [1] * len(images_R)
images_FL = [0] * len(images_F)

TRAIN_RATE = 0.7
train_X = images_R[:int(TRAIN_RATE * len(images_R))] + images_F[:int(TRAIN_RATE * len(images_F))]
train_Y = images_RL[:int(TRAIN_RATE * len(images_RL))] + images_FL[:int(TRAIN_RATE * len(images_FL))]
test_X = images_R[int(TRAIN_RATE * len(images_R)):]+ images_F[int(TRAIN_RATE * len(images_F)):]
test_Y = images_RL[int(TRAIN_RATE * len(images_RL)):]+ images_FL[int(TRAIN_RATE * len(images_FL)):]
train_X, train_Y = shuffle(train_X,train_Y)
test_X, test_Y = shuffle(test_X,test_Y)

X,y,Val_X,Val_y = [],[],[],[]

for item,label in zip(train_X,train_Y):
    try:
        image = Image.open(item)
        data = asarray(image)
        X.append(data)
        y.append(label)
    except: pass
    
for item,label in zip(test_X,test_Y):
    try:
        image = Image.open(item)
        data = asarray(image)
        Val_X.append(data)
        Val_y.append(label)
    except: pass

np.save("X.npy",X)
np.save("y.npy",y)
np.save("Val_X.npy",Val_X)
np.save("Val_y.npy",Val_y)

# from skimage import io
# import torchvision.transforms as transforms
# # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
# # normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])

# trans = transforms.Compose([
#     transforms.ToTensor(),
#     # normalize
# ])

# #(150, 150,3)
# hf = h5py.File("train.h5", "w")
# hf.create_dataset('train_gts', data=np.asarray(train_Y))
# hf.create_dataset('test_gts', data=np.asarray(test_Y))


# first = True
# for item in train_X:
#     try:
#         img = io.imread(item)
#         # img = trans(img).numpy()
#         # img = np.swapaxes(img,0,2)  # (150,150,3)
#         if first:
#             dataset1 = hf.create_dataset("train_imgs", (1, img.shape[0], img.shape[1], img.shape[2]), maxshape=(None, img.shape[0], img.shape[1], img.shape[2]))
#             dataset1[0] = img
#             first = False
#             print(dataset1.shape)
#         else:
#             shape = dataset1.shape
#             dataset1.resize([shape[0] + 1, shape[1], shape[2], shape[3]])
#             dataset1[shape[0]:shape[0] + 1] = img
#             print(img.shape)
#     except: pass


# first = True
# for item in test_X:
#     try:
#         img = io.imread(item)
#         # img = trans(img).numpy()
#         # img = np.swapaxes(img,0,2) 
#         if first:
#             dataset1 = hf.create_dataset("test_imgs", (1, img.shape[0], img.shape[1], img.shape[2]), maxshape=(None, img.shape[0], img.shape[1], img.shape[2]))
#             dataset1[0] = img
#             first = False
#         else:
#             shape = dataset1.shape
#             dataset1.resize([shape[0] + 1, shape[1], shape[2], shape[3]])
#             dataset1[shape[0]:shape[0] + 1] = img
#             print(img.shape)
#     except: pass

# hf.close()


