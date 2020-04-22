from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import h5py
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
from sklearn.decomposition import PCA
from image_transformer import RGB2GrayTransformer, HogTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump,load
from sklearn.metrics import log_loss, f1_score, roc_auc_score
import matplotlib as plt
from skimage.feature import hog
from PIL import Image
import cv2

load_train = False
file = h5py.File('train.h5','r')
X,y,Val_X, Val_y = file['train_imgs'], file['train_gts'],file['test_imgs'],file['test_gts']
y = y[:len(X)]
Val_y = Val_y[:len(Val_X)]
X = np.load("X.npy")
y = np.load("y.npy")
Val_X = np.load("Val_X.npy")
Val_y = np.load("Val_y.npy")

hogify = HogTransformer()
scalify = StandardScaler()

#Grayify
if not load_train:
    X_train_gray0 = np.empty((len(X),X[0].shape[0],X[0].shape[1]))
    print(X_train_gray0.shape)
    count = 0
    for i in range(0,len(X)):
        count += 1
        if count%100 == 0: print(count)
        gray = cv2.cvtColor(X[i],cv2.COLOR_BGR2GRAY)
        X_train_gray0[i,:,:] = gray

    np.save("X_train_gray0",X_train_gray0)

    X_test_gray0 = np.empty((len(Val_X),Val_X[0].shape[0],Val_X[0].shape[1]))
    count = 0
    for i in range(0,len(Val_X)):
        count += 1
        if count%100 == 0: print(count)
        gray = cv2.cvtColor(Val_X[i],cv2.COLOR_BGR2GRAY)
        X_test_gray0[i,:,:] = gray

    np.save("X_test_gray0",X_test_gray0)
else:
    X_train_gray0 = np.load("X_train_gray0.npy")
    X_test_gray0 = np.load("X_test_gray0.npy")


# #Hogify
# if not load_train:
#     X_train_hog0 = np.empty((len(X),X_train_gray0.shape[1],X_train_gray0.shape[2]))
#     count = 0
#     for i in range(0,len(X)):
#         count += 1
#         if count%100 == 0: print(count)
#         hog_image = hog(X_train_gray0[i],visualize=True)
#         X_train_hog0[i,:,:] = hog_image
#     np.save("X_train_hog0",X_train_hog0)

#     X_test_hog0 = np.empty((len(Val_X),X_test_gray0.shape[1],X_test_gray0.shape[2]))
#     count = 0
#     for i in range(0,len(Val_X)):
#         count += 1
#         if count%100 == 0: print(count)
#         hog_image = hog(X_test_gray0[i],visualize=True)
#         X_test_hog0[i,:,:] = hog_image
#     np.save("X_test_hog0",X_test_hog0)
# else:
#     X_train_prepared0 = np.load("X_train_hog0.npy")
#     X_test_prepared0 = np.load("X_test_hog0.npy")

# #Normalize
# X_train_prepared0 = scalify.fit_transform(X_train_hog0.reshape((-1,150*150)))
# X_test_prepared0 = scalify.fit_transform(X_test_hog0.reshape((-1,150*150)))

img0 = Image.fromarray(X_train_gray[0], 'L')
img0.save('grayify.png')
img0.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
fd, hog_image = hog(X_train_gray0[0], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
ax1.imshow(hog_image, cmap=plt.cm.gray)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.savefig("hog.png")











if load_train:
    X_train_hog = hogify.fit_transform(X_train_gray0)
    X_train_prepared = scalify.fit_transform(X_train_hog)
    np.save("X_train_prepared",X_train_prepared)
else:
    X_train_prepared = np.load("X_train_prepared.npy")

if load_train:
    X_test_gray = grayify.fit_transform(Val_X)
    X_test_hog = hogify.fit_transform(X_test_gray)
    X_test_prepared0 = scalify.fit_transform(X_test_hog0)
    np.save("X_test_prepared",X_test_prepared)
else:
    X_test_prepared = np.load("X_test_prepared.npy")


#Dimension reduction
pca = PCA(n_components=2000)
pca.fit(X_train_prepared)
X_train_prepared0 = pca.transform(X_train_prepared)
X_test_prepared0 = pca.transform(X_test_prepared)

print(X_train_prepared.shape)
print(X_test_prepared.shape)

np.save("X_train_prepared",X_train_prepared)
np.save("X_test_prepared",X_test_prepared)


#plot PCA
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_[:10000]))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Explained Variance')
plt.savefig('PCA.png')

pk.dump(pca, open("pca3-2000.pkl","wb"))


if not os.path.exists("sgd.joblib"):
    sgd = SGDClassifier(loss='modified_huber',random_state=40,max_iter=5000,tol=1e-4,n_jobs=-1)
    sgd.fit(X_train_prepared0,y)
    dump(sgd,"sgd.joblib")
else: 
    sgd = load("sgd.joblib")

print("sgd load")
if not os.path.exists("gbc.joblib"):
    gbc = GradientBoostingClassifier(random_state=40)
    gbc.fit(X_train_prepared0,y)
    dump(gbc,"gbc.joblib")
else:
    gbc = load("gbc.joblib")

print("gbc load")
if not os.path.exists("adb.joblib"):
    adb = AdaBoostClassifier(random_state=40)
    adb.fit(X_train_prepared,y)
    dump(adb,"adb.joblib")
else:
    adb = load("adb.joblib")

print("adb load")
if not os.path.exists("dtc.joblib"):
    dtc = DecisionTreeClassifier(random_state=40)
    dtc.fit(X_train_prepared,y)
    dump(dtc,"dtc.joblib")
else:
    dtc = load("dtc.joblib")

print("dtc load")



y_pred_sgd = sgd.predict(X_test_prepared)
print("sgd percentage: ",100*np.sum(y_pred_sgd==Val_y)/len(Val_y))

y_pred_gbc = gbc.predict(X_test_prepared)
print("gbc percentage: ",100*np.sum(y_pred_gbc==Val_y)/len(Val_y))

y_pred_adb = adb.predict(X_test_prepared0)
print("adb percentage: ",100*np.sum(y_pred_adb==Val_y)/len(Val_y))

y_pred_dtc = dtc.predict(X_test_prepared0)
print("dtc percentage: ",100*np.sum(y_pred_dtc==Val_y)/len(Val_y))


y_score_sgd = f1_score(Val_y,y_pred_sgd)
print("sgd f1 score: ",y_score_sgd)
y_score_gbc = f1_score(Val_y,y_pred_gbc)
print("gbc f1 score: ",y_score_gbc)
y_score_adb = f1_score(Val_y,y_pred_adb)
print("adb f1 score: ",y_score_adb)
y_score_dtc = f1_score(Val_y,y_pred_dtc)
print("dtc f1 score: ",y_score_dtc)

auc_sgd = roc_auc_score(Val_y,y_pred_sgd)
auc_gbc = roc_auc_score(Val_y,y_pred_gbc)
auc_adb = roc_auc_score(Val_y,y_pred_adb)
auc_dtc = roc_auc_score(Val_y,y_pred_dtc)

print("sgd auc: ",auc_sgd)
print("gbc auc: ",auc_gbc)
print("adb auc: ",auc_adb)
print("dtc auc: ",auc_dtc)


# y_sgd_predict = sgd.predict_proba(X_test_prepared)
y_gbc_predict = gbc.predict_proba(X_test_prepared)
y_adb_predict = adb.predict_proba(X_test_prepared)
y_dtc_predict = dtc.predict_proba(X_test_prepared)

np.save("y_gbc_predict",y_gbc_predict)
np.save("y_adb_predict",y_adb_predict)
np.save("y_dtc_predict",y_dtc_predict)

