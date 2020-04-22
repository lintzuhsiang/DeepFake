from tensorflow.keras.callbacks import LearningRateScheduler
from model import define_model
import h5py
from sklearn.metrics import log_loss, f1_score

df_model=define_model()
df_model.load_weights('./MesoInception_DF')
f2f_model=define_model()
f2f_model.load_weights('./MesoInception_F2F')

file = h5py.File('train2.h5','r')

X,y,Val_X, Val_y = file['train_imgs'], file['train_gts'],file['test_imgs'],file['test_gts']
y = y[:len(X)]
Val_y = Val_y[:len(Val_X)]
print(X.shape,y.shape,Val_X.shape,Val_y.shape)
lrs=[1e-3,5e-4,1e-4]
def schedule(epoch):
    return lrs[epoch]

LOAD_PRETRAIN=True
import gc
kfolds=5
losses=[]
if LOAD_PRETRAIN:
    import keras.backend as K
    df_models=[]
    f2f_models=[]
    i=0
    while len(df_models)<kfolds:
        model=define_model((150,150,3))
        
        if i==0:
            model.summary()
        #model.load_weights('../input/meso-pretrain/MesoInception_DF')
        for new_layer, layer in zip(model.layers[1:-8], df_model.layers[1:-8]):
            new_layer.set_weights(layer.get_weights())
        model.fit([X],[y],epochs=1,callbacks=[LearningRateScheduler(schedule)],shuffle='batch')
        pred=model.predict([Val_X])
        loss=log_loss(Val_y,pred)
        losses.append(loss)
        print('fold '+str(i)+' model loss: '+str(loss))
        df_models.append(model)
        # K.clear_session()
        del model
        gc.collect()
        i+=1
    i=0
    while len(f2f_models)<kfolds:
        model=define_model((150,150,3))
        #model.load_weights('../input/meso-pretrain/MesoInception_DF')
        for new_layer, layer in zip(model.layers[1:-8], f2f_model.layers[1:-8]):
            new_layer.set_weights(layer.get_weights())
        model.fit([X],[y],epochs=2,callbacks=[LearningRateScheduler(schedule)],shuffle='batch')
        pred=model.predict([Val_X])
        loss=log_loss(Val_y,pred)
        losses.append(loss)
        print('fold '+str(i)+' model loss: '+str(loss))
        f2f_models.append(model)
        # K.clear_session()
        del model
        gc.collect()
        i+=1
        models=f2f_models+df_models
else:
    models=[]
    i=0
    while len(models)<kfolds:
        model=define_model((150,150,3))
        if i==0:
            model.summary()
        model.fit([X],[y],epochs=2,callbacks=[LearningRateScheduler(schedule)])
        pred=model.predict([Val_X])
        loss=log_loss(Val_y,pred)
        losses.append(loss)
        print('fold '+str(i)+' model loss: '+str(loss))
        if loss<0.68:
            models.append(model)
        else:
            print('loss too bad, retrain!')
        # K.clear_session()
        del model
        gc.collect()
        i+=1

