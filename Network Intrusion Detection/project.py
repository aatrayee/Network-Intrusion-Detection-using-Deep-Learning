import pandas as pd
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation , Dropout
from sklearn import svm
df = pd.read_csv('train.txt', names = [ 'duration'
, 'protocol_type' 
, 'service'
, 'flag'
, 'src_bytes' 
, 'dst_bytes' 
, 'land' 
, 'wrong_fragment' 
, 'urgent' 
, 'hot' 
, 'num_failed_logins' 
, 'logged_in' 
, 'num_compromised' 
, 'root_shell' 
, 'su_attempted' 
, 'num_root' 
, 'num_file_creations' 
, 'num_shells' 
, 'num_access_files' 
, 'num_outbound_cmds' 
, 'is_host_login'  
, 'is_guest_login' 
, 'count' 
, 'srv_count' 
, 'serror_rate' 
, 'srv_serror_rate' 
, 'rerror_rate' 
, 'srv_rerror_rate' 
, 'same_srv_rate' 
, 'diff_srv_rate' 
, 'srv_diff_host_rate' 
, 'dst_host_count' 
, 'dst_host_srv_count' 
, 'dst_host_same_srv_rate' 
, 'dst_host_diff_srv_rate' 
, 'dst_host_same_src_port_rate' 
, 'dst_host_srv_diff_host_rate' 
, 'dst_host_serror_rate' 
, 'dst_host_srv_serror_rate' 
, 'dst_host_rerror_rate' 
, 'dst_host_srv_rerror_rate' 
, 'diffic' 
, 'class' ])

df['service'].unique().shape
print(len(df[df['class'] == 'normal']))
print(len(df[df['class'] == 'Dos']))
print(len(df[df['class']== 'Probe']))
print(len(df[df['class']== 'U2R']))
print(len(df[df['class']== 'R2L']))
df['protocol_type']= df['protocol_type'].astype('category').cat.codes
df['flag'] = df['flag'].astype('category').cat.codes
df['service'] = df['service'].astype('category').cat.codes

print(df.head())

y_train = df['class'].astype('category').cat.codes
y_train.unique()
y_train = np_utils.to_categorical(y_train,5)
y_train.shape
y_train[2]
del df['class']
x_train = df.values
df_test = pd.read_csv('NSLkdd_test.txt', names = [ 'duration'
, 'protocol_type' 
, 'service'
, 'flag'
, 'src_bytes' 
, 'dst_bytes' 
, 'land' 
, 'wrong_fragment' 
, 'urgent' 
, 'hot' 
, 'num_failed_logins' 
, 'logged_in' 
, 'num_compromised' 
, 'root_shell' 
, 'su_attempted' 
, 'num_root' 
, 'num_file_creations' 
, 'num_shells' 
, 'num_access_files' 
, 'num_outbound_cmds' 
, 'is_host_login'  
, 'is_guest_login' 
, 'count' 
, 'srv_count' 
, 'serror_rate' 
, 'srv_serror_rate' 
, 'rerror_rate' 
, 'srv_rerror_rate' 
, 'same_srv_rate' 
, 'diff_srv_rate' 
, 'srv_diff_host_rate' 
, 'dst_host_count' 
, 'dst_host_srv_count' 
, 'dst_host_same_srv_rate' 
, 'dst_host_diff_srv_rate' 
, 'dst_host_same_src_port_rate' 
, 'dst_host_srv_diff_host_rate' 
, 'dst_host_serror_rate' 
, 'dst_host_srv_serror_rate' 
, 'dst_host_rerror_rate' 
, 'dst_host_srv_rerror_rate' 
, 'diffic' 
, 'class' ])

df_test['protocol_type']= df_test['protocol_type'].astype('category').cat.codes
df_test['flag'] = df_test['flag'].astype('category').cat.codes
df_test['service'] = df_test['service'].astype('category').cat.codes

df_test.head()
y_test = df_test['class'].astype('category').cat.codes
y_test.unique()
y_test = np_utils.to_categorical(y_test,5)
y_test.shape
del df_test['class']
x_test = df_test.values
x_train
y_train[2363]
x_test
y_test[1016]
x_test.shape
y_test.shape
x_train.shape


from keras.models import load_model
from keras.callbacks import EarlyStopping 
#early_stopping_monitor = EarlyStopping(patience=100)
from keras.optimizers import RMSprop 
from keras.callbacks import ModelCheckpoint

opt = RMSprop(lr = 0.0006, decay = 1e-6)


model = Sequential()

model.add(Dense(35, activation = 'sigmoid', input_dim = 42))
model.add(Dense(22, activation = 'sigmoid',))
#model.add(Dense(15, activation = 'sigmoid'))
model.add(Dense(10, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax' ))
model.summary()
#model = model.load_weights('weights_rmsprop_99.35.hdf5')

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])
filepath="weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, 
          batch_size=32,validation_split = 0.33, nb_epoch=100, verbose=1, callbacks = callbacks_list)

#model.predict

score = model.evaluate(x_test, y_test, verbose=2)
print(score)

from keras.models import model_from_json

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
	


#model = model_from_json(open('weights_rmsprop_99.35.json').read())

model = model.load_weights('weights_rmsprop_99.35.hdf5')

model.compile(loss='categorical_crossentropy',
              optimizer= opt1,
              metrics=['accuracy'])
			  
model.save_weights('weights_rmsprop_99.35.hdf5', overwrite = False)
y_pred = model.predict(x_test[0:])
import numpy as np
np.argmax(y_pred, axis =1)
np.argmax(y_test, axis = 1 )
print(confusion_matrix(np.argmax(y_test,axis = 1),np.argmax( y_pred, axis = 1)))

from keras.models import model_from_json
from keras.optimizers import RMSprop
opt = RMSprop(lr = 0.0006, decay = 1e-6)


# load json and create model
json_file = open('best/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best/weights-improvement-153-THEBEST-1.00.hdf5")
print("Loaded model from disk")
 
# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])


#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

y_pred = loaded_model.predict(x_test[0:])
from sklearn.metrics import confusion_matrix


matrix = confusion_matrix(np.argmax(y_test,axis = 1),np.argmax( y_pred, axis = 1))
classes = ['Dos', 'probe', 'R2L', 'U2R','normal']
matrix
