from tensorflow.python.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPool2D
from tensorflow.python.keras import Sequential
from tensorflow.keras.utils import to_categorical

(x_train , y_train) , (x_test , y_test) = fashion_mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#%%
from tensorflow.python.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss' , mode='min' , verbose=1)

model = Sequential()

model.add(Conv2D(filters=32, input_shape=(28,28,1) ,
                 kernel_size=(3,3) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, input_shape=(28,28,1) ,
                 kernel_size=(3,3) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128 , activation='relu'))
model.add(Dense(10 , activation='softmax'))

model.compile(optimizer='adam' , loss='categorical_crossentropy'
              , metrics=['accuracy'])

#%%
model.fit(x_train,y_train_cat,epochs=100 ,
          validation_data=(x_test,y_test_cat) , callbacks=[early_stop])

#%%
import pandas as pd
matric_loss = pd.DataFrame(model.history.history)
matric_loss[['accuracy', 'val_accuracy', ]].plot()

#%%
from sklearn.metrics import confusion_matrix,classification_report
predict = model.predict_classes(x_test)
classification_report(y_test,predict)

#%%
labels = ['T-shirt/top' , 'Trouser/pants' ,'Pullover shirt' , 'Dress'
    ,'Coat' , 'Sandal' ,'Shirt' ,'Sneaker' ,'Bag' ,'Ankle boot']

for i in range(100):
    if y_test[i]!=predict[i]:
        print(i)

#%%
no = 439
plt.imshow(x_test[no].reshape(28,28))
plt.title(str(labels[predict[no]]) + "  p - r  " + str(labels[y_test[no]]))

