#@Author:Hazar Bilgin
# cnn model ile çeşitli resimlerin sınıflandırılma modeli veri olarak cifar10 verilerini kullanıyoruz ve modelimizi eğitip 
#hangi resimleri nasıl sınıflandırıldığını modelimize eğitip ekranda çıktı alıp daha sonrasında doğruluk payını grafikleştiriyoruz
from typing import Self
import tensorflow as tf
import keras 
from keras import datasets,layers,models
import matplotlib.pyplot as plt
#(x_train , y_train) , (x_test,y_test)=indirilen veriler gerekli dosyalar tensorflow datasets sayesinde indirilir
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
#veri kümesinin doğru göründüğünü doğrulamak için 
#eğitim kümesindeki ilk 25
#görüntüyü çizdirelim ve her görüntünün altına 
# sınıf adlarını yazdıralım

class_names=['airplane','automobile','bird','cat','deer',
'dog', 'frog','horse','ship','truck']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    #The CIFAR labels happen to be arrays,
    #which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


#set your convolutional layer this 6 line codes are show
#maxpooling and Conv layers

#burdaki girdi bir CNN'in shape tensorlerini alırlar((image_height, image_width, color_channels))
#ve batch boyutları ihmal ederler ve color_channelslar R,G,B Ye denk gelmektedir
#bu CNN modelinde shape girdisi olarak (32,32,3) giriyoruz bu da CIFAR resimlerinin formatı 
# olan şekil girişlerini (32,32,3) işlemek için yapılandırcaktır
#bunu input_shape'in argumanları ileterek ilk katmanı oluşturur
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#şimdiye kadar modelimizin mimarisini gösterelim
model.summary()
print(model.summary)
#Yukarıda, her Conv2D ve MaxPooling2D katmanının çıktısının bir 3B şekil tensörü
# (yükseklik, genişlik, kanallar)   olduğunu görebilirsiniz. Ağda daha derine indikçe genişlik ve yükseklik
# boyutları küçülme eğilimindedir.Her Conv2D katmanı için çıktı kanallarının sayısı, ilk 
# argüman tarafından kontrol edilir (örneğin, 32 veya 64). 
# Tipik olarak, genişlik ve yükseklik küçüldükçe, her Conv2D katmanına daha fazla 
# çıktı kanalı eklemeyi (hesaplamalı olarak) karşılayabilirsiniz.


#--------------------------------------------------------#

#                                 Üzerine Yoğun katmanlar ekleyin
# Modeli tamamlamak için, sınıflandırma gerçekleştirmek için 
# konvolüsyonel tabandan (şekil (4, 4, 64)) son çıktı tensörünü 
# bir veya daha fazla Yoğun katmana besleyeceksiniz. 
# Yoğun katmanlar, giriş olarak vektörleri (1B olan) alırken, 
# mevcut çıktı bir 3B tensördür. İlk olarak, 3B çıktıyı 
# 1B olarak düzleştirecek (veya açacaksınız), ardından 
# üstüne bir veya daha fazla Yoğun katman ekleyeceksiniz. 
# CIFAR'ın 10 çıktı sınıfı vardır, bu nedenle 10 çıktılı 
# son bir Yoğun katman kullanırsınız.

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))

#modelimizn tam mimarisi:
model.summary()

#Modeli derleyin ve eğitin
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
#Modeli değerlendirin (grafikte modelin hata doğruluk payı)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#modelimizn test doğrulunu yazdıralım
print(test_acc)
#@Author:Hazar Bilgin
