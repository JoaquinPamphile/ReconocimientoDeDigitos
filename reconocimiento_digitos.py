import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#Definimos los sets de trainign y testing, X representa la imagen e Y la cateogria, importando el dataset con Keras. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Testeo de que se imprima correctamente la imagen y su valor
image_index = 900 # Cualquier valor hasta 60000
print(y_train[image_index]) 
plt.imshow(x_train[image_index], cmap='gray')
plt.show()

#Obtebemos las dimensiones del dataset, en este caso son de (60000,28,28) -> tenemos 60000 imagenes de 28 por 28 pixeles
x_train.shape

#Antes de entrenar debemos adaptar el formato de los datos
#Necesitamos un formato de 4 dimensiones para poder entrenar el modelo, asi pasamos nuestras imagenes de 3 dimensiones a 4
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) 
x_train.shape
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test.shape
input_shape = (28, 28, 1)

#Asegurarnos que nuestros valores sean flotantes, pues trabajaremos con probabilidades en el output de 0 a 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalizamos el RGB para que pase de escala 0-255 a escala 0-1, dividiendolo por le maximo valor de RGB
x_train /= 255
x_test /= 255

#Comprobamos que cantidad de imagenes tenemos para entrenar y para testear el modelo
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#Creamos el modelo y añadimos las capas con sus respectivas funcionalidades
model = Sequential()

#1- capa de convulsional 2, multiplicaremos matrices de 2 dimensiones, le pasamos el tamaño de la imagen, el tamaño de la matriz con la que generalizaremos, cuanto es el conjunto de imagenes que vamos a convulsionar
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

#2- una vez creada la capa convulsional, le aplicamos el maxpooling de tamaño de ventana 2x2, basicamente nos estaremos moviendo con una ventana de 2x2 por una matriz de 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))

#3- capa flatten que nos permite que la salida se mantenga con el mismo tamaño, no queremos que las operacions sobre la entrada alteren la salida.
model.add(Flatten()) 

#4-parte de las neuronas, una capa de 128 neuronas con una activacion de tipo relu
model.add(Dense(128, activation=tf.nn.relu))

#5. añadimos dropout para no considerar un porcentaje de neuronas en la fase de entranamiento, de esta forma evitamos el sobreaprendizaje
model.add(Dropout(0.2))

#6 definimos la capa de salida, tenemos 10 digitos (del 0 al 9), por lo que necesitamos una neurona de salida para identificar cada imagen y le vamos a aplicar una funcion de activacion softmax
model.add(Dense(10,activation=tf.nn.softmax))

#PROCESO DE COMPILACION, en este proceso ya entrenamos la red
#definimos el optimizador, la funcion de perdida, y la metrica que queremos medir
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#establezco los datos con los que entrenaremos la red y cuantas iteraciones de entrenamiento haremos
model.fit(x=x_train,y=y_train, epochs=10)

#PARTE DE EVALUACION Y TESTEO, en esta parte medimos el accuracy (de 0 a 100)
model.evaluate(x_test, y_test)

#Testeamos el funcionamiento
image_index = 1001  #puede ir entre 0 y 9999
plt.imshow(x_test[image_index].reshape(28, 28),cmap='gray')
plt.show()
pred = model.predict(x_test[image_index].reshape(1, 28,28, 1)) #Nos devuelve el valor de mayor porcentaje (0-9)
print(pred.argmax())