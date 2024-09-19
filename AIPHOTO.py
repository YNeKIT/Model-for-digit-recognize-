import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Transformă imaginile 28x28 într-un vector
    tf.keras.layers.Dense(128, activation='relu'),   # Strat ascuns cu 128 de neuroni
    tf.keras.layers.Dense(10, activation='softmax')  # Strat de ieșire pentru 10 clase
])

# Compilarea modelului
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Antrenarea modelului
model.fit(x_train, y_train, epochs=10)

# Evaluarea modelului
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Acuratețea pe setul de test:', test_acc)

# Salvarea modelului în formatul nativ .keras
model.save('mnist_model.keras')

# Încărcarea modelului
new_model = tf.keras.models.load_model('mnist_model.keras')

# Selectarea și afișarea unei imagini aleatorii din setul de test
random_index = np.random.randint(len(x_test))
random_image = x_test[random_index]
random_label = y_test[random_index]

# Prezicerea cifrei folosind modelul încărcat
predictions = new_model.predict(np.expand_dims(random_image, axis=0))
predicted_label = np.argmax(predictions[0])

# Afișarea imaginii și a predicției
plt.imshow(random_image, cmap=plt.cm.gray)
plt.title(f'Eticheta adevărată: {random_label}, Eticheta prezisă: {predicted_label}')
plt.show()