import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.models import load_model

# Load data and labels (sử dụng code của bạn để load dữ liệu)
dir = "D:/KhaiThacDuLieu/LicensePlateDetection/data/kyTu/kyTu"
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z']
data = []
labels = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        kyTuImg = cv2.imread(imgPath, 0)

        image = np.array(kyTuImg)
        data.append(image)
        labels.append(label)

# Chuyển danh sách data và labels thành mảng numpy
data = np.array(data)
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Chuyển nhãn thành dạng one-hot
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Biên soạn mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')

model.save("D:/KhaiThacDuLieu/LicensePlateDetection/model/cnn.keras")
# Load mô hình
loaded_model = load_model('D:/KhaiThacDuLieu/LicensePlateDetection/model/cnn.keras')
