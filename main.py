
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import util
import pytesseract
from keras.preprocessing import image
from keras.models import load_model

# định nghĩa các đường dẫn đến cfg, weights , nameclass của model
model_cfg_path = os.path.join(".", "model", "cfg", "yolov3-tiny.cfg")
model_weights_path = os.path.join(".", "model", "weights", "yolov3-tiny_15000.weights")
class_names_path = os.path.join(".", "model", "class.names")
save_dir = "./char_imgs"

img_path = "D:/KhaiThacDuLieu/LicensePlateDetection/pic/car1.jpg"  # đường dẫn đến ảnh của bạn

class_ky_tu = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z']
print(len(class_ky_tu))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load các tên class của mình
with open(class_names_path, "r") as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# load image
img = cv2.imread(img_path)
H, W, _ = img.shape

# chuyển đổi kích thước hình ảnh sao cho phù hợp với mạng YOLO đã cài đặt
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

# get detections
net.setInput(blob)
detections = util.get_outputs(net)

# bboxes, class_ids, confidences
bboxes = []
class_ids = []
scores = []

for detection in detections:
    bbox = detection[:4]  # [x1, x2, x3, x4, x5, x6]
    xc, yc, w, h = bbox
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    bbox_confidence = detection[4]
    class_id = np.argmax(detection[5:])
    score = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

# loại bỏ các bbox có độ chính xác không cao và giữ lại 1 bbox
bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    # lấy tọa độ của biển số xe
    license_plate = img[
        int(yc - (h / 2)) : int(yc + (h / 2)),
        int(xc - (w / 2)) : int(xc + (w / 2)),
        :,
    ].copy()
    img = cv2.rectangle(
        img,
        (int(xc - (w / 2)), int(yc - (h / 2))),
        (int(xc + (w / 2)), int(yc + (h / 2))),
        (0, 255, 0),
        10,
    )
    # xử lý ảnh biển số xe trước khi nhận diện kí tự
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, license_plate_thresh = cv2.threshold(
        license_plate_gray, 103, 255, cv2.THRESH_BINARY_INV
    )

# plt.figure()
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

centers = []
count = 0
# Loop through each license plate
for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    # Lấy tọa độ của biển số xe
    license_plate = img[
        int(yc - (h / 2)) : int(yc + (h / 2)), int(xc - (w / 2)) : int(xc + (w / 2)), :
    ].copy()

    # Chuyển đổi ảnh biển số xe sang ảnh đen trắng
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, license_plate_thresh = cv2.threshold(
        license_plate_gray, 103, 255, cv2.THRESH_BINARY_INV
    )

    # Sử dụng Tesseract OCR để nhận diện ký tự
    custom_config = r"--oem 3 --psm 6"  # Cấu hình Tesseract OCR
    text = pytesseract.image_to_string(license_plate_thresh, config=custom_config)

    # Vẽ kết quả lên ảnh gốc
    img = cv2.putText(
        img,
        text,
        (int(xc - (w / 2)), int(yc - (h / 2))),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # Tách từng ký tự từ vùng chứa ảnh của biển số xe
    contours, _ = cv2.findContours(
        license_plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Lọc contours dựa trên tiêu chí
    filtered_contours = [
        contour
        for contour in contours
        if (cv2.contourArea(contour) > 30 and cv2.contourArea(contour) < 400)
    ]

    filtered_contours.sort(key=lambda x: cv2.boundingRect(x)[0])

    # Duyệt qua từng contour và tạo hình chữ nhật bao quanh ký tự
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        character = license_plate_thresh[y : y + h, x : x + w]

        count += 1
        filename = f"char_{count}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, character)

        # Hiển thị từng ký tự
        plt.figure()
        plt.imshow(character, cmap="gray")
        plt.title(f"Character {len(filtered_contours)}")
        plt.show()
    # cv2.imshow('img',img)
    # cv2.waitKey()


# Đường dẫn đến mô hình CNN
model_path = "D:/KhaiThacDuLieu/LicensePlateDetection/model/cnn.keras"

# Đường dẫn đến thư mục chứa ảnh cần dự đoán
image_folder = "D:/KhaiThacDuLieu/LicensePlateDetection/char_imgs"

# Đọc mô hình từ đường dẫn
model = load_model(model_path)
Bien_so_xe = []
# Duyệt qua từng ảnh trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):  # Đảm bảo chỉ đọc file hình ảnh
        char_img_path = os.path.join(image_folder, filename)

        # Load ảnh và chuyển về kích thước phù hợp với mô hình
        img = image.load_img(char_img_path, target_size=(20,20))  # Thay your_target_size bằng kích thước bạn đã sử dụng khi train mô hình
        img_array = image.img_to_array(img)
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.reshape((20, 20, 1))  # Thêm chiều cao số kênh


        # Dự đoán
        prediction = model.predict(np.array([gray_img]))

        # Xử lý kết quả dự đoán theo nhu cầu của bạn
        # Ví dụ:
        predicted_class_index = np.argmax(prediction)
        # print(len(class_ky_tu))
        # print(predicted_class_index)

        predicted_class = class_ky_tu[predicted_class_index]
        Bien_so_xe.append(class_ky_tu[predicted_class_index])
        # predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"File: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}")
        # print(prediction)

result_string = ''.join(Bien_so_xe)

print("biển số xe: ", result_string)

