from ultralytics import YOLO
import cv2
import os

model_apple = YOLO('roboflow_weights/model_2.pt')
# model.train(data="C:/Users/Sabyasachi/PycharmProjects/science exhibition/datasets/bad vs good apples.v4i.yolov5pytorch/data.yaml", epochs=2, imgsz=416)

IMAGE_PATH = input("Enter path of image you wish to segment \n")

result = model_apple.predict(source=cv2.imread(IMAGE_PATH), save=True)
predictions = cv2.resize(cv2.imread("C:/Users/Sabyasachi/PycharmProjects/science exhibition/runs/detect/predict/image0.jpg"),[500, 500], cv2.INTER_AREA)
print(predictions)
cv2.imshow("Predicted boxes and classes", predictions)
cv2.waitKey(0)
os.remove("C:/Users/Sabyasachi/PycharmProjects/science exhibition/runs/detect/predict/image0.jpg")
os.rmdir("C:/Users/Sabyasachi/PycharmProjects/science exhibition/runs/detect/predict")