from roboflow import Roboflow
rf = Roboflow(api_key="IH7U3j2LPFug954Xt2sq")
project = rf.workspace().project("bad-vs-good-apples")
model = project.version(3).model

# infer on a local image
print(model.predict("apple_dataset/validation/images/apple (295).jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("apple_dataset/validation/images/apple (295).jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())