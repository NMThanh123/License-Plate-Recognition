# import time
# s = time.time()
# import onnxruntime as ort
# import cv2
# import numpy as np

# # Load the ONNX model
# onnx_model_path = 'model/det/det.onnx'  # Replace with the path to your ONNX model
# ort_session = ort.InferenceSession(onnx_model_path)

# # Load and preprocess an input image
# input_image_path = 'Bicycles/0000_00532_b.jpg'  # Replace with the path to your input image
# input_shape = (1, 3, 640, 640)  # Modify based on your model's input shape

# input_image1 = cv2.imread(input_image_path)
# input_image = cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB)
# input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
# input_image = input_image.astype(np.float32)
# input_image /= 255.0  # Normalize pixel values to [0, 1]
# input_image = np.transpose(input_image, (2, 0, 1))  # Change to NCHW format
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# # Perform inference
# outputs = ort_session.run(None, {'images': input_image})
# output = outputs[0]

# def bbox(box):
#     x1 = int(box[0] - box[2]/2)
#     y1 = int(box[1] - box[3]/2)
#     x2 = int(box[0] + box[2]/2)
#     y2 = int(box[1] + box[3]/2)
#     return x1, y1, x2, y2
# a = output[0][4]
# b = np.where(a == np.max(a))[0][0]
# box=output[0,:,b][:4]
# print(bbox(box))

a = 4

print('{}')