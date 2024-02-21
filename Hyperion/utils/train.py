import torch
from ultralytics import YOLO


def main():
	# Create a new YOLO model from scratch
	model = YOLO('yolov8n.yaml')

	# Load a pretrained YOLO model (recommended for training)
	model = YOLO('yolov8n.pt')

	# Train the model using the custom dataset for 1000 epochs
	results = model.train(data='..\\datasets\\Carla.v20-carla_v20.yolov8\\data.yaml', epochs=1000)

	# Evaluate the model's performance on the validation set
	results = model.val()

	# Perform object detection on an image using the model
	results = model('..\\frame.jpg')

	# Export the model to ONNX format
	success = model.export(format='onnx')


if __name__ == '__main__':
    main()