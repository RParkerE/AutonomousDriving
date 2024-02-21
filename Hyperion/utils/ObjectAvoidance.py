from ultralytics import YOLO

class ObjectDetector:
	def __init__(self):
		pass

	def detect(self, frame):
		model = YOLO('yolov8n.yaml')
		model = YOLO('.\\utils\\best.pt')
		results = model(source=frame)

		return results