import glob
import os
import sys
import time

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla

import random
import numpy as np
import math
import cv2
import collections

from utils.LaneKeepAssist import LKA
from utils.ObjectAvoidance import ObjectDetector


class TestVehicle:
	def __init__(self, world):
		self.world = world
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]

		self.LKA = LKA()
		self.OA = ObjectDetector()

	def reset(self):
		self.actor_list = []

		self.transform = random.choice(self.world.get_map().get_spawn_points())
		self.ego_vehicle = self.world.spawn_actor(self.model_3, self.transform)
		self.actor_list.append(self.ego_vehicle)

		transform_front_cam = carla.Transform(carla.Location(x=0.8, z=2.6))

		self.front_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
		self.front_cam_bp.set_attribute('image_size_x', f'640')
		self.front_cam_bp.set_attribute('image_size_y', f'480')
		self.front_cam_bp.set_attribute('fov', '110')

		self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
		self.lidar_bp.set_attribute('rotation_frequency','10')

		self.front_cam_sensor = self.world.spawn_actor(self.front_cam_bp, transform_front_cam, attach_to=self.ego_vehicle)
		self.actor_list.append(self.front_cam_sensor)
		self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, transform_front_cam, attach_to=self.ego_vehicle)
		self.actor_list.append(self.lidar_sensor)

		self.front_cam_sensor.listen(lambda data: self.process_frame(data))

	def process_frame(self, data):
		frame = np.array(data.raw_data)
		frame = frame.reshape((480, 640, 4))
		frame = frame[:,:,:3]

		cv2.imwrite("test.jpg", frame)

		self.LKA.lane_detect(frame)
		results = self.OA.detect(frame)


		

if __name__ == '__main__':
	client = carla.Client("127.0.0.1", 2000)
	client.set_timeout(2.0)

	test_car = TestVehicle(client.get_world())
	try:
		test_car.reset()
		time.sleep(5)
	finally:
		for actor in test_car.actor_list:
			actor.destroy()