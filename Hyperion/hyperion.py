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


class Hyperion:
	def __init__(self, world):
		self.world = world
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]

		self.frames = {}
		self.imgBuffer = collections.deque(maxlen=2)
		self.imgWideBuffer = collections.deque(maxlen=2)

		self.LKA = LKA()
		self.OA = ObjectDetector()

	def reset(self):
		self.actor_list = []

		self.transform = random.choice(self.world.get_map().get_spawn_points())
		self.ego = self.world.spawn_actor(self.model_3, self.transform)
		self.actor_list.append(self.ego)

		# Sony IMX623
		fisheye_cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
		fisheye_cam_blueprint.set_attribute('image_size_x', '1937')
		fisheye_cam_blueprint.set_attribute('image_size_y', '1553')
		fisheye_cam_blueprint.set_attribute('fov', '195')

		
		# Sony IMX728
		wide_angle_cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
		wide_angle_cam_blueprint.set_attribute('image_size_x', '3857')
		wide_angle_cam_blueprint.set_attribute('image_size_y', '2177')
		wide_angle_cam_blueprint.set_attribute('fov', '120')

		
		# Sony IMX728
		cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
		cam_blueprint.set_attribute('image_size_x', '3904')
		cam_blueprint.set_attribute('image_size_y', '2177')
		cam_blueprint.set_attribute('fov', '70')

		
		# Sony IMX728
		tele_cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
		tele_cam_blueprint.set_attribute('image_size_x', '3857')
		tele_cam_blueprint.set_attribute('image_size_y', '2177')
		tele_cam_blueprint.set_attribute('fov', '30')

		
		# Luminar Iris
		lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
		lidar_blueprint.set_attribute('channels', '2')
		lidar_blueprint.set_attribute('range', '600')
		lidar_blueprint.set_attribute('points_per_second', '2700000')
		lidar_blueprint.set_attribute('upper_fov', '26')
		lidar_blueprint.set_attribute('lower_fov', '0')
		lidar_blueprint.set_attribute('horizontal_fov', '120')

		
		# Hella Corner SRR
		corner_and_side_radar_blueprint = self.world.get_blueprint_library().find('sensor.other.radar')
		corner_and_side_radar_blueprint.set_attribute('horizontal_fov', '160')
		corner_and_side_radar_blueprint.set_attribute('range', '200')
		corner_and_side_radar_blueprint.set_attribute('vertical_fov', '36')

		
		# Continental Front Imaging Radar
		front_imaging_radar_blueprint = self.world.get_blueprint_library().find('sensor.other.radar')
		front_imaging_radar_blueprint.set_attribute('horizontal_fov', '120')
		front_imaging_radar_blueprint.set_attribute('range', '300')
		front_imaging_radar_blueprint.set_attribute('vertical_fov', '40')

		
		# Continental Front LRR
		front_long_range_radar_blueprint = self.world.get_blueprint_library().find('sensor.other.radar')
		front_long_range_radar_blueprint.set_attribute('horizontal_fov', '120')
		front_long_range_radar_blueprint.set_attribute('range', '250')
		front_long_range_radar_blueprint.set_attribute('vertical_fov', '30')

		
		# Continental Rear LRR
		rear_long_range_radar_blueprint = self.world.get_blueprint_library().find('sensor.other.radar')
		rear_long_range_radar_blueprint.set_attribute('horizontal_fov', '30')
		rear_long_range_radar_blueprint.set_attribute('range', '255')
		rear_long_range_radar_blueprint.set_attribute('vertical_fov', '20')

		
		#  Bosch MMP, Continental SC13S, u-blox ZED-F9K in C103-F9K
		imu_blueprint = self.world.get_blueprint_library().find('sensor.other.imu')
		gps_blueprint = self.world.get_blueprint_library().find('sensor.other.gnss')

		
		# TODO: Get proper placements for all sensors in relation to car
		# Place where the sensors should be on the car
		transform_front_cam = carla.Transform(carla.Location(x=0.8, z=2.6))
		transform_rear_cam = carla.Transform(carla.Location(x=-0.5, z=0.9), carla.Rotation(yaw=180))
		transform_front_left_cam = carla.Transform(carla.Location(x=-0.5, y=-1.0, z=0.5), carla.Rotation(yaw=45))
		transform_front_right_cam = carla.Transform(carla.Location(x=-0.5, y=1.0, z=0.5), carla.Rotation(yaw=315))
		transform_rear_left_cam = carla.Transform(carla.Location(x=-0.5, y=-1.0, z=0.5), carla.Rotation(yaw=135))
		transform_rear_right_cam = carla.Transform(carla.Location(x=-0.5, y=1.0, z=0.5), carla.Rotation(yaw=225))
		transform_front_left_corner_radar = carla.Transform(carla.Location(x=-2.6, y=-1.0, z=-0.25), carla.Rotation(yaw=45))
		transform_front_right_corner_radar = carla.Transform(carla.Location(x=-2.6, y=1.0, z=-0.25), carla.Rotation(yaw=315))
		transform_rear_left_corner_radar = carla.Transform(carla.Location(x=2.6, y=-1.0, z=-0.25), carla.Rotation(yaw=135))
		transform_rear_right_corner_radar = carla.Transform(carla.Location(x=2.6, y=1.0, z=-0.25), carla.Rotation(yaw=225))
		transform_left_side_radar = carla.Transform(carla.Location(y=-1.0, z=-0.25), carla.Rotation(yaw=90))
		transform_right_side_radar = carla.Transform(carla.Location(y=1.0, z=-0.25), carla.Rotation(yaw=270))


		# CAMERAS #
		front_wide_cam_sensor = self.world.spawn_actor(wide_angle_cam_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_wide_cam_sensor)
		front_wide_cam_sensor.listen(lambda data: self.data_store(data, 'front_wide_cam_sensor'))

		rear_wide_cam_sensor = self.world.spawn_actor(wide_angle_cam_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_wide_cam_sensor)
		rear_wide_cam_sensor.listen(lambda data: self.data_store(data, 'rear_wide_cam_sensor'))

		front_tele_cam_sensor = self.world.spawn_actor(tele_cam_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_tele_cam_sensor)
		front_tele_cam_sensor.listen(lambda data: self.data_store(data, 'front_tele_cam_sensor'))

		rear_tele_cam_sensor = self.world.spawn_actor(tele_cam_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_tele_cam_sensor)
		rear_tele_cam_sensor.listen(lambda data: self.data_store(data, 'rear_tele_cam_sensor'))

		front_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_cam_sensor)
		front_cam_sensor.listen(lambda data: self.data_store(data, 'front_cam_sensor'))

		rear_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_cam_sensor)
		rear_cam_sensor.listen(lambda data: self.data_store(data, 'rear_cam_sensor'))

		front_left_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_front_left_cam, attach_to=self.ego)
		self.actor_list.append(front_left_cam_sensor)
		front_left_cam_sensor.listen(lambda data: self.data_store(data, 'front_left_cam_sensor'))

		front_right_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_front_right_cam, attach_to=self.ego)
		self.actor_list.append(front_right_cam_sensor)
		front_right_cam_sensor.listen(lambda data: self.data_store(data, 'front_right_cam_sensor'))

		rear_right_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_rear_right_cam, attach_to=self.ego)
		self.actor_list.append(rear_right_cam_sensor)
		rear_right_cam_sensor.listen(lambda data: self.data_store(data, 'rear_right_cam_sensor'))

		rear_left_cam_sensor = self.world.spawn_actor(cam_blueprint, transform_rear_left_cam, attach_to=self.ego)
		self.actor_list.append(rear_left_cam_sensor)
		rear_left_cam_sensor.listen(lambda data: self.data_store(data, 'rear_left_cam_sensor'))

		left_fisheye_cam_sensor = self.world.spawn_actor(fisheye_cam_blueprint, transform_left_side_radar, attach_to=self.ego)
		self.actor_list.append(left_fisheye_cam_sensor)
		left_fisheye_cam_sensor.listen(lambda data: self.data_store(data, 'left_fisheye_cam_sensor'))

		right_fisheye_cam_sensor = self.world.spawn_actor(fisheye_cam_blueprint, transform_right_side_radar, attach_to=self.ego)
		self.actor_list.append(right_fisheye_cam_sensor)
		right_fisheye_cam_sensor.listen(lambda data: self.data_store(data, 'right_fisheye_cam_sensor'))

		front_fisheye_cam_sensor = self.world.spawn_actor(fisheye_cam_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_fisheye_cam_sensor)
		front_fisheye_cam_sensor.listen(lambda data: self.data_store(data, 'front_fisheye_cam_sensor'))

		rear_fisheye_cam_sensor = self.world.spawn_actor(fisheye_cam_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_fisheye_cam_sensor)
		rear_fisheye_cam_sensor.listen(lambda data: self.data_store(data, 'rear_fisheye_cam_sensor'))

		
		# LiDAR #
		lidar_sensor = self.world.spawn_actor(lidar_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(lidar_sensor)
		lidar_sensor.listen(lambda data: self.data_store(data, 'lidar_sensor'))

		
		# RADARS #
		front_left_corner_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_front_left_corner_radar, attach_to=self.ego)
		self.actor_list.append(front_left_corner_radar_sensor)
		front_left_corner_radar_sensor.listen(lambda data: self.data_store(data, 'front_left_corner_radar_sensor'))

		front_right_corner_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_front_right_corner_radar, attach_to=self.ego)
		self.actor_list.append(front_right_corner_radar_sensor)
		front_right_corner_radar_sensor.listen(lambda data: self.data_store(data, 'front_right_corner_radar_sensor'))

		rear_left_corner_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_rear_left_corner_radar, attach_to=self.ego)
		self.actor_list.append(rear_left_corner_radar_sensor)
		rear_left_corner_radar_sensor.listen(lambda data: self.data_store(data, 'rear_left_corner_radar_sensor'))

		rear_right_corner_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_rear_right_corner_radar, attach_to=self.ego)
		self.actor_list.append(rear_right_corner_radar_sensor)
		rear_right_corner_radar_sensor.listen(lambda data: self.data_store(data, 'rear_right_corner_radar_sensor'))

		left_side_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_left_side_radar, attach_to=self.ego)
		self.actor_list.append(left_side_radar_sensor)
		left_side_radar_sensor.listen(lambda data: self.data_store(data, 'left_side_radar_sensor'))

		right_side_radar_sensor = self.world.spawn_actor(corner_and_side_radar_blueprint, transform_right_side_radar, attach_to=self.ego)
		self.actor_list.append(right_side_radar_sensor)
		right_side_radar_sensor.listen(lambda data: self.data_store(data, 'right_side_radar_sensor'))

		front_imaging_radar_sensor = self.world.spawn_actor(front_imaging_radar_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_imaging_radar_sensor)
		front_imaging_radar_sensor.listen(lambda data: self.data_store(data, 'front_imaging_radar_sensor'))

		front_long_range_radar_sensor = self.world.spawn_actor(front_long_range_radar_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_long_range_radar_sensor)
		front_long_range_radar_sensor.listen(lambda data: self.data_store(data, 'front_long_range_radar_sensor'))

		rear_long_range_radar_sensor = self.world.spawn_actor(rear_long_range_radar_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_long_range_radar_sensor)
		rear_long_range_radar_sensor.listen(lambda data: self.data_store(data, 'rear_long_range_radar_sensor'))


		# 6DoF IMUs #
		front_imu_sensor = self.world.spawn_actor(imu_blueprint, transform_front_cam, attach_to=self.ego)
		self.actor_list.append(front_imu_sensor)
		front_imu_sensor.listen(lambda data: self.data_store(data, 'front_imu_sensor'))

		rear_imu_sensor = self.world.spawn_actor(imu_blueprint, transform_rear_cam, attach_to=self.ego)
		self.actor_list.append(rear_imu_sensor)
		rear_imu_sensor.listen(lambda data: self.data_store(data, 'rear_imu_sensor'))

		
		# GPS #
		gps_sensor = self.world.spawn_actor(gps_blueprint, attach_to=self.ego)
		self.actor_list.append(gps_sensor)
		gps_sensor.listen(lambda data: self.data_store(data, 'gps_sensor'))

		self.ego.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0))


	def data_store(self, data, sensor_name):
		pass


if __name__ == '__main__':
	client = carla.Client("127.0.0.1", 2000)
	client.set_timeout(2.0)

	test_car = Hyperion(client.get_world())
	try:
		test_car.reset()
		time.sleep(30)
	finally:
		for actor in test_car.actor_list:
			actor.destroy()