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
from utils.ExtendedKalmanFilter import EKF

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class TestVehicle:
	def __init__(self, world):
		self.world = world
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]

		self.sensor_data = {
			"camera": {},
			"lidar": {},
			"radar": {},
			"gnss": {},
			"imu": {},
		}

		self.est_traj = []

		self.LKA = LKA()
		self.OA = ObjectDetector()
		self.EKF = EKF()

	def reset(self):
		self.actor_list = []

		self.transform = random.choice(self.world.get_map().get_spawn_points())
		self.ego_vehicle = self.world.spawn_actor(self.model_3, self.transform)
		self.actor_list.append(self.ego_vehicle)
		self.ego_vehicle.set_autopilot(True)

		transform_front_cam = carla.Transform(carla.Location(x=0.8, z=2.6))
		transform_front_lidar = carla.Transform(carla.Location(x=0.8, z=2.6))
		transform_front_radar = carla.Transform(carla.Location(x=2.0, z=1.0))
		transform_origin = carla.Transform(carla.Location(x=0.0, z=0.0))

		self.front_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
		self.front_cam_bp.set_attribute('image_size_x', f'640')
		self.front_cam_bp.set_attribute('image_size_y', f'480')
		self.front_cam_bp.set_attribute('fov', '110')

		self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
		self.lidar_bp.set_attribute('rotation_frequency','10')
		#self.lidar_bp.set_attribute('noise_stddev', str(5e-5))

		self.radar_bp = self.blueprint_library.find('sensor.other.radar')
		self.radar_bp.set_attribute('horizontal_fov', str(35))
		self.radar_bp.set_attribute('vertical_fov', str(20))
		self.radar_bp.set_attribute('range', str(20))

		self.imu_bp = self.blueprint_library.find("sensor.other.imu")
		#self.imu_bp.set_attribute('noise_accel_stddev_x', str(5e-5))
		#self.imu_bp.set_attribute('noise_accel_stddev_y', str(5e-5))
		#self.imu_bp.set_attribute('noise_accel_stddev_z', str(5e-5))
		#self.imu_bp.set_attribute('noise_gyro_stddev_x', str(5e-5))
		#self.imu_bp.set_attribute('noise_gyro_stddev_y', str(5e-5))
		#self.imu_bp.set_attribute('noise_gyro_stddev_z', str(5e-5))
		#self.imu_bp.set_attribute('noise_gyro_bias_x', str(1e-5))
		#self.imu_bp.set_attribute('noise_gyro_bias_y', str(1e-5))
		#self.imu_bp.set_attribute('noise_gyro_bias_z', str(1e-5))
		self.imu_bp.set_attribute('sensor_tick', str(1.0 / 200))

		self.gnss_bp = self.blueprint_library.find("sensor.other.gnss")
		#self.gnss_bp.set_attribute('noise_alt_bias', str(1e-5))
		#self.gnss_bp.set_attribute('noise_alt_stddev', str(5e-5))
		#self.gnss_bp.set_attribute('noise_lat_bias', str(1e-5))
		#self.gnss_bp.set_attribute('noise_lat_stddev', str(5e-5))
		#self.gnss_bp.set_attribute('noise_lon_bias', str(1e-5))
		#self.gnss_bp.set_attribute('noise_lon_stddev', str(5e-5))
		self.gnss_bp.set_attribute('sensor_tick', str(1.0 / 5))

		self.front_cam_sensor = self.world.spawn_actor(self.front_cam_bp, transform_front_cam, attach_to=self.ego_vehicle)
		self.actor_list.append(self.front_cam_sensor)
		
		self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, transform_front_lidar, attach_to=self.ego_vehicle)
		self.actor_list.append(self.lidar_sensor)
		
		self.radar_sensor = self.world.spawn_actor(self.radar_bp, transform_front_cam, attach_to=self.ego_vehicle)
		self.actor_list.append(self.radar_sensor)

		self.imu_sensor = self.world.spawn_actor(self.imu_bp, transform_origin, attach_to=self.ego_vehicle)
		self.actor_list.append(self.imu_sensor)

		self.gnss_sensor = self.world.spawn_actor(self.gnss_bp, transform_origin, attach_to=self.ego_vehicle)
		self.actor_list.append(self.gnss_sensor)

		self.front_cam_sensor.listen(lambda data: self.runner(data,"camera"))
		self.lidar_sensor.listen(lambda data: self.runner(data,"lidar"))
		self.radar_sensor.listen(lambda data: self.runner(data,"radar"))
		self.gnss_sensor.listen(lambda data: self.runner(data,"gnss"))
		self.imu_sensor.listen(lambda data: self.runner(data,"imu"))

	def runner(self, data, sensor):
		if sensor == "camera":
			frame = np.array(data.raw_data)
			frame = frame.reshape((480, 640, 4))
			frame = frame[:,:,:3]

			self.sensor_data[sensor][data.frame] = frame

			laneImg = self.LKA.lane_detect(frame)
			results = self.OA.detect(frame)
		elif sensor == "lidar":
			self.sensor_data[sensor][data.frame] = data

			#if self.EKF.is_initialized():
			#	self.EKF.correct_state_with_lidar(data)
		elif sensor == "radar":
			self.sensor_data[sensor][data.frame] = data
		elif sensor == "gnss":
			alt = data.altitude
			lat = data.latitude
			lng = data.longitude 

			# Convert to xyz coordinates
			x = lng*1e7
			y = lat*-1e7
			z = alt

			xyz_data = {"x": x, "y": y, "z": z}
			self.sensor_data[sensor][data.frame] = xyz_data

			if not self.EKF.is_initialized():
				self.EKF.initialize_with_gnss(xyz_data)
				self.est_traj.append(self.EKF.get_location())
			else:
				self.EKF.correct_state_with_gnss(xyz_data)
				self.est_traj.append(self.EKF.get_location())
		elif sensor == "imu":
			self.sensor_data[sensor][data.frame] = data

			if self.EKF.is_initialized():
				self.EKF.predict_state_with_imu(data)
				self.est_traj.append(self.EKF.get_location())
		

if __name__ == '__main__':
	client = carla.Client("127.0.0.1", 2000)
	client.set_timeout(2.0)

	test_car = TestVehicle(client.get_world())
	try:
		test_car.reset()
		client.start_recorder("E:\\CARLA\\CARLA_0.9.14\\WindowsNoEditor\\PythonAPI\\custom\\Hyperion\\SimData\\recording01.log", True)
		time.sleep(540)
		with open(".\\estimated.dat", "w+") as f:
			for traj in test_car.est_traj:
				f.write(f"{traj}\n")
			f.close()
		client.stop_recorder()
	finally:
		for actor in test_car.actor_list:
			actor.destroy()