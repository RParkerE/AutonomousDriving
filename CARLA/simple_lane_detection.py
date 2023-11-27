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

import math
import random
import queue
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import keras.backend as backend
from threading import Thread
from tqdm import tqdm
from keras.callbacks import TensorBoard

SHOW_PREVIEW = True
SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.5
MIN_REWARD = -200
DISCOUNT = 0.99
EPISODES = 100
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10

IM_WIDTH = 640
IM_HEIGHT = 480


class ModifiedTensorBoard(TensorBoard):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.create_file_writer(self.log_dir)
		self._log_write_dir = self.log_dir

	def set_model(self, model):
		self.model = model

		self._train_dir = os.path.join(self._log_write_dir, 'train')
		self._train_step = self.model._train_counter

		self._val_dir = os.path.join(self._log_write_dir, 'validation')
		self._val_step = self.model._test_counter

		self._should_write_train_graph = False

	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	def on_batch_end(self, batch, logs=None):
		pass

	def on_train_end(self, _):
		pass

	def update_stats(self, **stats):
		with self.writer.as_default():
			for key, value in stats.items():
				tf.summary.scalar(key, value, step = self.step)
				self.writer.flush()

class CarEnv:
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = IM_WIDTH
	im_height = IM_HEIGHT
	front_camera = None

	def __init__(self):
		self.client = carla.Client("127.0.0.1", 2000)
		self.client.set_timeout(2.0)
		self.world = self.client.get_world()
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]

	def reset(self):
		self.collision_hist = []
		self.actor_list = []

		self.transform = random.choice(self.world.get_map().get_spawn_points())
		self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
		self.actor_list.append(self.vehicle)

		self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
		self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.rgb_cam.set_attribute("fov", "110")

		transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(4)

		col_sensor = self.blueprint_library.find("sensor.other.collision")
		self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
		self.actor_list.append(self.col_sensor)
		self.col_sensor.listen(lambda event: self.collision_data(event))

		while self.front_camera is None:
			time.sleep(0.01)

		self.episode_start = time.time()

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

		return self.front_camera

	def collision_data(self, event):
		self.collision_hist.append(event)

	def process_img(self, image):
		i = np.array(image.raw_data)
		i2 = i.reshape((self.im_height, self.im_width, 4))
		i3 = i2[:,:,:3]
		if self.SHOW_CAM:
			cv2.imshow("", i3)
			cv2.waitKey(1)

		self.front_camera = i3

	def step(self, action):
		if action == 0:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
		elif action == 1:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
		elif action == 2:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

		v = self.vehicle.get_velocity()
		kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

		if len(self.collision_hist) != 0:
			done = True
			reward = -200
		elif kmh < 50:
			done = False
			reward = -1
		else:
			done = False
			reward = 1

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True

		return self.front_camera, reward, done, None

class VehiclePIDController():
	def __init__(self, vehicle, args_lateral, args_longitudnal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
		self.max_brake = max_brake
		self.max_steering = max_steering
		self.max_throttle = max_throttle

		self.vehicle = vehicle
		self.world = vehicle.get_world()

		self.past_steering = self.vehicle.get_control().steer

		self.lon_ctrlr = PIDLongitudnalControl(self.vehicle, **args_longitudnal)
		self.lat_ctrlr = PIDLateralControl(self.vehicle, **args_lateral)

	def run_step(self, tgt_speed, waypoint):
		acceleration = self.lon_ctrlr.run_step(tgt_speed)
		current_steering = self.lat_ctrlr.run_step(waypoint)
		control = carla.VehicleControl()

		if acceleration >= 0.0:
			control.throttle = min(abs(acceleration), self.max_throttle)
			control.brake = 0.0
		else:
			control.throttle = 0.0
			control.brake = min(abs(acceleration), self.max_brake)

		if current_steering > self.past_steering + 0.1:
			current_steering = self.past_steering + 0.1
		elif current_steering < self.past_steering - 0.1:
			current_steering = self.past_steering - 0.1
		if current_steering >= 0:
			steering = min(self.max_steering, current_steering)
		else:
			steering = max(-self.max_steering, current_steering)

		control.steer = steering
		control.hand_brake = False
		control.manual_gear_shift = False
		self.past_steering = steering

		return control

class PIDLongitudnalControl():
	def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
		self.vehicle = vehicle
		self.K_D = K_D
		self.K_P = K_P
		self.K_I = K_I
		self.dt = dt
		self.errorBuffer = queue.deque(maxlen=10)

	def run_step(self, tgt_speed):
		current_speed = 3.6 * math.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2 + self.vehicle.get_velocity().z**2)
		return self.pid_controller(tgt_speed, current_speed)

	def pid_controller(self, tgt_speed, current_speed):
		error = tgt_speed - current_speed
		self.errorBuffer.append(error)

		if len(self.errorBuffer) >= 2:
			de = (self.errorBuffer[-1] - self.errorBuffer[-2])/self.dt
			ie = sum(self.errorBuffer)*self.dt
		else:
			de = 0.0
			ie = 0.0

		return np.clip(self.K_P * error + self.K_D * de + self.K_I * ie, -1.0, 1.0)

class PIDLateralControl():
	def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
		self.vehicle = vehicle
		self.K_D = K_D
		self.K_P = K_P
		self.K_I = K_I
		self.dt = dt
		self.errorBuffer = queue.deque(maxlen=10)

	def run_step(self, waypoint):
		return self.pid_controller(waypoint, self.vehicle.get_transform())

	def pid_controller(self, waypoint, vehicle_transform):
		v_begin = vehicle_transform.location
		v_end = v_begin + carla.Location(x = math.cos(math.radians(vehicle_transform.rotation.yaw)), y = math.sin(math.radians(vehicle_transform.rotation.yaw)))
		v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

		w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])

		dot = math.acos(np.clip(np.dot(w_vec, v_vec)/np.linalg.norm(w_vec)*np.linalg.norm(v_vec), -1.0, 1.0))
		cross = np.cross(v_vec, w_vec)

		if cross[2] < 0:
			dot *= -1

		self.errorBuffer.append(dot)

		if len(self.errorBuffer) >= 2:
			de = (self.errorBuffer[-1] - self.errorBuffer[-2])/self.dt
			ie = sum(self.errorBuffer)*self.dt
		else:
			de = 0.0
			ie = 0.0

		return np.clip((self.K_P * dot) + (self.K_I * ie) + (self.K_D * de), -1.0, 1.0)


if __name__ == '__main__':
	env = CarEnv()
	env.reset()
	try:
		control_vehicle = VehiclePIDController(env.vehicle, args_lateral={'K_P':1, 'K_D':0.0, 'K_I':0.0}, args_longitudnal={'K_P':1, 'K_D':0.0, 'K_I':0.0})
		while True:
			waypoints = env.world.get_map().get_waypoint(env.vehicle.get_location())
			waypoint = np.random.choice(waypoints.next(0.3))
			control_signal = control_vehicle.run_step(5, waypoint)
			env.vehicle.apply_control(control_signal)

			"""camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
									camera_bp.set_attribute('image_size_x', '800')
									camera_bp.set_attribute('image_size_y', '600')
									camera_bp.set_attribute('fov', '90')
									camera_transform = carla.Transform(carla.Location(x=1.5, y=2.4))
									camera = env.world.spawn_actor(camera_bp, camera_transform)
									camera.listen(lambda image: image.save_to_disk(f'output/{image.frame}', carla.ColorConverter.CityScapesPalette))
							
									depth_camera_bp = blueprint_library.find('sensor.camera.depth')
									depth_camera_bp.set_attribute('image_size_x', '800')
									depth_camera_bp.set_attribute('image_size_y', '600')
									depth_camera_bp.set_attribute('fov', '90')
									depth_camera_transform = carla.Transform(carla.Location(x=1.5, y=2.4))
									depth_camera = env.world.spawn_actor(depth_camera_bp, depth_camera_transform)
									depth_camera.listen(lambda image: image.save_to_disk(f'output/{image.frame}', carla.ColorConverter.LogarithmicDepth))

		time.sleep(15)"""

	finally:
		for actor in env.actor_list:
			actor.destroy()
