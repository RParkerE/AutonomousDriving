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
import cv2

from ultralytics import YOLO

import pickle
from numpy.linalg import inv

SHOW_PREVIEW = True

IM_WIDTH = 1280
IM_HEIGHT = 720

class YOLOv8:
	CONFIDENCE = 0.5
	font_scale = 1
	thickness = 1

	def __init__(self):
		self.model = YOLO("yolov8n.pt")
		self.labels = open("data/coco.names").read().strip().split("\n")
		self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

	def yolo_network(self, image):
		results = self.model.predict(image, conf=self.CONFIDENCE)[0]

		for data in results.boxes.data.tolist():
			xmin, ymin, xmax, ymax, confidence, class_id = data
			xmin = int(xmin)
			ymin = int(ymin)
			xmax = int(xmax)
			ymax = int(ymax)
			class_id = int(class_id)

			color = [int(c) for c in self.colors[class_id]]
			cv2.rectangle(image.astype(np.uint8), (xmin, ymin), (xmax, ymax), color=color, thickness=self.thickness)
			text = f"{self.labels[class_id]}: {confidence:.2f}"
			(text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_scale, thickness=self.thickness)[0]
			text_offset_x = xmin
			text_offset_y = ymin - 5
			box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
			overlay = image.copy()
			cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
			image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
			cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=self.font_scale, color=(0, 0, 0), thickness=self.thickness)

		return image

class LaneDetect:
	def __init__(self):
		nx = 9
		ny = 5

		self.objpoints = []
		self.imgpoints = []

		objp = np.zeros((nx*ny, 3), np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

		images = glob.glob('camera_cal/*.jpg')

		for idx, fname in enumerate(images):
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

			if ret == True:
				self.imgpoints.append(corners)
				self.objpoints.append(objp)

	def calibrate(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
		undist = cv2.undistort(img, mtx, dist, None, mtx)

		dist_pickle = {}
		dist_pickle["mtx"] = mtx
		dist_pickle["dist"] = dist
		pickle.dump(dist_pickle, open('calibration_pickle.p', 'wb'))

		return undist

	def undistort(self, img):
		dist_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
		mtx = dist_pickle["mtx"]
		dist = dist_pickle["dist"]

		undistorted = cv2.undistort(img, mtx, dist, None, mtx)

		return undistorted

	def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0,255)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		abs_sobelx = np.abs(sobelx)
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

		return sxbinary

	def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0,255)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		magnitude = np.sqrt(sobelx**2 + sobely**2)
		scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

		return sxbinary

	def dir_thresh(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		abs_sobelx = np.abs(sobelx)
		abs_sobely = np.abs(sobely)
		direction = np.arctan2(abs_sobely, abs_sobelx)
		sbinary = np.zeros_like(direction)
		sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

		return sbinary

	def combined_thresh(self, img):
		ksize = 21

		gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20,100))
		grady = self.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20,100))
		mag_binary = self.mag_thresh(img, sobel_kernel=7, mag_thresh=(50,100))
		dir_binary = self.dir_thresh(img, sobel_kernel=15, thresh=(0.4,1.3))

		combined = np.zeros_like(dir_binary)
		combined[((gradx == 1) | (grady == 1)) & ((mag_binary == 1) | (dir_binary == 1))] = 1

		return combined

	def color_thresh(self, img, s_thresh=(170,255), l_thresh=(30,255)):
		img = np.copy(img)
		
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]

		color_gradient_binary = np.zeros_like(s_channel)
		color_gradient_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & ((l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1]))] = 1

		return color_gradient_binary

	def color_gradient_thresh(self, img, s_thresh=(170,255), l_thresh=(30,255), sx_thresh=(65,100)):
		img = np.copy(img)

		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]

		sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
		abs_sobelx = np.abs(sobelx)
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

		color_gradient_binary = np.zeros_like(s_channel)
		color_gradient_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & ((l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1]))] = 1

		return color_gradient_binary

	def perspective_transform(self, img, mtx, dist, isColor=True):
		undist = cv2.undistort(img, mtx, dist, None, mtx)

		if(isColor):
			gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
		else:
			gray = undist

		xoffset = 0
		yoffset = 0
		img_size = (undist.shape[1], undist.shape[0])

		#TODO: Use computer vision using HOUGH LINES to get bounding box
		src = np.float32([(600,450), (730,450), (1150,700), (170,700)])
		dst = np.float32([[xoffset,yoffset], [img_size[0]-xoffset, yoffset],
						  [img_size[0]-xoffset, img_size[1]-yoffset],
						  [xoffset, img_size[1]-yoffset]])

		M = cv2.getPerspectiveTransform(src, dst)
		warped = cv2.warpPerspective(undist, M, img_size)

		return warped, M

	def process_img(self, img):
		dist_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
		mtx = dist_pickle["mtx"]
		dist = dist_pickle["dist"]

		img = self.undistort(img)
		img = self.color_gradient_thresh(img)
		img, M = self.perspective_transform(img, mtx, dist, isColor=False)

		return img, M

	def get_curvature_for_lanes(self, processed_img, prev_left_fitx, prev_right_fitx, prev_left_peak, prev_right_peak):
		ym_per_pix = 30/720
		xm_per_pix = 3.7/700

		yvals = []
		leftx = []
		rightx = []
		image_h = processed_img.shape[0]
		image_w = processed_img.shape[1]
		buffer_for_deciding_by_distance_from_mid = 10

		left_hist = np.sum(processed_img[int(image_h/4):,:int(image_w/2)], axis=0)
		right_hist = np.sum(processed_img[int(image_h/4):,:int(image_w/2)], axis=0)

		starting_left_peak = np.argmax(left_hist)
		leftx.append(starting_left_peak)

		starting_right_peak = np.argmax(right_hist)
		rightx.append(starting_right_peak + image_w/2)

		curH = image_h
		yvals.append(curH)
		increment = 25
		column_width = 150
		leftI = 0
		rightI = 0

		while (curH - increment >= image_h/4):
			curH = curH - increment
			leftCenter = leftx[leftI]
			leftI += 1
			rightCenter = rightx[rightI]
			rightI += 1

			leftColumnL = max((leftCenter - column_width/2), 0)
			rightColumnL = min((leftCenter + column_width/2), image_w)

			leftColumnR = max((rightCenter - column_width/2), 0)
			rightColumnR = min((rightCenter + column_width/2), image_w)

			left_hist = np.sum(processed_img[int(curH - increment):int(curH), int(leftColumnL):int(rightColumnL)], axis=0)
			right_hist = np.sum(processed_img[int(curH - increment):int(curH), int(leftColumnR):int(rightColumnR)], axis=0)

			left_peak = np.argmax(left_hist)
			right_peak = np.argmax(right_hist)

			if(left_peak):
				leftx.append(left_peak + leftColumnL)
			else:
				leftx.append(leftx[leftI-1])
			if(right_peak):
				rightx.append(right_peak + leftColumnR)
			else:
				rightx.append(rightx[rightI-1])

			yvals.append(curH)

		yvals = np.array(yvals)
		rightx = np.array(rightx)
		leftx = np.array(leftx)

		left_fit_cr = np.polyfit(yvals * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(yvals * ym_per_pix, rightx * xm_per_pix, 2)

		y_eval = np.max(yvals)
		left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
		right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])

		left_fit = np.polyfit(yvals, leftx, 2)
		left_fitx = left_fit[0] * yvals**2 + left_fit[1] * yvals + left_fit[2]
		right_fit = np.polyfit(yvals, rightx, 2)
		right_fitx = right_fit[0] * yvals**2 + right_fit[1] * yvals + right_fit[2]

		return left_curverad, right_curverad, left_fitx, right_fitx, yvals, starting_right_peak, starting_left_peak

	def draw_lane(self, warped, M, undist, left_fitx, right_fitx, yvals):
		warp_zero = np.zeros_like(warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
		pts = np.hstack((pts_left, pts_right))

		cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

		Minv = inv(M)
		new_warp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

		return cv2.addWeighted(undist, 1, new_warp, 0.3, 0)


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

		self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
		time.sleep(4)

		col_sensor = self.blueprint_library.find("sensor.other.collision")
		self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
		self.actor_list.append(self.col_sensor)
		self.col_sensor.listen(lambda event: self.collision_data(event))

		while self.front_camera is None:
			time.sleep(0.01)

		self.episode_start = time.time()

		self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))

		return self.front_camera

	def collision_data(self, event):
		self.collision_hist.append(event)

	def process_img(self, image):
		yolo = YOLOv8()
		ld = LaneDetect()
		i = np.array(image.raw_data)
		i2 = i.reshape((self.im_height, self.im_width, 4))
		i3 = i2[:,:,:3]
		ld.calibrate(i3)
		i4, M = ld.process_img(i3)
		left_curverad, right_curverad, left_fitx, right_fitx, yvals, right_peak, left_peak = ld.get_curvature_for_lanes(i4, [], [], [], [])
		i5 = ld.draw_lane(i4, M, i3, left_fitx, right_fitx, yvals)
		i6 = yolo.yolo_network(i3)
		if self.SHOW_CAM:
			cv2.imshow("Lane Detect", i5)
			cv2.imshow("Object Detect", i6)
			cv2.waitKey(1)

		self.front_camera = i3
		"""
		i = np.array(image.raw_data)
		i2 = i.reshape((self.im_height, self.im_width, 4))
		i3 = i2[:,:,:3]
		if self.SHOW_CAM:
			cv2.imshow("", i3)
			cv2.waitKey(1)

		self.front_camera = i3
		"""

if __name__ == '__main__':
	env = CarEnv()
	env.reset()
	time.sleep(30)

	for actor in env.actor_list:
		actor.destroy()
