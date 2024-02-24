import numpy as np

from .rotations import Quaternion, omega, skew_symmetric, angle_normalize

class EKF:
	def __init__(self):
		self.p = np.zeros([3, 1])
		self.v = np.zeros([3, 1])
		self.q = np.zeros([4, 1])

		self.p_cov = np.zeros([9, 9])

		self.last_ts = 0

		self.g = np.array([0, 0, -9.81]).reshape(3, 1)

		self.var_imu_acc = 0.01
		self.var_imu_gyro = 0.01

		self.var_gnss = np.eye(3) * 100

		self.var_lidar = 0.25

		self.l_jac = np.zeros([9, 6])
		self.l_jac[3:, :] = np.eye(6)

		self.h_jac = np.zeros([3, 9])
		self.h_jac[:, :3] = np.eye(3)

		self.n_gnss_taken = 0
		self.gnss_init_xyz = None
		self.initialized = False

	def is_initialized(self):
		return self.initialized

	def initialize_with_gnss(self, gnss, num_samples=1):
		if self.gnss_init_xyz is None:
			self.gnss_init_xyz = np.array([gnss["x"], gnss["y"], gnss["z"]])
		else:
			self.gnss_init_xyz[0] += gnss["x"]
			self.gnss_init_xyz[1] += gnss["y"]
			self.gnss_init_xyz[2] += gnss["z"]
		self.n_gnss_taken += 1

		if self.n_gnss_taken == num_samples:
			self.gnss_init_xyz /= num_samples
			self.p[:, 0] = self.gnss_init_xyz
			self.q[:, 0] = Quaternion().to_numpy()

			pos_var = 1
			orien_var = 1000
			vel_var = 1000
			self.p_cov[:3, :3] = np.eye(3) * pos_var
			self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
			self.p_cov[6:, 6:] = np.eye(3) * orien_var
			self.initialized = True

	def get_location(self):
		return self.p.reshape(-1).tolist()

	def predict_state_with_imu(self, imu):
		imu_f = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]).reshape(3, 1)
		imu_w = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]).reshape(3, 1)

		delta_t = imu.timestamp - self.last_ts
		self.last_ts = imu.timestamp

		# Linearize Motion Model
		R = Quaternion(*self.q).to_mat()
		self.p = self.p + delta_t * self.v + delta_t ** 2 / 2 * (R @ imu_f + self.g)
		self.v = self.v + delta_t * (R @ imu_f - self.g)
		self.q = omega(imu_w, delta_t) @ self.q

		# Propagate Uncertainty
		F = np.identity(9)
		F[:3, 3:6] = np.identity(3) * delta_t
		F[3:6, -3:] = -skew_symmetric(R @ imu_f) * delta_t

		Q = np.identity(6)
		Q[:3, :3] *= delta_t**2 * self.var_imu_acc
		Q[-3:, -3:] *= delta_t**2 * self.var_imu_gyro

		self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T

	def correct_state_with_gnss(self, gnss):
		x = gnss["x"]
		y = gnss["y"]
		z = gnss["z"]

		# Kalman Gain
		K = self.p_cov @ self.h_jac.T @ (np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + self.var_gnss))

		delta_x = K @ (np.array([x, y, z])[:, None] - self.p)

		# Correction
		self.p = self.p + delta_x[:3]
		self.v = self.v + delta_x[3:6]
		delta_q = Quaternion(axis_angle=angle_normalize(delta_x[6:]))
		self.q = delta_q.quat_mult_left(self.q)

		self.p_cov = (np.identity(9) - K @ self.h_jac) @ self.p_cov
