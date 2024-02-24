import os
import cv2
import pickle
import numpy as np


class LKA:
	def __init__(self):
		self.cameraCalibration = pickle.load( open(f'{os.getcwd()}\\pickled_data\\camera_calibration.p', 'rb' ) )
		self.mtx, self.dist = map(self.cameraCalibration.get, ('mtx', 'dist'))
		self.transMatrix = pickle.load( open(f'{os.getcwd()}\\pickled_data\\perspective_transform.p', 'rb' ) )
		self.M, self.Minv = map(self.transMatrix.get, ('M', 'Minv'))

	def lane_detect(self, frame):
		left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = self.findLines(frame)
		output = self.drawLine(frame, left_fit, right_fit)
		# cv2.imwrite(f".\\tmp\\{time.localtime()}.jpg", output)
		return output

	def undistortAndHLS(self, image):
		"""
		Undistort the image with `mtx`, `dist` and convert it to HLS.
		"""
		undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
		return cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)

	def threshIt(self, img, thresh_min, thresh_max):
		"""
		Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
		"""
		xbinary = np.zeros_like(img)
		xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
		return xbinary
		
	def absSobelThresh(self, img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
		"""
		Calculate the Sobel gradient on the direction `orient` and return a binary thresholded image 
		on [`thresh_min`, `thresh_max`]. Using `sobel_kernel` as Sobel kernel size.
		"""
		if orient == 'x':
			yorder = 0
			xorder = 1
		else:
			yorder = 1
			xorder = 0
			
		sobel = cv2.Sobel(img, cv2.CV_64F, xorder, yorder, ksize=sobel_kernel)
		abs_sobel = np.absolute(sobel)
		scaled = np.uint8(255.0*abs_sobel/np.max(abs_sobel))
		return self.threshIt(scaled, thresh_min, thresh_max)

	def combineGradients(self, img):
		"""
		Compute the combination of Sobel X and Sobel Y or Magnitude and Direction
		"""
		useSChannel = lambda img: self.undistortAndHLS(img)[:,:,2]
		withSobelX = lambda img: self.absSobelThresh(useSChannel(img), thresh_min=10, thresh_max=160)
		withSobelY = lambda img: self.absSobelThresh(useSChannel(img), orient='y', thresh_min=10, thresh_max=160)

		sobelX = withSobelX(img)
		sobelY = withSobelY(img)
		combined = np.zeros_like(sobelX) 
		combined[((sobelX == 1) & (sobelY == 1))] = 1
		return combined

	def adjustPerspective(self, image):
		"""
		Adjust the `image` using the transformation matrix `M`.
		"""
		img_size = (image.shape[1], image.shape[0])
		warped = cv2.warpPerspective(image, self.M, img_size)
		return warped

	def findLines(self, image, nwindows=9, margin=110, minpix=50):
		"""
		Find the polynomial representation of the lines in the `image` using:
		- `nwindows` as the number of windows.
		- `margin` as the windows margin.
		- `minpix` as minimum number of pixes found to recenter the window.
		- `ym_per_pix` meters per pixel on Y.
		- `xm_per_pix` meters per pixels on X.
		
		Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
		""" 
		combineAndTransform = lambda img: self.adjustPerspective(self.combineGradients(img))

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		# Make a binary and transform image
		binary_warped = combineAndTransform(image)
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Set height of windows
		window_height = np.int(binary_warped.shape[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []
		
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		
		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		
		# Fit a second order polynomial to each
		left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
		
		return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)

	def calculateCurvature(self, yRange, left_fit_cr):
	    """
	    Returns the curvature of the polynomial `fit` on the y range `yRange`.
	    """
	    
	    return ((1 + (2*left_fit_cr[0]*yRange*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])


	def drawLine(self, img, left_fit, right_fit):
	    """
	    Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
	    """
	    yMax = img.shape[0]
	    ploty = np.linspace(0, yMax - 1, yMax)
	    color_warp = np.zeros_like(img).astype(np.uint8)
	    
	    # Calculate points.
	    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	    
	    # Recast the x and y points into usable format for cv2.fillPoly()
	    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	    pts = np.hstack((pts_left, pts_right))
	    
	    # Draw the lane onto the warped blank image
	    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	    
	    # Warp the blank back to original image space using inverse perspective matrix (Minv)
	    newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0])) 
	    return cv2.addWeighted(img, 1, newwarp, 0.3, 0) 