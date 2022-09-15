import matplotlib.image as mpimg # uses Pillow
import matplotlib.pyplot as plt
import numpy as np
import pyproj.geod as gd
import cv2

traversals = ['mercury_south_hemisphere_blue.png',
			'mercury_south_hemisphere_green.png',
			'mercury_south_hemisphere_orange.png',
			'mercury_south_hemisphere_purple.png',
			'mercury_south_hemisphere_red.png',
			'mercury_south_hemisphere_turqoise.png',
			'mercury_south_hemisphere_yellow.png']

R = 2440.5 # equatorial radius
R_pix = 327 # equatorial radius in pixels

img = mpimg.imread('mercury_south_hemisphere.jpg') # import image
img_array = np.array(img)

for traversal in traversals:
	img_traversal = mpimg.imread(traversal)

	img_array_traversal = np.array(img_traversal)
	#imgplot = plt.imshow(img_array, extent=[-img_array.shape[1]/2., img_array.shape[1]/2., -img_array.shape[0]/2., img_array.shape[0]/2. ]) # plot the default image
	#plt.show()
	#plt.close()

	## ISOLATE PATHS AND CALCULATE LAT LONG ##

	img_array_traversal[img_array_traversal[:,:, 0] != 1] = [1,1,1,0]
	img_array_traversal[img_array_traversal[:,:, 1] != 0] = [1,1,1,0]
	img_array_traversal[img_array_traversal[:,:, 2] != 0] = [1,1,1,0]

	line_traversal = np.where(img_array_traversal[:,:, 3] != 0) # get traversal path points
	line_traversal = np.array([line_traversal[1], line_traversal[0]]) # fuse x and y coordinates into one array
	line_traversal[0] = line_traversal[0] - img_array.shape[1]/2. # adjust x values to center at pole
	line_traversal[1] = line_traversal[1] - img_array.shape[0]/2. # adjust y values to center at pole

	# lat = arccos(dr/R) where dr is radial distance from pole and R is total radius, long = arctan(x/y)
	lat_long = np.array([np.arccos(np.sqrt(np.square(line_traversal[0])+np.square(line_traversal[1])) / R_pix), np.arctan2(line_traversal[0],line_traversal[1]) + np.pi])
	lat_long_deg = np.rad2deg(lat_long) # convert lat and long to degrees

	lat_max = np.max(lat_long_deg[0])
	lat_min = np.min(lat_long_deg[0])
	long_max = np.max(lat_long_deg[1])
	long_min = np.min(lat_long_deg[1])

	#x = np.arange(start = -img_array.shape[1]/2., stop = img_array.shape[1]/2.+1, step=1) # get all x coordinates
	#y = np.arange(start = -img_array.shape[0]/2., stop = img_array.shape[0]/2.+1, step=1) # get all y coordinates

	#print(lat_long_deg)

	imgplot_traversal = plt.imshow(img_array_traversal, extent=[-img_array.shape[1]/2., img_array.shape[1]/2., -img_array.shape[0]/2., img_array.shape[0]/2. ])
	plt.show()
	plt.close()


	## DISTANCE CALCULATIONS ##

	# Haversine formula:
	# a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
	# c = 2 ⋅ atan2( √a, √(1−a) )
	# d = R ⋅ c

	# Law of Cosines: d = acos( sin φ1 ⋅ sin φ2 + cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R

	# where φ is latitude, λ is longitude, and R is Mercury's radius

	total_d = 0
	for i in range(0, len(lat_long)-1):
		d_phi = lat_long[0,i+1] - lat_long[0,i]
		d_lambda = lat_long[1,i+1] - lat_long[1,i]
		a = np.sin(d_phi/2)**2 + np.cos(lat_long[0,i]) * np.cos(lat_long[0,i+1]) * np.sin(d_lambda/2)**2
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
		d = R * c
		#d = np.arccos(np.sin(lat_long[0,i])*np.sin(lat_long[0,i+1]) + np.cos(lat_long[0,i])*np.cos(lat_long[0,i+1])*np.cos(d_lambda)) * R # NOTE: cosine method gives same results as Haversine
		total_d = total_d + d

	print('----------------------------------------------')
	print('Traversal: ' + traversal)
	print('Max Lat: ' + str(lat_max) + ' deg')
	print('Min Lat: ' + str(lat_min) + ' deg')
	print('Max Long: ' + str(long_max) + ' deg')
	print('Min Long: ' + str(long_min) + ' deg')
	print('distance: ' + str(total_d) + ' km')


	'''
	Validation tests (comparing longitude angles to observation)

	x = np.arange(0, np.shape(lat_long_deg)[1], 1)
	print(np.shape(lat_long_deg[1]))
	print(np.shape(x))
	plt.scatter(x, lat_long_deg[1])
	plt.show()
	plt.close()
	'''


# geod = gd.Geod('+a=2440.5 +b=2438.3 +f=0.0009')
