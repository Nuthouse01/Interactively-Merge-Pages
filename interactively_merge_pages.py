"""
Requires Python 3.7 or higher.

Windows command-line commands needed to install proper library versions:
pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade pillow
pip install --upgrade scikit-image
pip install --upgrade opencv-python

Articles that describe how to do the nerd stuff:
https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
"""

try:
	import copy
	import os
	import random
	
	import cv2
	import numpy as np
	from PIL import Image
	from skimage.metrics import structural_similarity
except ImportError:
	print("ERROR: necessary libraries are missing!!")
	print("The commands to install them are listed near the top of this file.")
	input("")
	quit(0)

# the height in pixels(???) of any images that open in windows.
# if it is too big, you cannot see the bottom part of the image :(
DISPLAY_HEIGHT = 780


# if this is TRUE then you will see the pages in a random order (recommended).
# this should help prevent you from getting... distracted.
# if this is FALSE then you will see the pages in numerical order.
RANDOMIZE_PAGE_ORDER = True

DEBUG_ALIGNMENT = False
DEBUG_DIFF = False
DEBUG_INTERACTIVE = False

RESIZE_BEFORE_ALIGN = True
ALIGN_USE_QUADRANT_METHOD = True
ALIGN_MAX_FEATURES = 3000
ALIGN_KEEP_PERCENT = 0.30
DEBUG_ALIGNMENT_ANIMATION_TIME = 500
SAVE_ALIGNED_IMAGE = False
ALIGNED_IMAGE_NAME = "aligned.jpg"

BLUR_BEFORE_DIFF = True
BLUR_SIZE = 7           ###
COLORCLIP_BEFORE_DIFF = True
COLORCLIP_AMOUNT = 10
MIN_CONTOUR_AREA = 300  ###
OPENING_SIZE = 5        ### more "opening" to eliminate slivers & bridges
CLOSING_SIZE = 11       ### more "closing" to bridge gaps
DEBUG_DIFF_ANIMATION_TIME = 1000

PREVIEW_BLINK_RATE_MS = 400
PREVIEW_BLINK_DUTY_CYCLE = 0.5
BASEIMG_COLOR = (255, 0, 0)
NEWIMG_COLOR = (0, 0, 255)
BORDER_THICKNESS = 2  # this is a purely visual thing, does not effect the image saved to disk at all
INTERACTIVE_GROWSHRINK_SIZE = 5  ###


KEYS_RESET =    [ord(k) for k in ('r','R')]
KEYS_GROW =     [ord(k) for k in ('=','+')]
KEYS_SHRINK =   [ord(k) for k in ('-','_')]
KEYS_FILL =     [ord(k) for k in ('f','F')]
KEYS_MARKDIFF = [ord(k) for k in ('d','D')]
KEYS_NOTDIFF =  [ord(k) for k in ('n','N')]
KEYS_ALLBASE =  [ord(k) for k in ('b','B')]
KEYS_ALLSEC =   [ord(k) for k in ('s','S')]

# problem: images are different sizes & slightly rotated
# solution: "image feature detection and registration"
# problem: image alingment isn't perfect, test image selecting too many points from tophalf & few from bottom
# solution: JUST GRAB MORE POINTS WHO GIVES A SHIT ABOUT PROCESSING TIME WOOOOOOO
# problem: dot-array shading ruins structural diff
# solution: blur with 7-pixel kernel to smear things around and make it look like a gradient
# problem: alignment still isn't perfect, some borders dont match, make long slivers show up in diff
# solution: morphological "opening", also discard diff regions with small total area
# problem: "opening" is making me lose some borders around textboxes
# solution: less opening
# problem: borders around textboxes are not quite complete, must be complete to be filled in!
# solution: morphological "closing"
# problem: "closing" is making textbox regions link with true difference regions
# solution: less closing
# problem: alignment isn't good enough, significantly off for some some corners of some pages
# solution: FORCE the algorithm to select match points in all 4 quadrants of the image, get better alignment this way
# problem: blacks are noisy in one ver but clean in the other ver, makes the stuctural diff go wild & flag stuff it shouldn't
# solution: clip all color values from [0-255] -> [5-250] and squash the tiny noise

# status: each test image down to ~25 diff regions, not missing any textboxes :-)

####################################################################################################
####################################################################################################
####################################################################################################

def my_resize(input_img: np.ndarray, newheight=None, newwidth=None) -> np.ndarray:
	if newheight is None and newwidth is None:
		return input_img
	im = Image.fromarray(input_img)
	if isinstance(newheight, int) and newwidth is None:
		# new HEIGHT is specified, derive new WIDTH
		newdim = (int(im.width * newheight / im.height), newheight)
	elif isinstance(newwidth, int) and newheight is None:
		# new WIDTH is specified, derive new HEIGHT
		newdim = (newwidth, int(im.height * newwidth / im.width))
	else:
		# both are specified!
		newdim = (newwidth, newheight)
	im_resized = im.resize(newdim, resample=Image.BICUBIC)
	matchedVis2 = np.array(im_resized)
	return matchedVis2

####################################################################################################
####################################################################################################
####################################################################################################

def find_matchpoints_basic(maxFeatures, keepPercent, imageGray, templateGray):
	# use ORB to detect keypoints and extract (binary) local invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descripA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descripB) = orb.detectAndCompute(templateGray, None)
	
	# create BFMatcher object
	# i dont understand the difference between these two lines, but i think BFMatcher performs better?
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	# match the features
	matches = matcher.match(descripA, descripB, None)

	# sort the matches by their distance (the smaller the distance, the "more similar" the features are)
	matches = sorted(matches, key=lambda x: x.distance)
	
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	
	return kpsA, kpsB, matches

def find_matchpoints_quadrants(maxFeatures, keepPercent, imageGray, templateGray):
	
	# NEW IDEA:
	#  force the algorithm to find features in each quadrant of the image!
	#  like sticking thumbtacks in each corner, no one spot is perfectly aligned but no spot is awful either
	#  do the same exact process as whole-image, but repeat 4x
	
	kpsA_out = []  # list of keypoints for "image"
	kpsB_out = []  # list of keypoints for "template"
	matches_out = []
	matchct = []
	halfHA = int(imageGray.shape[0] / 2)
	halfWA = int(imageGray.shape[1] / 2)
	halfHB = int(templateGray.shape[0] / 2)
	halfWB = int(templateGray.shape[1] / 2)
	for hA,hB in ((0,0), (halfHA,halfHB)):
		for wA,wB in ((0,0), (halfWA,halfWB)):
			# isolate one quadrant of the image (each image)
			quadrantA =    imageGray[hA:hA + halfHA, wA:wA + halfWA]
			quadrantB = templateGray[hB:hB + halfHB, wB:wB + halfWB]
			
			kpsA, kpsB, matches = find_matchpoints_basic(int(maxFeatures / 4), keepPercent, quadrantA, quadrantB)
			matchct.append(len(matches))
			# offset the keypoints' coordinates by h and w
			# they don't understand that they are located in one quadrant of a larger image so i need to modify them
			for k in kpsA: k.pt = (k.pt[0] + wA, k.pt[1] + hA)
			for k in kpsB: k.pt = (k.pt[0] + wB, k.pt[1] + hB)
			
			# each match object refers to the index of pointA and index of pointB that it links...
			# need to modify those indices so they still point at the right places after adding them to a bigger list
			for m in matches:
				m.queryIdx += len(kpsA_out)
				m.trainIdx += len(kpsB_out)
			# concat them onto the growing lists
			kpsA_out += kpsA
			kpsB_out += kpsB
			matches_out += matches
			
	return kpsA_out, kpsB_out, matches_out, matchct

def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False) -> np.ndarray:
	"""Step 1 of the 3-step process. Ensure the two images are perfectly aligned."""
	if image.shape != template.shape:
		print("size difference: baseimg = {}  newimg = {}".format(str(template.shape[0:2]), str(image.shape[0:2])))
		# i know the alignment procedure takes care of this already,
		# but i seem to get better results if i manually make their sizes match beforehand.
		# let the "homography matrix" be as sparse as it possibly can.
		image = my_resize(image, newheight=template.shape[0], newwidth=template.shape[1])
	else:
		print("size match")
		
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	
	print("matching...", end="")
	if ALIGN_USE_QUADRANT_METHOD:
		kpsA, kpsB, matches, matchct = find_matchpoints_quadrants(maxFeatures, keepPercent, imageGray, templateGray)
		if debug: print("found %d matches, %s" % (sum(matchct), str(matchct)))
	else:
		kpsA, kpsB, matches = find_matchpoints_basic(maxFeatures, keepPercent, imageGray, templateGray)
		if debug: print("found %d matches" % len(matches))

	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt

	print("aligning...")
	# compute the homography matrix between the two sets of matched points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	# note: align the ORIGINAL image, not the greyscale we created
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	
	# check to see if we should visualize the matched keypoints
	if debug:
		# print the homography matrix, to sorta indicate how much it needed to be distorted by
		# i dont understand the meanings of the individual entries tho...
		# also, get rid of this dumb scientific notation crap
		print("homography matrix:")
		for row in H: print("[{: 13.6f} {: 13.6f} {: 13.6f}]".format(row[0],row[1],row[2]))
		# this will display the 2 images side-by-side, with lines between matching points
		# matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
		# cv2.imwrite("1_matches.jpg", matchedVis)
		matchedVis2 = my_resize(matchedVis, newheight=DISPLAY_HEIGHT)
		cv2.imshow("Matched Keypoints", matchedVis2)
		template2 = my_resize(template, newheight=DISPLAY_HEIGHT)
		aligned2 = my_resize(aligned, newheight=DISPLAY_HEIGHT)
		print("debug: press any key to continue")
		frame = 0
		while True:
			if frame == 0:
				cv2.imshow("Aligned", template2)
			elif frame == 1:
				cv2.imshow("Aligned", aligned2)
			frame = (frame + 1) % 2
			v = cv2.waitKey(DEBUG_ALIGNMENT_ANIMATION_TIME)
			if v != -1:
				break

		cv2.destroyWindow("Matched Keypoints")
		cv2.destroyWindow("Aligned")
	# return the aligned image
	return aligned


def find_image_differences(image, template, debug=False) -> list:
	"""Step 2 of the 3-step process. Find all regions of (significant) difference betweent the images."""
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	
	if BLUR_BEFORE_DIFF:
		print("blurring...", end="")
		# note, blur operation is not commutative! blur3 then blur3 != blur5. which is better? idk.
		img_gray = cv2.blur(img_gray,   (BLUR_SIZE,BLUR_SIZE))
		temp_gray = cv2.blur(temp_gray, (BLUR_SIZE,BLUR_SIZE))
	
	if COLORCLIP_BEFORE_DIFF:
		print("clipping...", end="")
		# most noise seems to exist in solid-black or solid-white areas
		# i can remove alot by clipping the [0-255] range to [5-250] and force all those noisy [0-5] values to be uniform 5
		img_gray = np.clip(img_gray, COLORCLIP_AMOUNT, 255-COLORCLIP_AMOUNT)
		temp_gray = np.clip(temp_gray, COLORCLIP_AMOUNT, 255-COLORCLIP_AMOUNT)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	print("calculating diff...", end="")
	(score, diff) = structural_similarity(img_gray, temp_gray, full=True)
	if debug: print("SSIM: %f" % score)
	# diff comes back as array of float -1.0 to 1.0, cast to unsigned int for image display
	diff = (diff * 255).astype("uint8")
	
	# next, binarize the difference mask (OTSU uses a fancy auto threshold thing? might be bad)
	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	
	print("cleaning diff...")
	# next, use contours as a way to fill all internal holes
	# returns as a list of "contour objects" that describe the perimiter of each regions, don't care yet
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	mask_1 = np.zeros(thresh.shape, dtype='uint8')
	# just draw them back to the mask, all i care about for now is filling in the regions
	# "index" -1 means all, "line thickness" -1 means fill!
	cv2.drawContours(mask_1, contours, -1, (255, 255, 255), -1)

	# next, use opening & closing!
	mask_2 = mask_1.copy()
	# opening will remove some noise (the lines)
	kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_SIZE, OPENING_SIZE))
	mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_OPEN, kernel_open)
	# closing will link nearby regions, hopefully closing any crescent-shapes
	kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_SIZE, CLOSING_SIZE))
	mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_CLOSE, kernel_close)
	
	# next, use contours a second time!
	# this time it's because I want to discard small ones, also because i want to ultimately
	# be returning a list of the differing regions and contours are perfect for that.
	contours = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	# discard any contours that dont enclose enough total area
	contours_final = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
	if debug: print("found %d regions" % len(contours_final))
	
	if debug:
		marked = image.copy()
		cv2.drawContours(marked, contours_final, -1, (0,255,255), 2)
		
		cv2.imshow("Marked Image", my_resize(marked, newheight=DISPLAY_HEIGHT))
		
		# cv2.imwrite("2_rawdiff.jpg", diff)
		diff2 = my_resize(diff, newheight=DISPLAY_HEIGHT)
		
		mask_3 = np.zeros(image.shape, dtype='uint8')
		# "index" -1 means all, "line thickness" -1 means fill!
		cv2.drawContours(mask_3, contours_final, -1, (255, 255, 255), -1)
		# cv2.imwrite("3_thresh.jpg", thresh)
		# cv2.imwrite("4_cleandiff.jpg", mask_3)
		
		thresh2 = my_resize(thresh, newheight=DISPLAY_HEIGHT)
		mask_1r = my_resize(mask_1, newheight=DISPLAY_HEIGHT)
		mask_2r = my_resize(mask_2, newheight=DISPLAY_HEIGHT)
		mask_3r = my_resize(mask_3, newheight=DISPLAY_HEIGHT)
		cv2.imshow("Raw Difference", diff2)
		# cv2.imwrite("diff22_uncorrected.png", diff2)
		print("debug: press any key to continue")
		frame = 0
		while True:
			if frame == 0:
				cv2.imshow("Thresh (animated)", thresh2)
			elif frame == 1:
				cv2.imshow("Thresh (animated)", mask_1r)
			elif frame == 2:
				cv2.imshow("Thresh (animated)", mask_2r)
			elif frame == 3:
				cv2.imshow("Thresh (animated)", mask_3r)
			frame = (frame + 1) % 4
			v = cv2.waitKey(DEBUG_DIFF_ANIMATION_TIME)
			if v != -1:
				break
		cv2.destroyWindow("Marked Image")
		cv2.destroyWindow("Thresh (animated)")
		cv2.destroyWindow("Raw Difference")
		
	return contours_final


# this is the mouse click callback
# very simple, whenever a click happens, set a flag saying it happened and where
# handle the rest outside the callback
ixy = []
def mouse_callback(event, x, y, flags, param):
	global ixy
	if event == cv2.EVENT_LBUTTONDOWN:
		ixy = [x,y]

def interactive_merge_images(newimg: np.ndarray, baseimg: np.ndarray, contours: list, debug=False):
	"""Step 3 of the 3-step process. Select which image source to use for each region."""
	global ixy
	fill_borders = False
	lastregion = -1
	superlist = []
	scale = newimg.shape[0] / DISPLAY_HEIGHT
	
	# INTERFACE PLAN:
	# display the JP image with all detected diff regions shown as BLINKING OUTLINES (green? sure)
	# initially, all diff regions show JP
	# set up mouse event listener, click in each outline to toggle from JP to EN version, or back again
	# outlines are blinking so you can see what they look like when they're gone
	# press enter to commit & save changes, then load the next
	
	def redraw_img_for_display():
		# read superlist, preview from above (not as args)
		ret = preview.copy()
		for cc in superlist:
			if cc[1]: # use en! draw in blue!
				color = NEWIMG_COLOR
			else: # use jp! draw in red
				color = BASEIMG_COLOR
			# draw the contour on the preview in the proper color
			if fill_borders:
				cv2.drawContours(ret, [cc[0]], 0, color, -1)
			else:
				cv2.drawContours(ret, [cc[0]], 0, color, BORDER_THICKNESS)
		# now have preview image w/ contour lines drawn in corresponding colors
		preview22 = my_resize(preview, newheight=DISPLAY_HEIGHT)
		preview_outlines22 = my_resize(ret, newheight=DISPLAY_HEIGHT)
		return preview22, preview_outlines22
	
	def paste_contour_region(which2):
		cont2, is_en2, mask2 = superlist[which2]
		for ii in range(3):
			# use the 2d mask to transfer data between the 3d images... ugly but w/e
			if is_en2:  # region is currently EN, so copy from EN onto preview
				np.copyto(preview[:,:,ii], newimg[:,:,ii], where=np.not_equal(mask2, 0))
			else:  # region is currently JP, so copy from JP onto preivew
				np.copyto(preview[:,:,ii], baseimg[:,:,ii], where=np.not_equal(mask2, 0))
	
	# first, build a superlist of all contours & associated data
	# item format is [contour, (1=use en, 0=use jp), image_of_just_that_contour]
	superlist_original = []
	for c in contours:
		# make a 2d mask
		mask = np.zeros(newimg.shape[0:2], dtype="uint8")
		# thickness -1 means fill
		cv2.drawContours(mask, [c], 0, 255, -1)
		superlist_original.append([c, 0, mask])
		
	# make a copy! and superlist_original stays in case I want to reset to base state
	superlist = copy.deepcopy(superlist_original)
	# the preview starts as an unmodified copy of the JP template
	preview = baseimg.copy()
	
	# create the smaller displayable versions
	preview2, preview_outlines2 = redraw_img_for_display()
	
	cv2.namedWindow('Merged Image (interactive preview)', flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL))
	# cv2.namedWindow('Merged Image (interactive preview)', flags=(cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_NORMAL))
	cv2.moveWindow('Merged Image (interactive preview)',0,0)
	cv2.setMouseCallback('Merged Image (interactive preview)', mouse_callback)
	
	print(">>>>>> begin interactive merge <<<<<<")
	print("found %d regions of major difference" % len(superlist))
	print("click on a region to toggle its source image")
	print("   ENTER      ... commit & save")
	print("   ESC        ... abort & exit")
	print("   R          ... reset all changes")
	print("   - (or _)   ... shrink last region clicked")
	print("   + (or =)   ... grow last region clicked")
	print("   F          ... toggle border-region or fill-region (preview only)")
	print("   D          ... draw a box to mark as different")
	print("   N          ... draw a box to mark as NOT different")
	print("   B          ... set all regions to BASE image")
	print("   S          ... set all regions to SECONDARY image")
	
	framenum = 0
	while True:
		##########################
		# first, handle clicks
		if ixy:
			# this means a click has happened!
			# 1. transform display-image coordinates to full-image coordinates
			ixy = [round(xy * scale) for xy in ixy]
			# 2. determine which contour it was on/inside by looking at filled versions of each contour
			which = -1
			for d,c in enumerate(superlist):
				cont, is_en, mask = c
				# note: need to access pixel values with y,x
				if mask[ixy[1],ixy[0]] != 0:
					which = d
					# 3, swap the EN/JP it is linked with
					c[1] = not c[1]
					# 4, paste that section of "newimg" or "baseimg" onto "preview"
					paste_contour_region(which)
					# 5, update the smaller displayable versions
					preview2, preview_outlines2 = redraw_img_for_display()
					break
			# save the index of the last contour clicked on
			lastregion = which
			if debug: print("click: xy={} region={}".format(ixy, which))
			# whether it was a click inside a contour or not, clear the flag
			ixy = []
			
		v = cv2.waitKey(25)
		
		if debug and v!=-1: print(v)
		
		##########################
		# next, handle key presses
		# esc = 27, enter = 13, others = ord("x")
		if v == -1:
			pass
		elif v in KEYS_ALLSEC:
			for d,s in enumerate(superlist):
				if s[1] != 1:
					s[1] = 1
					paste_contour_region(d)
			preview2, preview_outlines2 = redraw_img_for_display()
		elif v in KEYS_ALLBASE:
			for d,s in enumerate(superlist):
				if s[1] != 0:
					s[1] = 0
					paste_contour_region(d)
			preview2, preview_outlines2 = redraw_img_for_display()
		elif v in KEYS_RESET:
			print("reset")
			lastregion = -1
			fill_borders = False
			superlist = copy.deepcopy(superlist_original)
			preview = baseimg.copy()
			preview2, preview_outlines2 = redraw_img_for_display()
		elif v == 27:  # esc
			print("exit")
			cv2.destroyWindow("Merged Image (interactive preview)")
			return None
		elif v == 13: # or v == ord(' '): # enter or space
			print("commit changes!")
			break
		elif v in KEYS_FILL:
			print("toggle fill/borders")
			fill_borders = not fill_borders
			preview2, preview_outlines2 = redraw_img_for_display()
		elif v in KEYS_MARKDIFF or v in KEYS_NOTDIFF:
			if v in KEYS_MARKDIFF:
				print("mark region as diff:")
			else:
				print("mark region as NOT diff:")
			# print("   click-and-drag to select a rectangle region")
			# print("   ESC, SPACE, or ENTER to submit selection")
			# print("   'c' to cancel")
			# BUG: for some reason the script prints the instructions only after the script completes?????
			#   and it prints it for every time the selectROI was called...
			#   actually, this only happens when running in Pycharm. strange.
			# due to the ROI function we are gonna be stuck here until it returns. no blinking.
			# begin ROI prompt. force window to display the image i want.
			# "r" is in format (x1, y1, width, height)
			r = cv2.selectROI("Merged Image (interactive preview)", preview_outlines2, showCrosshair=False, fromCenter=False)
			# when done, must restore the mouse callback!
			cv2.setMouseCallback('Merged Image (interactive preview)', mouse_callback)
			# only continue past here if a real-sized rect is selected
			if r[2] != 0 and r[3] != 0:
				# transform display-image coordinates to full-image coordinates
				r = [round(xy * scale) for xy in r]
				# draw all contours onto a zero-mask, then draw the box in white, then completely re-evaluate contours
				newconts = np.zeros(newimg.shape[0:2], dtype="uint8")  # blank mask
				existing_contours = [c[0] for c in superlist]
				cv2.drawContours(newconts, existing_contours, -1, 255, -1)  # draw all contours w/ fill
				if v in KEYS_MARKDIFF:
					cv2.rectangle(newconts, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), 255, -1)  # draw rect in white w/ fill
				else:
					cv2.rectangle(newconts, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), 0, -1)  # draw rect in BLACK w/ fill
				contours = cv2.findContours(newconts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours = contours[0] if len(contours) == 2 else contours[1]
				new_superlist = []
				for d,sss in enumerate(superlist):
					match = -1 # does the contour of this entry show up in the list of new contours?
					for i in range(len(contours)):
						if np.array_equiv(sss[0],contours[i]):
							match = i
							break
					if match == -1: # if no, set it to jp then paste
						sss[1] = 0
						paste_contour_region(d)
					else: # if yes, add it to the output & remove from list of new contours
						new_superlist.append(sss)
						contours.pop(match)
				aaaa = len(contours)                        # number added
				bbbb = len(superlist) - len(new_superlist)  # number deleted
				# any new contours get created as jp, no paste necessary
				for c in contours:
					mask = np.zeros(newimg.shape[0:2], dtype="uint8")
					cv2.drawContours(mask, [c], 0, 255, -1)
					new_superlist.append([c, 0, mask])
				# overwrite superlist
				superlist = new_superlist
				if debug:
					print("%d added, %d deleted, now %d" % (aaaa, bbbb, len(superlist)))
				# unset the lastregion cuz i changed the superlist
				lastregion = -1
				# FINALLY, redraw
				preview2, preview_outlines2 = redraw_img_for_display()
			pass
		elif v in KEYS_SHRINK:
			# when a region shrinks, it may break a bridge and split into 2 (or more) regions.
			# when this happens, the regions are "deselected" and all are returned to JP state.
			if lastregion == -1:
				print("no region selected, cannot shrink")
			else:
				sss = superlist[lastregion]
				if debug: print("erode(shrink): region %d" % lastregion)
				# when shrinking the region, I need to put back all the JP pixels that were "behind" the borders of the region...
				region_status_was = sss[1]  # save how it was, so i can put it back
				sss[1] = 0  # force this region to JP mode
				paste_contour_region(lastregion)  # copy the JP portion to the preview img
				sss[1] = region_status_was  # set this region to the mode it originally had
				# THEN do the shrink
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INTERACTIVE_GROWSHRINK_SIZE, INTERACTIVE_GROWSHRINK_SIZE))
				sss[2] = cv2.morphologyEx(sss[2], cv2.MORPH_ERODE, kernel)
				# then, build a new contour from this mask
				contours = cv2.findContours(sss[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours = contours[0] if len(contours) == 2 else contours[1]
				if len(contours) != 1:
					# note: after regions split, set all new regions to jp so they need to be toggled again if desired
					# note: a region can only split after a 'shrink' operation. that operation includes copying from jp onto preview
					print("WARNING: region %d just split into %d separate regions, please re-select the one you want to be modifying" % (
						lastregion, len(contours)))
					superlist.pop(lastregion)  # the current entry for region within superlist (sss) is invalid, must be deleted
					lastregion = -1  # set "last region" to invalid too, must be selected again
					# for each of the new contours, build a new "superlist" item
					for c in contours:
						# make a 2d mask
						mask = np.zeros(newimg.shape[0:2], dtype="uint8")
						# thickness -1 means fill
						cv2.drawContours(mask, [c], 0, 255, -1)
						superlist.append([c, 0, mask])
				else:
					# save the new contour into the superlist so the border visibly grows
					sss[0] = contours[0]
					# re-update the paste thing, doesn't depend on the actual contour part, only the mask part
					paste_contour_region(lastregion)
				# redraw the display images to update the borders
				preview2, preview_outlines2 = redraw_img_for_display()

		elif v in KEYS_GROW:
			# when regions grow, they may start to overlap. it may make things odd but not my problem.
			# they may also close a loop and form internal holes, these will be filled.
			if lastregion == -1:
				print("no region selected, cannot grow")
			else:
				sss = superlist[lastregion]
				if debug: print("dilate(grow): region %d" % lastregion)
				# first, perform dilate on the mask
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INTERACTIVE_GROWSHRINK_SIZE, INTERACTIVE_GROWSHRINK_SIZE))
				sss[2] = cv2.morphologyEx(sss[2], cv2.MORPH_DILATE, kernel)
				# then, build a new contour from this mask
				# (guaranteed to not split, guaranteed to get exactly 1 contour in return, even if it closes a loop)
				contours = cv2.findContours(sss[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours = contours[0] if len(contours) == 2 else contours[1]
				# fill & set mask, to handle if loop becomes closed
				cv2.drawContours(sss[2], contours, 0, 255, -1)
				# save the new contour into the superlist so the border visibly grows
				sss[0] = contours[0]
				# re-update the paste thing, doesn't depend on the actual contour part, only the mask part
				paste_contour_region(lastregion)
				# finally, redraw the display images to update the borders
				preview2, preview_outlines2 = redraw_img_for_display()

		
		##########################
		# last, display/animate the blinking frames
		if framenum < int(PREVIEW_BLINK_DUTY_CYCLE * PREVIEW_BLINK_RATE_MS / 25):
			# with outlines
			cv2.imshow("Merged Image (interactive preview)", preview_outlines2)
		else:
			# without outlines
			# (preview of what will be saved)
			cv2.imshow("Merged Image (interactive preview)", preview2)
		# inc with wrap
		framenum = (framenum + 1) % int(PREVIEW_BLINK_RATE_MS / 25)
	
	cv2.destroyWindow("Merged Image (interactive preview)")
	# done editing, now return the final product
	return preview


####################################################################################################
####################################################################################################
####################################################################################################

def everything(filename_baseimg, filename_newimg, filename_outimg):
	"""Execute the 3-step process needed to merge the images."""
	# safely open the baseimg
	print("reading baseimg = '%s'" % filename_baseimg)
	if not os.path.isfile(filename_baseimg):
		print("ERROR: file '%s' does not exist." % filename_baseimg)
		return
	try:
		template = cv2.imread(filename_baseimg)
	except Exception as e:
		print(e.__class__.__name__, e)
		print("ERROR: file '%s' could not be opened for some reason?" % filename_baseimg)
		return
	# safely open the newimg
	print("reading newimg =  '%s'" % filename_newimg)
	if not os.path.isfile(filename_newimg):
		print("ERROR: file '%s' does not exist." % filename_newimg)
		return
	try:
		image = cv2.imread(filename_newimg)
	except Exception as e:
		print(e.__class__.__name__, e)
		print("ERROR: file '%s' could not be opened for some reason?" % filename_newimg)
		return

	# both images need to be RGB, or both greyscale
	assert len(template.shape) == len(image.shape)
	if len(template.shape) == 3:
		assert template.shape[2] == image.shape[2]
	
	# verify that the output path makes sense
	if os.path.splitext(filename_outimg)[1] == '':
		print("ERROR: filetype extension of output file '%s' is missing?" % filename_outimg)
		return
	
	# FIRST, use fancy algorithms to ensure the two images are aligned
	image_aligned = align_images(image, template, debug=DEBUG_ALIGNMENT,
								 maxFeatures=ALIGN_MAX_FEATURES,
								 keepPercent=ALIGN_KEEP_PERCENT)
	
	if SAVE_ALIGNED_IMAGE:
		print("saving aligned image to '%s'" % ALIGNED_IMAGE_NAME)
		cv2.imwrite(ALIGNED_IMAGE_NAME, image_aligned)
	
	# SECOND, detect the regions where the two images differ
	contours = find_image_differences(image_aligned, template, debug=DEBUG_DIFF)
	
	# THIRD, open the interactive window to select which diff regions to toggle
	merged = interactive_merge_images(image_aligned, template, contours, debug=DEBUG_INTERACTIVE)
	
	if merged is not None:
		print("saving '%s'..." % filename_outimg, end="")
		try:
			# create folder if necessary
			if os.path.dirname(filename_outimg):
				os.makedirs(os.path.abspath(os.path.dirname(filename_outimg)), exist_ok=True)
			# TODO: do write with PIL, maybe? what is advantage, disadvantage?
			# NOTE: default JPEG write quality is 95
			cv2.imwrite(filename_outimg, merged)
		except Exception as e:
			print("")
			print(e.__class__.__name__, e)
			print("ERROR: cannot save result to destination '%s' for some reason?" % filename_outimg)
			return
		print("...done")

	return

####################################################################################################
####################################################################################################
####################################################################################################

def main():
	"""How to iterate over files to do merge for all pages."""
	
	print("Select input mode: 1=batch, 2=single-page")
	r = input("> ")
	
	if r not in ("1", "2"):
		print("invalid input!")
		return None
	
	# this is a list where each entry is a page to merge, [base new out]
	process_me = []
	
	if r == '1':
		# batch mode (queue up many pages at once)
		print("Running in multi-page mode")
		print("  note: filepath templates follow the Python Formatting String syntax.")
		print("  each template should have a slot for exactly one integer number that corresponds to the page number.")
		print("  example: book_jp/scan_{:03d}.jpg will generate a filepath like book_jp/scan_009.jpg or book_jp/scan_013.jpg")
		print('  {:03d} will be replaced with an integer and left-filled with zeros to 3 digits.')
		print("  website: https://docs.python.org/3/library/string.html#format-string-syntax")
		print("")
		
		print("Please enter the GENERIC FILEPATH TEMPLATE for the BASE image files:")
		r = input("> ")
		# strip whitespace cuz yeah
		base_template = os.path.normpath(r).strip()
		try:
			base_template.format(4)
		except Exception as e:
			print(e.__class__.__name__, e)
			print("invalid filepath template")
			return None
		
		print("What page number should the BASE image files begin counting from?")
		r = input("> ")
		try:
			base_startnum = int(r)
		except ValueError:
			print("invalid input")
			return None
		
		print("Please enter the GENERIC FILEPATH TEMPLATE for the SECONDARY image files you will take pieces from:")
		r = input("> ")
		# strip whitespace cuz yeah
		new_template = os.path.normpath(r).strip()
		try:
			new_template.format(4)
		except Exception as e:
			print(e.__class__.__name__, e)
			print("invalid filepath template")
			return None
		
		print("What page number should the SECONDARY image files begin counting from?")
		r = input("> ")
		try:
			new_startnum = int(r)
		except ValueError:
			print("invalid input")
			return None
		
		print("Please enter the GENERIC FILEPATH TEMPLATE for the OUTPUT image files:")
		r = input("> ")
		# strip whitespace cuz yeah
		out_template = os.path.normpath(r).strip()
		try:
			out_template.format(4)
		except Exception as e:
			print(e.__class__.__name__, e)
			print("invalid filepath template")
			return None
		
		print("How many pairs of pages will be merged?")
		r = input("> ")
		try:
			runlength = int(r)
		except ValueError:
			print("invalid input")
			return None
		
		# now have base_template, base_startnum, new_template, new_startnum, runlength, out_template
		# use these to iterate and fill the process_me list
		base_run = list(range(base_startnum, base_startnum + runlength))
		new_run = list(range(new_startnum, new_startnum + runlength))
		assert len(base_run) == len(new_run) == runlength
		for basenum, newnum in zip(base_run, new_run):
			item = [base_template.format(basenum),
					new_template.format(newnum),
					out_template.format(basenum)]
			process_me.append(item)
		pass
	elif r == '2':
		# single-page mode
		# prompt for the two input files & name/location of the output file
		print("Running in single-page mode")
		print("Please enter the filepath of the BASE image file:")
		r = input("> ")
		base_path = os.path.normpath(r).strip()
		
		print("Please enter the filepath of the SECONDARY image file you will take pieces from:")
		r = input("> ")
		new_path = os.path.normpath(r).strip()
		
		print("Please enter the filepath for the desired output file:")
		r = input("> ")
		out_path = os.path.normpath(r).strip()
		
		# assemble them into the same list-structure that the batch mode uses
		item = (base_path, new_path, out_path)
		process_me.append(item)
		pass
		
	# now i have a list of all the [in1, in2, out] filename threeples to process
	
	# shuffle the list
	if RANDOMIZE_PAGE_ORDER:
		print("shuffling page order")
		random.shuffle(process_me)
	
	# now iterate over all threeples and do the merging
	for d,group in enumerate(process_me):
		print("")
		print(">>>> processing page %d / %d" % (d+1, len(process_me)))
		filename_base, filename_secondary, filename_out = group
		everything(filename_base, filename_secondary, filename_out)
		
	input("done! press ENTER to quit")
	return None


if __name__ == "__main__":
	# root = tkinter.Tk()
	# height = root.winfo_screenheight()
	# # root.quit()
	# root.destroy()
	# recommend_height = height - 100
	# print("detect total screen height = '%d', recommended display height = '%d'" % (height, recommend_height))
	# if DISPLAY_HEIGHT <= 0:
	# 	DISPLAY_HEIGHT = recommend_height
	print("")
	print("using display_height = '%d'" % DISPLAY_HEIGHT)
	print("to change the display height, open this file with a text editor and")
	print("change DISPLAY_HEIGHT near the top of the file. different monitors will")
	print("have more or less resolution to display images with, I can't cleanly")
	print("auto-detect the screen height.")
	print("")
	print("note: the parameters near the top of the file marked with ### may need")
	print("to be tweaked depening on the resolution of the input images.")
	print("")
	print("note: the outupt image will have the same resolution as the BASE image, and")
	print("will have the same page numbers as the BASE image.")
	print("")

	main()
	# everything("input/tamaya80_jp.jpg",
	# 		   "input/tamaya80_en.jpg",
	# 		   "out2.png")

# TODO: how hard would it be to convert the interface to use TK?
# TODO: optional color-matching stage? some versions have higher/lower contrast, it would be smart to match that...
