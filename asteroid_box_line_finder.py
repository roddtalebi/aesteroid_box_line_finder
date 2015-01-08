# This file strictly outputs and saves data

print "\n------------------------------------------------------------"
# $ Python Final.py (names of fits files) (names of txt files) (n=#candidates)
#					(size=est size of asteroid) (box=x1:x2,y1:y2 is box to be excluded)
#					(isbinary=True for binary files) (ncol=#col(x) ) (nrow=#row(y) )
#                   (drift=# this gives the drift rate for seconds/pixel OR m/s, can be + or -
#		also make drift so it can be a range drift=start:finish,increment
#               If conversion rate for meters to pixel given (#meters/pixel), then assume drift is m/s, else assume sec/pixel)
#               NOTE: only thing that can have a numeric decimal is drift
#				times.dat file

from sys import argv
from string import find

#set defaults
file_list = "0"
text = "default.txt"
r1 =5
size = 3
box = None
xd = 100
yd = 2046
drift = '0'
Meters = False
conversion = 1
dat = False
Boxes = True
Lines = True
#extract values
for i in range (0, len(argv)):
	
	# find fits files
	a = find(argv[i], '.fits')
	if a != -1:
		if file_list == "0":
			file_list = []
		file_list.append(argv[i])
		isbinary = False
	
	#find binary files
	b = find(argv[i], 'binary')
	if b != -1:
		verdict = argv[i][len(argv[i])-4:]
		if verdict == "True":
			isbinary = True
	
	c = find(argv[i], '.')
	if c != -1 and i >= 1: #has . in name
		d = find(argv[i], '.txt') #just to make sure not txt file
		if d == -1: #so it isn't a .txt file
			e = find(argv[i], '.fits') # make sure not fits file
			if e == -1: # not fits file
				if file_list == "0":
					file_list = []
				asd = find(argv[i], 'drift')
				if asd == -1:
					asde = find(argv[i], '.dat')
					if asde == -1:
				 		file_list.append(argv[i])
	
	# find text file
	ff = find(argv[i], '.txt')
	if ff != -1:
		text = argv[i]
	
	# find candidates
	g = find(argv[i], 'n=')
	if g != -1:
		r1 = int(argv[i][2:])
	
	# find size of box
	h = find(argv[i], 'size=')
	if h != -1:
		size = int(argv[i][5:])
	
	#exclude this box
	hh = find(argv[i], 'box=')
	if hh != -1:
		box = "Found"
		#in formate of box=x1:x2,y1:y2
		# now extract those terms
		m = find(argv[i], ',')
		
		xrange = argv[i][4:m]
		j = find(xrange, ':')
		x1 = int(xrange[:j])
		x2 = int(xrange[j+1:])
		
		yrange = argv[i][m+1:]
		l = find(yrange, ':')
		y1 = int(yrange[:l])
		y2 = int(yrange[l+1:])
	
	#set dimensions
	aa = find(argv[i], 'ncol=')
	if aa != -1:
		xd = int(argv[i][5:])
	m = find(argv[i], 'nrow=')
	if m != -1:
		yd = int(argv[i][5:])

	# find drift rate
	qq = find(argv[i], 'drift=')
	if qq != -1:
		qr = find(argv[i], ':')
		if qr != -1:
			dstart = float(argv[i][6:qr])
			qrq = find(argv[i], ',')
			if qrq != -1:
				dfinish = float(argv[i][qr+1:qrq])
				dstep = float(argv[i][qrq+1:])
				drift = [dstart, dfinish, dstep]
		else:
			drift = argv[i][6:]
	#find converstion rate
	rr = find(argv[i], 'meters/pixel')
	if rr != -1:
		Meters = True
		conversion = int(argv[i][:rr])
		#meters/pixel
	
	#find .dat files
	prp = find(argv[i], '.dat')
	if prp != -1:
		dat_name = argv[i]
		dat_file = open(argv[i], 'r')
		dat_lines = dat_file.readlines()
		dat = True
	
	#check to see what program should look up: Boxes, Lines, or Both
	woop = find(argv[i], 'Boxes')
	if woop != -1:
		Lines = False #only check boxes
	poow = find(argv[i], 'Lines')
	if poow != -1:
		Boxes = False #only check lines

                



#check work
print "Files =", file_list
print "Number of Files =", len(file_list)
print "Is it a Raw Binary File?", isbinary
print "Save data to text file:", text
print "Look at", r1, "Canidates"
print "Size of Boxes to look at is:", size
print "The included drift rate btwn images is", drift, "seconds per pixel"
print "Did you find any boxes to exclude?", box
print "The dimensions are", yd, "rows (y-value) and", xd, "columns (x-values)"
print "------------------------------------------------------------\n"

#-----------------------------------------------
#introduce the probability density
# just assume that it follows the normal standard distribution

import numpy as np
import math

def fff(x):
	return 1/(np.sqrt(2*math.pi))*np.exp((-x**2)/2)

def dist(x, a, b):
	# a is the mean. b is the standard deviation
	dist = 1 / (b * np.sqrt(2*math.pi)) * np.exp(-((x-a)**2) / (2*b**2))
	return dist

#-----------------------------------------------
# calculate and estimate paramters for satelite asteroid

#ask for (m/s and pixel conversion) or (sec/pixel)


#NOTE: How to get velocity of satelite
#velocity=np.sqrt(G*(M_main + M_satelite)/orbit_radius)
# where G is gravitational constant

#radius = 300 #meters
#vol = math.pi*(radius**3) #meters cubed
#density = 2000 #kg/m^3
#mass = vol*density #kg
#orbit_rad = 30000 #meters. 3km
#conversion = 300 #300meters/1pixel



#-----------------------------------------------
# extract time differences in images to incorporate drift

def time_convert(hms):
        #converts HMS (Hours Minutes Seconds), to total seconds
        Num = len(hms)
        if Num >= 8:
        	day = int(hms[Num-8:Num-6])
        else:
        	day = 0
        hour = int(hms[Num-6:Num-4])
        minute = int(hms[Num-4:Num-2])
        second = int(hms[Num-2:])
        totsecond = day*24*60*60+hour*60*60+minute*60+second
        return totsecond

#drift rate = seconds per pixel

if type(drift) != str:
	diff = abs(dfinish-dstart)
	n = int(round(diff/dstep))
	drift = []
	for i in range (0, n+1):
		drift.append(dstart + i*dstep)
	if drift[n] != dfinish:
		drift.append(dfinish)
	
	jic = []
	jic_drift = []
	if Meters == True:
		jic_drift = drift[:]
		for j in range (0, len(drift)):
			jic.append("or "+ str(drift[j]) + ' m/s')
			if drift[j] != 0:
				drift[j] = conversion/float(drift[j])
	else:
		for j in range (0, len(drift)):
			jic.append("")
			jic_drift.append(0.0)


if type(drift) == str and drift != '0':
	drift = [float(drift)]
	
	if Meters == True:
		jic_drift = drift[:]
		jic = ["or " + str(drift[0]) + ' m/s']
		drift = [conversion/drift[0]]
	else:
		jic=[""]
		jic_drift=[0.0]


#want drift to be seconds/pixel
if drift != '0':
        time_list = []
        image_drift = []
        if isbinary==False:
                #Then Fits File
                for i in range (0, len(file_list)):
                	
					if dat == False:
						z=find(file_list[i], '.fits')
						if z != -1:
							time_list.append(time_convert(file_list[i][z-6:z]))
                    
					if dat == True:
						gh = len(dat_lines[i])
						time_list.append(time_convert(dat_lines[i][:gh-1]))
                    	
                        #time_list gives the time each image taken
                #find the pixel drift between each image
                for z in range (0, len(drift)):
					im_drift = []
					for j in range (0, len(time_list)):
						diff = time_list[j]-time_list[0]
						if drift[z] == 0:
							im_drift.append(int(round(float(0))))
						if drift[z] != 0:
							im_drift.append(int(round(diff/float(drift[z]))))
							#im_drift gives pixel shift for each image wrt to first image
					image_drift.append(im_drift)

        if isbinary==True:
			#Then Binary File
			for i in range (0, len(file_list)):
				
				if dat == False:
					z=find(file_list[i], '.map')
					if z != -1:
					 time_list.append(time_convert(file_list[i][z-6:z]))
				
				if dat == True:
					gh = len(dat_lines[i])
					time_list.append(time_convert(dat_lines[i][:gh-1]))
            
			for z in range (0, len(drift)):
				im_drift = []
				for j in range (0, len(time_list)):
					diff = time_list[j]-time_list[0]
					if drift[z] == 0:
						im_drift.append(int(round(float(0))))
					if drift[z] != 0:
						im_drift.append(int(round(diff/float(drift[z]))))
						#im_drift gives pixel shift for each image wrt to first image
				image_drift.append(im_drift)
else:
	drift = [0]
	jic = drift
	jic_drift = [0.0]
	image_drift = np.empty((len(drift), len(file_list)))
	for i in range (0, len(drift)):
		for j in range (0, len(file_list)):
			image_drift[i][j]=0

dstart = drift[0]

#each element of image drift represents each drift rate, and one element contains the pixel shift of each image wrt first image

#-----------------------------------------------
# open fits file
# set fits_name = file name
import pyfits
import struct
from scipy.optimize import curve_fit

max_X = xd-1 #column 
max_Y = yd-1 #row

#extract points from either .fits file or raw binary data

#.FITS file
yx_files = []
if isbinary == False:
	#if it is a fits file, assume it is already scaled
	for i in range (0, len(file_list)):
		fits_name = file_list[i]
		hdulist = pyfits.open(fits_name)
		yx = hdulist[0].data
		yx_files.append((yx))
		mean=0
		stdev=0

#BINARY RAW files
else:
	#scale these files regardless
	mean = []
	stdev = []
	for i in range (0, len(file_list)):
		yx = np.empty((yd, xd)) #create tuple array with appropriate dimensions
		nbytes = 4
		endian = "<" #little endian
		dtype = "f"
		x = "a"*nbytes # creates string st x = "aaaa" if nbyte = 4
		c = 0
		with open(file_list[i], 'rb') as f: #use 'rb' for non text files
			while True:
				x = f.read(nbytes)
				#file.read(size)-->reads some quantitiy of data and returns it as a string
				if len(x) != nbytes: break 
					#SEE IMG.PY FOR DETAILED EXPLAINATION
				val = struct.unpack(endian+dtype,x)[0]
				
				#now add each item to point
				row = int(c/xd)
				col = c - row*xd
				#print col, row
				yx[row][col] = val
				c += 1
		#now get standard dev and mean of image
		# use only first and last 1/4 columns to exclude asteroid
		l = xd/4
		l2 = xd-xd/4
		a = []
		for m in range (0, yd):
			for n in range (0, l):
				a.append(yx[m][n])
			for o in range (l2, xd):
				a.append(yx[m][o])
		N = len(a)
		n = N/10
		#a must increase monotonically
		a.sort(key=None, reverse=False)
		
		#counts, bins = np.histogram(a, bins=n, density=True)
		#x = bins[:-1] + (bins[1] - bins[0])/2 # convert bin edges to center
		## now guess line to standard gaussian curve
		#p_opt, p_cov = curve_fit(dist, x, counts, p0=[0, 1])
		##p_opt are optimal parameters 
		yx_files.append((yx))
		mean.append(np.mean(a))
		stdev.append(np.std(a)**2)
	mean = sum(mean)
	stdev = sum(stdev)

# All points extracted and stored from each file.
# Now go and sum images by each drift rate		


def sum_images(yx1, image_drift, isbinary, mean, stdev):
	#.Fits files
	if isbinary==False:
		yxdata = []
		yxdata = [[0 for col in range(xd)] for row in range(yd)]
		for zz in range (0, len(yx1)):
			print "image", zz
			yx = np.empty((yd, xd))
			if sum(image_drift) > 0 and drift != [0]:
				pdrift = image_drift[zz]
				extra = yx1[zz][yd-pdrift:]
				yx[pdrift:] = yx1[zz][:yd-pdrift]
				yx[:pdrift] = extra
			elif sum(image_drift) < 0:
				pdrift = abs(image_drift[zz])
				extra = yx1[zz][:pdrift]
				yx[:yd-pdrift] = yx1[zz][pdrift:]
				yx[yd-pdrift:] = extra
			elif drift == [0] or sum(image_drift) == 0:
				yx = yx1[zz][:]
			for m in range (0, yd):
				for n in range (0, xd):
					yxdata[m][n] += yx[m][n]
		#now yxdata is sum of all images. next divide by standard dev
		#since assuming scaled images, each has standard dev of 1, so divide by 
		# sqrt of number of images
		for m in range (0, yd):
			for n in range (0, xd):
				sq = np.sqrt(len(file_list))
				yxdata[m][n] = yxdata[m][n]/sq
		#wrt to the ds9 image axis, yxdata had reversed plots
		# also the yxdata is initialized at (0, 0) rather than (1, 1) for ds9
	#BINARY RAW files
	else:
		yxdata = []
		yxdata = [[0 for col in range(xd)] for row in range(yd)]
		for zz in range (0, len(yx1)):
			print "image", zz
			yx = np.empty((yd, xd))
			if sum(image_drift) > 0 and drift != [0]:
				pdrift = image_drift[zz]
				extra = yx1[zz][yd-pdrift:]
				yx[pdrift:] = yx1[zz][:yd-pdrift]
				yx[:pdrift] = extra
			elif sum(image_drift) < 0:
				pdrift = abs(image_drift[zz])
				extra = yx1[zz][:pdrift]
				yx[:yd-pdrift] = yx1[zz][pdrift:]
				yx[yd-pdrift:] = extra
			elif drift == [0] or sum(image_drift) == 0:
				yx = yx1[zz][:]
			for m in range (0, yd):
				for n in range (0, xd):
					yxdata[m][n] += yx[m][n]
		#now scale the final data, yxdata
		sq = np.sqrt(stdev)
		for m in range (0, yd):
			for n in range (0, xd):
				yxdata[m][n] = (yxdata[m][n]-mean)/sq
	#-------
	#now rescale final image just to be more accurate
	l = xd/4
	l2 = xd-xd/4
	a = []
	for j in range (0, yd):
		for i in range (0, l):
			a.append(yxdata[j][i])
		for o in range (l2, xd):
			a.append(yxdata[j][o])
	N = len(a)
	n = N/10
	#make it increase monotonically		
	a.sort(key=None, reverse=False)
	#counts, bins = np.histogram(a, bins=n, density=True)
	##maybe try changing bins to be more like example
	#x = bins[:-1] + (bins[1] - bins[0])/2   # convert bin edges to centers
	## now have our data: x and counts.
	## lets try to get the best fit line by guessing the gaussian distribution
	#p_opt, p_cov = curve_fit(dist, x, counts, p0=[0, 1])
	mean = np.mean(a)
	stdev = np.std(a)**2
	#now scale the final data, yxdata
	sq = np.sqrt(stdev)
	for m in range (0, yd):
		for n in range (0, xd):
			yxdata[m][n] = (yxdata[m][n]-mean)/sq
	#-------
	print "Files Read"
	return yxdata


#-----------------------------------------------
# if a box is to be excluded, let it be introduced as
# x1, x2, y1, y2
#to exclude a boxed area, set each pixel in the area to 0
# st yxdata(j, i) = 0 for every (i, j) in box
def box_check(yxdata, box, y1, y2, x1, x2):
	for j in range (y1, y2+1):
		for i in range (x1, x2+1):
			yxdata[j][i] = 0
		#set all terms in box to 0
		# this doesn't get rid of them or exclude them from calc
		#  but it does make them extremely unlikely to show up in solution
	return yxdata

#-----------------------------------------------
#now lets try and find our canidates
# call the # of candidates to r1

def assprob(yxdata, max_X, max_Y):
	# now assign a probability to each point
	l = [] #= [prob, (x, y)]
	# will only list points greater than 1 sigma away from mean, this should make program more efficient
	for q in range (0, max_Y + 1):
		for r in range (0, max_X + 1):
			if yxdata[q][r]>=2:
				l.append((fff(yxdata[q][r]), (r, q)))
	m = l[:]
	m.sort(key=None, reverse=False)
	# m has format [probability, (x, y)]
	print "Probability assigned to each point"
	return m

#-----------------------------------------------
#now lets look at each possible box containing each point of the 1000 lowest rank
#if the size of the box is n x n, then there are n*n possible boxes for each point
#create a function to go through a n x n box and get its ave value

# defining function to look at each box of a point, and then give the box with lowest prob
def box_sort(point, prob_sq):
	x = point[0]
	y = point[1]
	for j in range (0, size):
		for i in range (0, size):
			#now for each point check the n x n box
			# where (x-i, y-j) is the upper left corner
			#  and then find the box with lowest prob
			ave_prob = 1
			for r in range (y-j, y-j+size):
				for q in range (x-i, x-i+size):
					if q > max_X or r > max_Y:
						ave_prob = 1 * ave_prob
					elif q < 0 or r < 0:
						ave_prob = 1 * ave_prob
					else:
						#n = r*(max_X+1) + q
						ave_prob = fff(yxdata[r][q]) * ave_prob
			# can alter program to take min prob and use that to represent area instead
			ave_prob1 = []
			ave_prob1.append((ave_prob, (x-i, y-j)))
	ave_prob1 = min(ave_prob1)
	prob_sq.append((ave_prob1))
	return prob_sq


def box_list(m, togo):
	prob_sq = []
	for q in range (0, togo):
		point = m[q][1]
		prob_sq = box_sort(point, prob_sq)
	#now extract prob_sq: [ave_prob, point] but have to remove duplicates:
	#remove duplicates in list
	r = set(prob_sq)
	#make into a list again
	prob_sq = list(r)
	prob_sq.sort(key=None, reverse=False)
	print "Boxes made and the lowest", togo, "selected"
	return prob_sq

#-----------------------------------------------
#look for lines instead of boxes

def line_sort(point, prob_sq):
	x = point[0]
	y = point[1]
	Y = y
	#make sure not a border point
	#check points bellow
	while Y+1 <= max_Y and yxdata[Y+1][x] >= 2:
		Y+=1
	bottom = Y
	Y = y
	#check points above
	while Y-1 >= 0 and yxdata[Y-1][x] >= 2:
		Y-=1
	top = Y
	length = bottom - top +1
	prob = 1
	if length >= 3:
		for i in range (0, length):
			prob = fff(yxdata[top+i][x]) * prob
		stat = 1/float(length)
		aveprob = prob**stat
		prob_sq.append((aveprob, prob, (x, top), length))
		#prob_sq = ave prob, total prob, top point, length
	return prob_sq

def line_list(m, togo, max_Y):
	line_prob = []
	for q in range (0, togo):
		point = m[q][1]
		if point[1]!=0 and point[1]!=max_Y:
			line_prob = line_sort(point, line_prob)
	#now extract prob_sq: [ave_prob, point] but have to remove duplicates:
	#remove duplicates in list
	r = set(line_prob)
	#make into a list again
	line_prob = list(r)
	line_prob.sort(key=None, reverse=False)
	
	print "Lines made and the lowest", togo, "selected"
	return line_prob

#-----------------------------------------------
# now lets display and save our top n or r1 candidates 

def display_data(prob_sq, line_prob, drift, jic, jic_drift):
	final = []
	print "\nThe following are the top Choices with drift rate %r sec/pixel %s \n" % (drift, jic)
	print "Rank\t\tX\t\tY\t\tSize\t\tProbability\n"
	print "Boxes:"
	
	for j in range (0,r1):
		print j+1, "\t\t", prob_sq[j][1][0], "\t\t", prob_sq[j][1][1], "\t\t", size, "\t\t", prob_sq[j][0], "\n"
		final.append((drift, jic_drift, j+1, prob_sq[j][1][0], prob_sq[j][1][1], size, prob_sq[j][0]))
		#final = drift rate, rank, x, y, size, probability 
	
	print "Lines:"
	for i in range (0,r1):
	        print i+1, "\t\t", line_prob[i][2][0], "\t\t", line_prob[i][2][1], "\t\t", line_prob[i][3], "\t\t", line_prob[i][0], "\n"
	        final.append((drift, jic_drift, i+1, line_prob[i][2][0], line_prob[i][2][1], line_prob[i][3], line_prob[i][0]))
	
	print "NOTE: this data set is initialized at (0, 0). DS9 is initialized at (1, 1)."
	print "------------------------------------------------------------\n"
	return final

#-----------------------------------------------

final_data = []
for zzz in range (0, len(image_drift)):
	print "Looking at drift rate", drift[zzz]
	if box != None:
		for xzx in range (0 ,len(yx_files)):
			yx_files[xzx] = box_check(yx_files[xzx], box, y1, y2, x1, x2)
		print "Box Excluded"
	yxdata = sum_images(yx_files, image_drift[zzz], isbinary, mean, stdev)
	m = assprob(yxdata, max_X, max_Y)
	#for large boxes (n>=10), the program will take forever for 1000 possibilites, so it is changed to 100
	togo = 1000
	if size >= 10:
		togo = 100
	if len(m) < togo:
		togo = len(m)
	#NOTE: TO CHECK ALL POINTS > 2 FROM MEAN, MAKE TOGO = LEN(M). UNCOMMENT NEXT LINE
	#togo = len(m)
	if Boxes == True:
		box_prob = box_list(m, togo)
	elif Boxes == False:
		box_prob = []
		for chkin in range (0, r1):
			box_prob.append((0,(0,0)))
	if Lines == True:
		line_prob = line_list(m, togo, max_Y)
	elif Lines == False:
		line_prob = []
		for chkin in range (0, r1):
			line_prob.append((0,0,(0,0),0))
	printdata = display_data(box_prob, line_prob, drift[zzz], jic[zzz], jic_drift[zzz])
	if zzz == 0:
		final_data = printdata
	else:
		final_data += printdata
new_file = text
np.savetxt(new_file, (final_data), fmt=('%-5.2d', '%-6.3f', '%-4i', '%-8i', '%-8i', '%-4i', '%-8.2e'))#, comments='#test')
# fmt--> %(width).(precision)(specifier)
#	width--> max number of digits
#	precision--> number of digits after decimal point
#	specifier--> i=integer, e=scientific notation, f=floating point
#now add the comments on top
file = open(new_file, 'r')
lines = file.read()

if dat == True:
	dfile = dat_name
else:
	dfile= ""

warning = "NOTE: this data set is initialized at (0, 0). DS9 is initialized at (1, 1).\n"
boxsize = "These points are upper-left (lowest value) corners or top points of lines.\n"
more = "The following files are Binary Files with dimensions and times list: %s, %rx%r; %s\n%r\n" % (
	isbinary, yd, xd, dfile, file_list)
new = "DriftRate    Rank X        Y        Size Probability\n"
new = warning + boxsize + more + new + lines
file.close()
file = open(new_file, 'w')
file.write(new)
file.close()


