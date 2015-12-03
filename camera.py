import picamera
camera = picamera.PiCamera()
camera.start_preview()

from time import sleep
#camera.capture('image.jpg')

while(True):
	a=0

sleep(10)
'''
	Like with the raspistill command, you can apply a horizontal and vertical flip if your camera is positioned 		upside-down. This is done by changing the hflip and vflip properties directly:
'''

camera.start_recording('video.h264')
sleep(5)
camera.stop_recording()
#camera.hflip = True
#camera.vflip = True
camera.stop_preview()

'''
	camera.sharpness = 0
	camera.contrast = 0
	camera.brightness = 50
	camera.saturation = 0
	camera.ISO = 0
	camera.video_stabilization = False
	camera.exposure_compensation = 0
	camera.exposure_mode = 'auto'
	camera.meter_mode = 'average'
	camera.awb_mode = 'auto'
	camera.image_effect = 'none'
	camera.color_effects = None
	camera.rotation = 0
	camera.hflip = False
	camera.vflip = False
	camera.crop = (0.0, 0.0, 1.0, 1.0)


	camera.start_recording('video.h264')
	sleep(5)
	camera.stop_recording()
'''
