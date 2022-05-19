import jetson.inference
import jetson.utils


path='/home/mikhailp/ssd-mobilenet/ssd-mobnet'
net = jetson.inference.detectNet(model=path, threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' 
display = jetson.utils.videoOutput("display://0") 

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
