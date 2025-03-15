import cv2
import threading
import base64
import time

from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Brightness,
    Contrast,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop

class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities for USB webcam."""

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 10
        self.recording = False

        self.video_writer = ""

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)

        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)

    def Queue_Sending(self):
        """Callback function for recording flag."""
        self.recordingSender.send(self.recording)
        threading.Timer(1, self.Queue_Sending).start()
        
    # =============================== STOP ================================================
    def stop(self):
        if self.recording:
            self.video_writer.release()
        super(threadCamera, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        if self.brightnessSubscriber.isDataInPipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            # Note: Camera controls may differ between PiCam and USB webcams
            # OpenCV does not directly support adjusting "Brightness" via set_controls(), so you may need to do this manually
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, max(0.0, min(1.0, float(message))))
            
        if self.contrastSubscriber.isDataInPipe():
            message = self.contrastSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            self.camera.set(cv2.CAP_PROP_CONTRAST, max(0.0, min(32.0, float(message))))

        threading.Timer(1, self.Configs).start()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from the USB webcam and sends it to the process gateway."""
        send = True
        while self._running:
            try:
                recordRecv = self.recordSubscriber.receive()
                if recordRecv is not None:
                    self.recording = bool(recordRecv)
                    if recordRecv == False:
                        self.video_writer.release()
                    else:
                        fourcc = cv2.VideoWriter_fourcc(
                            *"XVID"
                        )  # You can choose different codecs, e.g., 'MJPG', 'XVID', 'H264', etc.
                        self.video_writer = cv2.VideoWriter(
                            "output_video" + str(time.time()) + ".avi",
                            fourcc,
                            self.frame_rate,
                            (640, 480),  # Modify this resolution depending on your webcam
                        )

            except Exception as e:
                print(e)

            if send:
                ret, mainRequest = self.camera.read()  # Capture from main webcam
                ret, serialRequest = self.camera.read()  # Capture from low-res webcam (if necessary)

                if self.recording == True:
                    self.video_writer.write(mainRequest)

                # Process the frames (e.g., convert to BGR if needed)
                #mainRequest = cv2.cvtColor(mainRequest, cv2.COLOR_BGR2RGB)
                #serialRequest = cv2.cvtColor(serialRequest, cv2.COLOR_BGR2RGB)

                # Encode both images to JPEG and then to base64
                _, mainEncodedImg = cv2.imencode(".jpg", mainRequest)
                _, serialEncodedImg = cv2.imencode(".jpg", serialRequest)

                mainEncodedImageData = base64.b64encode(mainEncodedImg).decode("utf-8")
                serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")

                self.mainCameraSender.send(mainEncodedImageData)
                self.serialCameraSender.send(serialEncodedImageData)

            send = not send

    # =============================== START ===============================================
    def start(self):
        super(threadCamera, self).start()

    # ================================ INIT CAMERA ========================================
    def _init_camera(self):
        """This function will initialize the webcam object using OpenCV."""
        self.camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)  # OpenCV will use the first available webcam
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

        # Optionally, you can set the camera properties (e.g., resolution, brightness, contrast)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width (adjust as needed)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height (adjust as needed)
        self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)  # Set frame rate (optional)
        

        # The webcam only captures a single stream, so we don't need to configure separate channels like with PiCam