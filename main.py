from obj_detect import process_webcam
import Controller.sence as av

process_webcam()

velocity = 0
steer = 0

def velocity_steer(self):
    velocity = av.get_velocity()
    steer = av.get_steering()
    return velocity, steer