import cv2
import dlib
import numpy as np
import socket

from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer
from head_pose_estimation.visual import *

def get_face(detector, image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        box = detector(image)[0]
        x1 = box.left()    
        y1 = box.top()
        x2 = box.right()        
        y2 = box.bottom()        
        return [x1, y1, x2, y2]
    except:
        return None

def main():

    dlib_model_path = 'head_pose_estimation/Asset/shape_predictor_68_face_landmarks.dat'
    shape_predictor = dlib.shape_predictor(dlib_model_path)
    face_detector = dlib.get_frontal_face_detector() 

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _, sample_frame = cap.read()

    pose_estimator = PoseEstimator(img_size=sample_frame.shape[0:2])

    pose_stabilizers = [Stabilizer(
                        state_num=2,
                        measure_num=1,
                        cov_process=0.01,
                        cov_measure=0.1) for _ in range(6)]   

    '''address = ('127.0.0.1', 5066)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)'''

    frame_count = 0

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_count += 1
 
        facebox = get_face(face_detector, frame)

        if facebox is not None: 
            face = dlib.rectangle(left=facebox[0], top=facebox[1], right=facebox[2], bottom=facebox[3])
            shape_ = shape_predictor(frame, face)
            mask = shape_to_np(shape_)

            reprojection_error, rotation_vector, translation_vector = pose_estimator.solve_pose_by_68_points(mask)
            pose = list(rotation_vector) + list(translation_vector)

            if reprojection_error > 100:                             
                pose_estimator = PoseEstimator(img_size=sample_frame.shape[0:2])

            else:
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])

            roll = np.clip(180+np.degrees(steady_pose[2]), -50, 50)    #unity.z  pnp.z
            pitch = np.clip(-(np.degrees(steady_pose[1]))-25, -50, 50)    #unity.x  pnp.y
            yaw = np.clip(np.degrees(steady_pose[0]), -50, 50)         #unity.y  pnp.x

            print (pitch,yaw,roll)

            '''if frame_count > 40: 
                msg = '%.4f %.4f %.4f' % (roll, pitch, yaw)
                s.send(bytes(msg, "utf-8"))'''


            draw_box(frame, [facebox])

            if reprojection_error < 100:
                draw_marks(frame, mask, color=(0, 255, 0))

                pose_estimator.draw_annotation_box(
                    frame, np.expand_dims(steady_pose[:3],0), np.expand_dims(steady_pose[3:6],0), 
                    color=(128, 255, 128))

                pose_estimator.draw_axes(frame, np.expand_dims(steady_pose[:3],0), 
                                             np.expand_dims(steady_pose[3:6],0))

        cv2.imshow("face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    s.close()

if __name__ == '__main__':
    main()
