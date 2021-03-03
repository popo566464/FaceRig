import cv2
import numpy as np

def draw_box(image, boxes, box_color=(255, 255, 255)):
    """Draw square boxes on image"""
    for box in boxes:
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]), box_color, 3)

def draw_marks(image, marks, color=(255, 255, 255)):
    """Draw mark points on image"""
    for mark in marks:
        cv2.circle(image, (int(mark[0]), int(
            mark[1])), 1, color, -1, cv2.LINE_AA)

	
def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords