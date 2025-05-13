import cv2
import numpy as np
import mediapipe as mp

# 3D model points for solvePnP (in millimetres)
MODEL_POINTS = np.array([
    (0.0,   0.0,     0.0),     # Nose tip
    (0.0,  -330.0,  -65.0),    # Chin
    (-165.0, 170.0, -135.0),   # Left eye outer corner
    (165.0,  170.0, -135.0),   # Right eye outer corner
    (-150.0,-150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float32)

# Corresponding MediaPipe landmark indices
IMAGE_POINTS_IDX = [1, 152, 33, 263, 61, 291]


def get_euler_angles(rvec, tvec):
    """
    Convert rotation vector and translation vector to Euler angles (roll, pitch, yaw) in degrees.
    """
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)


def eye_aspect_ratio(landmarks, idx_top, idx_bottom, idx_left, idx_right, w, h):
    """
    Compute Eye Aspect Ratio (EAR) from MediaPipe landmarks.
    """
    def _pt(i):
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h])

    top    = _pt(idx_top)
    bottom = _pt(idx_bottom)
    left   = _pt(idx_left)
    right  = _pt(idx_right)
    vert   = np.linalg.norm(top - bottom)
    hor    = np.linalg.norm(left - right)
    return vert / hor if hor > 0 else 0


def process_image(img_path):
    """
    Reads an image, runs MediaPipe FaceMesh, solves PnP, and returns head pose + EAR metrics.

    Returns a dict: {'roll', 'pitch', 'yaw', 'left_ear', 'right_ear'} or None if detection fails.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Initialize MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mp_face.close()
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    image_points = np.array([[lm[i].x * w, lm[i].y * h] for i in IMAGE_POINTS_IDX], dtype=np.float32)

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, dist_coeffs)
    if not ok:
        return None

    roll, pitch, yaw = get_euler_angles(rvec, tvec)
    left_ear  = eye_aspect_ratio(lm, 159, 145, 33, 133, w, h)
    right_ear = eye_aspect_ratio(lm, 386, 374, 263, 362, w, h)

    return {
        'roll':  roll,
        'pitch': pitch,
        'yaw':   yaw,
        'left_ear':  left_ear,
        'right_ear': right_ear
    }

