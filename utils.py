import cv2
import numpy as np
import argparse
from easydict import EasyDict as edict
import yaml
from DragGanGenerator import DragGANGenerator
from collections import defaultdict

# calc pixel distance between two points
def calc_pixel_distance(p1, p2):
    return (p2[0] - p1[0]), (p2[1] - p1[1])

# create rectangular mask 
def create_rectangular_mask(image, point1, point2):
    # Define two points
    y1, x1 = point1
    y2, x2 = point2

    # Create a translucent mask
    image_cp = image.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Using green color for illustration

    # create mask 
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # fill in the rectangle as white
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Define transparency factor, alpha
    alpha = 0.6

    # Blend image and overlay to get the result
    cv2.addWeighted(overlay, 
                    alpha, 
                    image_cp, 
                    1. - alpha, 
                    0.0, 
                    image_cp)
    
    # return both blended image and mask 
    return image_cp, mask

# calculate eye target points 
def calc_eye_target_points(handle_points, 
                           pixel_diff_x, 
                           pixel_increment=1, 
                           max_away_steps=1):

    eye_target_points = []
    # generate target points for drag, change only x coordinate
    pixel_diff_temp = pixel_diff_x
    index = 1
    # since pixel increment is done on both sides, 
    # pixel_diff_temp should be greater than pixel_increment
    while pixel_diff_temp > pixel_increment:
        eye_target_points.append([(handle_points[0][0] + index*pixel_increment, 
                                   handle_points[0][1]),
                                    (handle_points[1][0] - index*pixel_increment, 
                                     handle_points[1][1])])
        pixel_diff_temp -= 2 * pixel_increment
        index += 1

    pixel_diff_temp = pixel_diff_x
    index = 1
    while index <= max_away_steps:
        eye_target_points.append([(handle_points[0][0] - index*pixel_increment, 
                                   handle_points[0][1]),
                                    (handle_points[1][0] + index*pixel_increment, 
                                     handle_points[1][1])])
        pixel_diff_temp += 2 * pixel_increment
        index += 1

    return eye_target_points

def calc_pose_target_points(handle_point, 
                            pixel_increment, 
                            sets_of_points):
    x, y = handle_point
    directions = {
        'left':       (0, -1),   # left 
        'right':      (0, 1),    # right 
        'up':         (-1, 0),   # up 
        'down':       (1, 0),    # down (makes nose longer)
        # 'up_left':    (-1, -1),  # up-left 
        # 'up_right':   (-1, 1),   # up-right 
        # 'down_right': (1, 1),    # down-right
        # 'down_left':  (1, -1),   # down-left  
    }

    pose_target_sets = defaultdict(list)
    for i in range(sets_of_points):
        offset = pixel_increment * (i + 1)
        for key, (dx, dy) in directions.items():
            target_point = (x + dx * offset, y + dy * offset)
            pose_target_sets[key].append(target_point)

    return pose_target_sets


def calc_mouth_target_points(handle_points, 
                             pixel_diff_x, 
                             pixel_increment=1, 
                             max_close_steps=1,
                             max_away_steps=1):
    mouth_target_points = []

    # Generate target points for drag, change only x coordinate
    pixel_diff_temp = pixel_diff_x
    index = 1
    # Move from current pixel difference to -max_pixel_diff (close mouth)
    while (pixel_diff_temp > pixel_increment) and (index <= max_close_steps):
        mouth_target_points.append([(handle_points[0][0] + index*pixel_increment, 
                                     handle_points[0][1]),
                                    (handle_points[1][0] - index*pixel_increment, 
                                     handle_points[1][1])])
        pixel_diff_temp -= 2 * pixel_increment
        index += 1

    pixel_diff_temp = pixel_diff_x
    index = 1
    # Move from current pixel difference (open mouth)
    while index <= max_away_steps:
        mouth_target_points.append([(handle_points[0][0] - index*pixel_increment, 
                                     handle_points[0][1]),
                                    (handle_points[1][0] + index*pixel_increment, 
                                     handle_points[1][1])])
        pixel_diff_temp += 2 * pixel_increment
        index += 1

    return mouth_target_points


def calc_smile_target_points(handle_points, 
                             pixel_increment=1, 
                             max_steps=1):
    smile_target_points = []
    index = 1
    # Move from current pixel difference (unsmile)
    while index <= max_steps:
        smile_target_points.append([(handle_points[0][0] - index*pixel_increment, 
                                     handle_points[0][1] - index*pixel_increment),
                                    (handle_points[1][0] - index*pixel_increment, 
                                     handle_points[1][1] + index*pixel_increment)])
        index += 1

    index = 1
    # Move from current pixel difference  (smile)
    while index <= max_steps:
        smile_target_points.append([(handle_points[0][0] + index*pixel_increment, 
                                     handle_points[0][1] - index*pixel_increment),
                                    (handle_points[1][0] + index*pixel_increment, 
                                     handle_points[1][1] + index*pixel_increment)])
        index += 1

    return smile_target_points


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Experiments for NetGAN"
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="config/0.yaml",
        required=True,
        help="Path of config file")
  
    args = parser.parse_args()

    return args


def get_config(config_file):
  config = edict(yaml.load(open(config_file, 'r'), 
                                Loader=yaml.FullLoader))

  return config


def process_variations(variations_dict, 
                       class_index, 
                       start_index=None, 
                       end_index=None,
                       output_dir=None,
                       batch_identities=None):

    dragan_gen = DragGANGenerator(model_path='./checkpoints/network_conditional_skin_tones.pkl', 
                                  z_seed=None,
                                  iterations=200, 
                                  lr=0.001, early_stop_patience=10,
                                  output_dir='generated_images'if output_dir == None else output_dir)


    if batch_identities is not None:
        # Use the provided list of identities
        identities_to_process = batch_identities
    elif start_index is not None and end_index is not None:
        # Use range of indices
        identities_to_process = range(start_index, end_index)
    else:
        raise ValueError("Either batch_identities or both start_index and end_index must be provided!")


    for seed_no in identities_to_process:

        dragan_gen.cond = class_index
        dragan_gen.w0_seed = seed_no
        dragan_gen.save_image_and_points = False

        dragan_gen.init_and_generate_image()
        
        # get landmark points for outer lips edge 
        outer_lips_handle_points = [dragan_gen.src_points[48],
                                    dragan_gen.src_points[54]]
        
        # get landmark points for inner lips top/bottom 
        inner_lips_handle_points = [dragan_gen.src_points[51],
                                    dragan_gen.src_points[57]]

        left_eye_handle_points = [dragan_gen.src_points[37],
                                  dragan_gen.src_points[41]]

        right_eye_handle_points = [dragan_gen.src_points[44],
                                   dragan_gen.src_points[46]]


        # create a rectangular mask for both the eyes 
        # eye_pixel_offset = 10
        # x_max, y_max, _ = image.shape
        # left_eye_left_point = dragan_gen.src_points[:, 36, :]
        # right_eye_right_point = dragan_gen.src_points[:, 45, :]
        # left_point = (left_eye_handle_points[0][0] - eye_pixel_offset, left_eye_left_point[1] - eye_pixel_offset)
        # right_point = (right_eye_handle_points[1][0] + eye_pixel_offset, right_eye_right_point[1] + eye_pixel_offset)
        # both_eyes_mask, both_eyes_overlay = create_rectangular_mask(image, 
        #                                                             left_point, 
        #                                                             right_point)


        # create a rectangular mask for left eye 
        # left_eye_right_point = dragan_gen.src_points[:, 39, :]
        # left_point = (left_eye_handle_points[0][0] - eye_pixel_offset, left_eye_left_point[1] - eye_pixel_offset)
        # right_point = (left_eye_handle_points[1][0] + eye_pixel_offset, left_eye_right_point[1] + eye_pixel_offset)
        # left_eye_mask, left_eye_overlay = create_rectangular_mask(image,
        #                                                         left_point,
        #                                                         right_point)


        if variations_dict['eyes']['include']:
            max_away_steps = 1
            pixel_increment = 6

            print('left eye handle points : ', left_eye_handle_points)
            left_eye_diff_x, left_eye_diff_y = calc_pixel_distance(left_eye_handle_points[0],
                                                                   left_eye_handle_points[1])
            print('left eye diff x : ', left_eye_diff_x)
            print('left eye diff y : ', left_eye_diff_y)

            print('right eye handle points : ', right_eye_handle_points)
            right_eye_diff_x, right_eye_diff_y = calc_pixel_distance(right_eye_handle_points[0],
                                                                     right_eye_handle_points[1])

            print('right eye diff x : ', right_eye_diff_x)
            print('right eye diff y : ', right_eye_diff_y)

            left_eye_target_points = calc_eye_target_points(left_eye_handle_points, 
                                                            left_eye_diff_x,
                                                            pixel_increment=pixel_increment,
                                                            max_away_steps=max_away_steps)
            print('left eye target points : ', left_eye_target_points)

            right_eye_target_points = calc_eye_target_points(right_eye_handle_points, 
                                                             right_eye_diff_x,
                                                             pixel_increment=pixel_increment,
                                                             max_away_steps=max_away_steps)
            print('right eye target points : ', right_eye_target_points)

            eye_target_points = zip(left_eye_target_points, right_eye_target_points)
            eye_handle_points = zip(left_eye_handle_points, right_eye_handle_points)

            dragan_gen.params['lr'] = 0.001
            dragan_gen.drag_gan_iterations = 200
            for eye_drag_index, (left_eye_target_pair, right_eye_target_pair) in enumerate(eye_target_points):
                for side, target_pair, handle_points in [('left', left_eye_target_pair, left_eye_handle_points),
                                                         ('right', right_eye_target_pair, right_eye_handle_points)]:
                    dragan_gen.init_and_generate_image()
                    dragan_gen.drag_image(handle_points,
                                          target_pair,
                                          f"{variations_dict['eyes']['var_name']}_{side}_{eye_drag_index}",
                                          None, save_w=variations_dict['eyes']['save_w']) 


        if variations_dict['mouth']['include']:
            max_away_steps = 1
            pixel_increment = 6

            print('inner lips handle points : ', inner_lips_handle_points)
            # inner lips top/bottom 
            inner_lips_diff_x, inner_lips_diff_y = calc_pixel_distance(inner_lips_handle_points[0], 
                                                                       inner_lips_handle_points[1])
        
            print('inner lips diff x : ', inner_lips_diff_x)
            print('inner lips diff y : ', inner_lips_diff_y)

            inner_lips_target_points = calc_mouth_target_points(inner_lips_handle_points, 
                                                                inner_lips_diff_x,
                                                                pixel_increment=pixel_increment,
                                                                max_away_steps=max_away_steps)

            print('inner lips target points : ', inner_lips_target_points)

            dragan_gen.params['lr'] = 0.001
            dragan_gen.drag_gan_iterations = 200 
            dragan_gen.early_stop_patience = 30
            for mouth_drag_index, inner_lips_target_pair in enumerate(inner_lips_target_points):

                dragan_gen.init_and_generate_image()

                dragan_gen.drag_image(inner_lips_handle_points,
                                      inner_lips_target_pair, 
                                      f"{variations_dict['mouth']['var_name']}_" + str(mouth_drag_index),
                                      None, save_w=variations_dict['mouth']['save_w']) 
                

        if variations_dict['smile']['include']:
            max_steps = 1
            pixel_increment = 12

            # calculate pixel distance between points outer lips edge 
            # outer_lips_diff_x, outer_lips_diff_y = calc_pixel_distance(outer_lips_handle_points[0], 
            #                                                            outer_lips_handle_points[1])
            
            outer_lips_target_points = calc_smile_target_points(outer_lips_handle_points, 
                                                                pixel_increment=pixel_increment,
                                                                max_steps=max_steps)

            dragan_gen.params['lr'] = 0.01
            dragan_gen.drag_gan_iterations = 200 
            dragan_gen.early_stop_patience = 30
            for smile_index, outer_lips_target_pair in enumerate(outer_lips_target_points):

                dragan_gen.init_and_generate_image()

                dragan_gen.drag_image(outer_lips_handle_points,
                                      outer_lips_target_pair, 
                                      f"{variations_dict['smile']['var_name']}_" + str(smile_index),
                                      None, save_w=variations_dict['smile']['save_w']) 


        if variations_dict['pose']['include']:
            nose_handle_point = dragan_gen.src_points[30]
            sets_of_points = 2
            pixel_increment = 10 
            pose_target_sets = calc_pose_target_points(nose_handle_point, 
                                                       pixel_increment=pixel_increment, 
                                                       sets_of_points=sets_of_points)
            
            dragan_gen.params['lr'] = 0.01
            dragan_gen.drag_gan_iterations = 200 
            dragan_gen.early_stop_patience = 30
            for pose_key, pose_target_values in pose_target_sets.items():
                dragan_gen.init_and_generate_image()

                for pose_i, pose_target in enumerate(pose_target_values):

                    dragan_gen.drag_image([nose_handle_point],
                                          [pose_target], 
                                          f"{variations_dict['pose']['var_name']}_" + f"{pose_key}_" + str(pose_i),
                                          None, 
                                          save_w=variations_dict['pose']['save_w']) 

