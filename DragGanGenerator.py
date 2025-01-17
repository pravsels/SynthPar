import os
import ast
from tqdm import tqdm
from draggan import DragGAN
import dlib 
import cv2 
import numpy as np 
from PIL import Image, ImageDraw

class AttrDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"No such attribute: {attr}")

    def __setattr__(self, attr, value):
        self[attr] = value

class_to_race_map = {
    0: 'ST1',
    1: 'ST2',
    2: 'ST3',
    3: 'ST4',
    4: 'ST5',
    5: 'ST6',
    6: 'ST7',
    7: 'ST8',
}

def visualize_points(points, image, color=(0, 0, 255)):
    if isinstance(image, Image.Image):
        draw = ImageDraw.Draw(image)
        for point in points:
            x, y = point
            draw.ellipse((x - 2, 
                          y - 2, 
                          x + 2, 
                          y + 2), fill=color)
    else:
        for point in points:
            cv2.circle(image, point, 1, color, 2)
    return image

def reverse_point_pairs(points):
    return [[p[1], p[0]] for p in points]

class DragGANGenerator:
    def __init__(self, model_path='', w0_seed=0, 
                       cond=2, z_seed=None, 
                       w_load=None, 
                       iterations=100, lr=0.001, 
                       early_stop_threshold=1e-5, 
                       early_stop_patience=10,
                       output_dir='generated_images'):
        self.output_dir = output_dir
        self.model_path = model_path
        self._w0_seed = w0_seed
        self._cond = cond
        self._z_seed = z_seed
        self._w_load = w_load
        self.output_subdir = class_to_race_map[cond]
        self.identity_dir = f'cond{str(self._cond)}_seed{str(self._w0_seed)}'
        self.drag_gan_iterations = iterations
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.params = {
            "seed": self._w0_seed,
            "cond": self._cond,
            "z_seed": self._z_seed,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": lr,
        }
        self.drag_gan = DragGAN()
        self.res = AttrDict()

        # create results folder 
        os.makedirs(self.output_dir, exist_ok=True)

        # create dragged dir 
        os.makedirs(f'{self.output_dir}/{self.output_subdir}', exist_ok=True)

        self.save_image_and_points = False 

        # Load dlib's facial landmarks detector model
        predictor_path = "./checkpoints/shape_predictor_68_face_landmarks.dat"  # Path to dlib's facial landmarks data file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    @property
    def cond(self):
        return self._cond
    @cond.setter
    def cond(self, value):
        self._cond = value
        # Recalculate output_subdir and identity_dir when cond is set
        self.output_subdir = class_to_race_map[self._cond]
        self.identity_dir = f'cond{str(self._cond)}_seed{str(self._w0_seed)}'
    @property
    def w0_seed(self):
        return self._w0_seed
    @w0_seed.setter
    def w0_seed(self, value):
        self._w0_seed = value
        # Recalculate identity_dir when cond is set
        self.identity_dir = f'cond{str(self._cond)}_seed{str(self._w0_seed)}'
    @property
    def z_seed(self):
        return self._z_seed
    @z_seed.setter
    def z_seed(self, value):
        self._z_seed = value
        # Reset z_seed for params
        self.params["z_seed"] = self._z_seed


    def init_and_generate_image(self):
        print('initializing and generating image...')
        # load model
        self.drag_gan.init_network(self.res, self.model_path, self._w0_seed, self._cond, self._z_seed, self._w_load)
        # generate image 
        self.drag_gan._render_drag_impl(self.res, is_drag=False, to_pil=True)

        if not self.save_image_and_points:
            # create dragged dir 
            os.makedirs(f'{self.output_dir}/{self.output_subdir}/{self.identity_dir}', exist_ok=True)

            # save image in results folder
            self.res.image.save(f'{self.output_dir}/{self.output_subdir}/{self.identity_dir}/original.png')

            # read from file and into points
            self.src_points = []

            landmarks_path = f'{self.output_dir}/{self.output_subdir}/{self.identity_dir}/landmarks.txt'

            if (not os.path.exists(landmarks_path)) or (os.path.getsize(landmarks_path) == 0):
                self.calculate_face_landmarks(landmarks_path)
    
            with open(landmarks_path, 'r') as f:
                for line in f:
                    point = ast.literal_eval(line.strip())
                    self.src_points.append(point)

            self.src_points = reverse_point_pairs(self.src_points)

            self.save_image_and_points = True

    def get_image(self):
        return self.res.image.copy()
    
    def set_image(self, image):
        self.res.image = image.copy()

    def calculate_face_landmarks(self, landmarks_path):
        # Convert PIL image to grayscale for dlib
        image = cv2.cvtColor(np.array(self.res.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = self.detector(gray)
        shape = None 
        for rect in faces:
            # Determine the facial landmarks for the face region
            shape = self.predictor(gray, rect)
            # Loop over the facial landmarks and draw them on the image
            for i in range(0, 68):
                cv2.circle(image, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1)

        points = []
        for i in range(0, 68):
            points.append((shape.part(i).x, shape.part(i).y))

        # save points to a file
        with open(landmarks_path, 'w') as f:
            for point in points:
                f.write("%s\n" % str(point))

    def drag_image(self, handle_points, target_points, expression_name, mask, 
                   save_w=False, viz=False):
        # run draggan for a few iterations
        print(f'running draggan for {expression_name}...')

        iter = 0
        best_loss = float('inf')
        no_improvement_count = 0

        src_points = handle_points.copy()

        if viz: 
            os.makedirs(f"{self.output_dir}/{self.output_subdir}/{self.identity_dir}/{expression_name}", 
                    exist_ok=True)

        for i in tqdm(range(self.drag_gan_iterations)):

            self.drag_gan._render_drag_impl(
                self.res,
                handle_points,  # point
                target_points,  # target
                mask,  # mask,
                self.params['motion_lambda'],  # lambda_mask
                reg=0.,
                facenet_reg=5.,
                feature_idx=5,  # NOTE: do not support change for now
                r1=self.params['r1_in_pixels'],  # r1
                r2=self.params['r2_in_pixels'],  # r2
                # random_seed     = 0,
                # noise_mode      = 'const',
                trunc_psi=self.params['trunc_psi'],
                # force_fp32      = False,
                # layer_name      = None,
                # sel_channels    = 3,
                # base_channel    = 0,
                # img_scale_db    = 0,
                # img_normalize   = False,
                # untransform     = False,
                is_drag=True,
                to_pil=True)
            
            iter+=1 

            # if self.res.loss < best_loss - self.early_stop_threshold:
            #     best_loss = self.res.loss
            #     no_improvement_count = 0
            # else:
            #     no_improvement_count += 1 

            # if no_improvement_count >= self.early_stop_patience:
            #     print(f'Early stopping at step {iter} with loss {self.res.loss:.6f}')
            #     break 

            if self.res.loss < 1e-5:
                print('Stopping because loss is close to 0')
                break 
        
            if viz: 
                # handle points viz 
                viz_image = visualize_points(reverse_point_pairs(src_points), 
                                            self.res.image.copy(), 
                                            color=(255, 0, 0))
                
                # target points viz 
                viz_image = visualize_points(reverse_point_pairs(target_points), 
                                            viz_image, 
                                            color=(0, 255, 0))

                # save image in results folder
                viz_image.save(f"{self.output_dir}/{self.output_subdir}/{self.identity_dir}/{expression_name}/{expression_name}_lr_{self.params['lr']}_steps_{iter}_loss{round(self.res.loss, 4)}.png")
        

        self.res.image.save(f'{self.output_dir}/{self.output_subdir}/{self.identity_dir}/{expression_name}_dragged.png')

        if save_w: 
            # save w code in the same folder
            np.save(f"{self.output_dir}/{self.output_subdir}/{self.identity_dir}/{expression_name}_dragged_w.npy", 
                    self.drag_gan.w.detach().cpu().numpy())

