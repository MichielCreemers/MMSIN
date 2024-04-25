import numpy as np
import open3d as o3d
import os
from PIL import Image
import cv2 as cv
import argparse
import time


def background_crop(image):
    """When projecting a 3D object to a 2D image, the background of the image that is due to the projection
    is pure white (value of 255 in grayscale). By converting the image to grayscale, the boundaries of the object
    can be detected. 
    * If the rows and columns sum to less than 255 * number of elements, a row/column contains the object. 
    * This function finds these rows and columns and crops the image to the bounding box of the object.

    Args:
        image (opencv loaded image): image to crop
    """
    # Convert the image to grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Find where the rows and columns sum to less than 255 * number of elements (not pure white)
    non_white_cols = np.where(np.mean(gray_img, axis=0) < 255)[0]
    non_white_rows = np.where(np.mean(gray_img, axis=1) < 255)[0]
    
    # Determine the bounding box
    col_start, col_end = non_white_cols[0], non_white_cols[-1]
    row_start, row_end = non_white_rows[0], non_white_rows[-1]
    
    # Crop the image
    cropped_image = image[row_start:row_end+1, col_start:col_end+1]

    aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
    
    # Ensure the cropped image is at least 224x224 in size while maintaining the aspect ratio
    if cropped_image.shape[0] < 224 or cropped_image.shape[1] < 224:
        if cropped_image.shape[0] < 224:
            new_height = 224
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = 224
            new_height = int(new_width / aspect_ratio)
        cropped_image = cv.resize(cropped_image, (new_width, new_height))

    return cropped_image

def create_image_dir(image_path):
    """This function creates the directory for the images if it does not exist.

    Args:
        image_path (str): Path to the directory where the images will be saved.
    """
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)

def calculate_rotation_strategy(number_y: int, number_x: int, predefined_strategy='default'):
    """This function calculates the rotation strategy of the camera around the object for the 2D projections.
    * In the default configuration, the camera first rotates around the y-axis until a full 360° rotation
    is completed, and then around the x-axis for a full rotation. The angular step size is automatically calculated.
    * There are also predefined strategies for the rotation of the camera around the object. One can choose between:
    1. 'default': The default strategy, just using number_y and number_x.
    2. '3_stage_x_4_60': 
    3. 
    4. 
    0. ... Feel free to implement more.

    Args:
        number_y (int): Number of images to be taken around the y-axis.
        number_x (int): Number of images to be taken around the x-axis.
        predefined_strategy (str, optional): See discription for predefined strategies. Defaults to 'default'.
    
    Returns:
        strategy (tuple): ([list of tuples (0,1), (1,0) or (1,1) indicating the type of rotation], x angle step, y angle step)
    """
    sequence = []
    # The rotation functions works as follows: rotate(x,y) where x is the distance the mouse is moved in the x direction and y in the y direction.
    # Moving the mouse horizontally or vertically a distance of 5.82 is equivalent to a rotation of 1° of their respective axis.
    distance_1_degree = 5.82
    
    if predefined_strategy != 'default':
        #Implementation of predefined strategies
        pass
    else:
        # Default strategy
        x_angle_step = 360 / number_x
        y_angle_step = 360 / number_y if number_y != 0 else 0
        for i in range(number_x):
            if i == 0:
                sequence.append((0, 0))
            else:
                sequence.append((x_angle_step * distance_1_degree, 0))
        for i in range(number_y):
            sequence.append((0, y_angle_step * distance_1_degree))
        return (sequence, x_angle_step, y_angle_step)

def rotate_and_capture_images(pcd, image_path: str, strategy: tuple[list, float, float], Nx, Ny, ps, visualise=False, y_start_from_top=True) -> None:
    """_summary_

    Args:
        pcd (_type_): _description_
        image_path (str): _description_
        strategy (tuple[list, float, float]): _description_
        visualise (bool, optional): _description_. Defaults to False.
    """
    sequence, x_angle_step, y_angle_step = strategy
    print("got here")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='EXTRACTOR', visible=visualise)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.light_on = False            # In this case, the visualiser added a light source which is not wanted.
    opt.point_size = ps
    
    ctrl = vis.get_view_control()
    current_x_angle = 0
    current_y_angle = 0   
    
    #print("sequence: ", sequence)
    projection_nr = 0
    
    for angle_index, (delta_x, delta_y) in enumerate(sequence):
        
        
        vis.reset_view_point(True)
        
        if angle_index < Nx:
            current_x_angle += delta_x
        else:
            current_x_angle = 0
        
        if y_start_from_top and angle_index == Nx:
            current_y_angle -= 90 * 5.82
        current_y_angle -= delta_y

        print(f"Current x angle: {current_x_angle}, Current y angle: {current_y_angle}")

        
        ctrl.rotate(current_x_angle, current_y_angle)
        vis.poll_events()
        vis.update_renderer()
        if(round(current_y_angle,1) != -1047.6 and round(current_y_angle,1) != -2095.2):
            projection = vis.capture_screen_float_buffer(True) # Image stored as floating point array with values between 0 and 1
            projection = (np.asarray(projection) * 255).astype(np.uint8) # Cast to 8bit format 
            projection = Image.fromarray(projection)
            projection = cv.cvtColor(np.asarray(projection), cv.COLOR_BGR2RGB) # OpenCV uses BGR format instead of RGB
            projection = background_crop(projection)
            cv.imwrite(os.path.join(image_path, f"projection_{projection_nr}.png"), projection)
            projection_nr += 1
    
    vis.destroy_window()
            
    
    return 1
        
def make_projections(pc_path: str, image_path: str, x_projections: int, y_projections: int, point_size: int, predefined_strategy='default', visualise=False):
    """_summary_

    Args:
        pc_path (str): Full path to the point cloud file.
        image_path (str): Full path to the directory where the images will be saved.
        x_projections (int): The number of images to be taken around the x-axis.
        y_projections (int): The number of images to be taken around the y-axis.
        predefined_strategy (str, optional): If you want a predefined rotation strategy. Defaults to 'default'.
        visualise (bool, optional): Wheather to visualise the 3D object projections/rotations. Defaults to False.
    """
    # Calculate the rotation strategy
    strategy = calculate_rotation_strategy(y_projections, x_projections, predefined_strategy)
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pc_path) 
    
    # Ensure/create the image directory
    create_image_dir(image_path) 
    
    # Actual rotation and projection
    start = time.time()
    rotate_and_capture_images(pcd, image_path, strategy, visualise=visualise, Nx=x_projections, Ny=y_projections , ps=point_size)
    end = time.time()
    
    print(f"Projections are completed in {end-start} seconds.")  
        
    
    
# pc_path = "test_data/soldier.ply"
# image_path = "test_data/soldier_projections"

# make_projections(pc_path, image_path, 4, 2, visualise=False)

if __name__ == '__main__':
    print("***********************************************************")
    parser = argparse.ArgumentParser(description='Make different projections from multiple angles of a point cloud and store these images.')
    parser.add_argument('--pc_path', type=str, help='Full path to the point cloud file.')
    parser.add_argument('--image_path', type=str, help='Full path to the directory where the images will be saved.')
    parser.add_argument('--x_projections', type=int, help='The number of images to be taken around the x-axis.')
    parser.add_argument('--y_projections', type=int, help='The number of images to be taken around the y-axis.')
    parser.add_argument('--point_size', type=int, help='The size of the point in the projection')
    args = parser.parse_args()
    make_projections(args.pc_path, args.image_path, args.x_projections, args.y_projections, args.point_size)
    print("***********************************************************")
   