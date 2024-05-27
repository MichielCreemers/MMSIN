import torch
import open3d as o3d
import os
from PIL import Image
import numpy as np

print(torch.cuda.is_available())
print(torch.__version__)
print("Open3D: ", o3d.__version__)

if torch.cuda.is_available():
    print("Cuda is Availabe")
else:
    print("Cuda Can't be found")
    
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
pc_path = "test_data/soldier.ply"
image_path = "test_data"
pcd = o3d.io.read_point_cloud(pc_path) 
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.light_on = False

ctrl = vis.get_view_control()

ctrl.rotate(180, 0)
vis.poll_events()
vis.update_renderer()


vis.run()

