# ----------------------------------------------------------------------------

# -                        Open3D: www.open3d.org                            -

# ----------------------------------------------------------------------------

# Copyright (c) 2018-2023 www.open3d.org

# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------
import time
import os
import glob
import torch
import joblib
import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os.path
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import projections
from utils.NSS import feature_extract, nss_functions, feature_functions
from models.main_model import MM_NSSInet


basedir = os.path.dirname(os.path.realpath(__file__))



class ExampleWindow:

    MENU_CHECKABLE = 1

    MENU_DISABLED = 2

    MENU_QUIT = 3


    def __init__(self):

        self.window = gui.Application.instance.create_window("Test", 1100, 500)

        w = self.window  # for more concise code


        # Rather than specifying sizes in pixels, which may vary in size based

        # on the monitor, especially on macOS which has 220 dpi monitors, use

        # the em-size. This way sizings will be proportional to the font size,

        # which will create a more visually consistent size across platforms.

        em = w.theme.font_size
        gui_layout = gui.Vert(0, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))
        gui_layout.frame = gui.Rect(w.content_rect.x, w.content_rect.y, 600, w.content_rect.height)


        # Create a file-chooser widget. One part will be a text edit widget for

        # the filename and clicking on the button will let the user choose using

        # the file dialog.

        self._pcfileedit = gui.TextEdit()

        pcfilebutton = gui.Button("...")
        pcfilebutton.horizontal_padding_em = 0.5
        pcfilebutton.vertical_padding_em = 0
        pcfilebutton.set_on_clicked(self._on_pc_button)


        # (Create the horizontal widget for the row. This will make sure the

        # text editor takes up as much space as it can.)

        pcfileedit_layout = gui.Horiz(0, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))
        pcfileedit_layout.add_child(gui.Label("Select point cloud file"))
        pcfileedit_layout.add_child(self._pcfileedit)
        pcfileedit_layout.add_fixed(2 * em)
        pcfileedit_layout.add_child(pcfilebutton)

        gui_layout.add_child(pcfileedit_layout)


        # add file selector for the model

        self._modelfileedit = gui.TextEdit()
        modelfilebutton = gui.Button("...")
        modelfilebutton.horizontal_padding_em = 0.5
        modelfilebutton.vertical_padding_em = 0
        modelfilebutton.set_on_clicked(self._on_model_button)



        # text editor takes up as much space as it can.)

        modelfileedit_layout = gui.Horiz(0, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))
        modelfileedit_layout.add_child(gui.Label("Select point cloud file"))
        modelfileedit_layout.add_child(self._modelfileedit)
        modelfileedit_layout.add_fixed(2 * em)
        modelfileedit_layout.add_child(modelfilebutton)

        gui_layout.add_child(modelfileedit_layout)
            
        # number editors for the x and y projections

        #x buttons
        x_proj = gui.NumberEdit(gui.NumberEdit.INT)
        x_proj.int_value = 1
        x_proj.set_limits(1, 100)  # value coerced to 1

        #y buttons
        y_proj = gui.NumberEdit(gui.NumberEdit.INT)
        y_proj.set_limits(1,100)
        y_proj.int_value = 1

        #create gui_layout
        projlayout = gui.Horiz(0, gui.Margins(1 * em, 1 * em, 1 * em,
                                        1 * em))
        projlayout.add_child(gui.Label("Number of x-projections"))
        projlayout.add_child(x_proj)
        projlayout.add_fixed(2 * em)
        projlayout.add_child(gui.Label("Number of y-projections"))
        projlayout.add_child(y_proj)

        gui_layout.add_child(projlayout)


        # Create a progress bar.

        #create bar and init
        self._progress = gui.ProgressBar()
        self._progress.value = 0.1

        #create lay out
        prog_layout = gui.Horiz(20, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))

        self._progress_label = gui.Label("Add point cloud")

        prog_layout.preferred_height = 50
        prog_layout.add_child(self._progress_label)
        prog_layout.add_child(self._progress)

        gui_layout.add_child(prog_layout)

        #create confirmation button
        button_layout = gui.Horiz()

        ok_button = gui.Button("Calculate_quality")
        ok_button.set_on_clicked(self._on_ok)
        button_layout.add_fixed(250)
        button_layout.add_child(ok_button)

        gui_layout.add_child(button_layout)

        self.pc_renderer = gui.SceneWidget()
        self.pc_renderer.scene = rendering.Open3DScene(w.renderer)
        self.pc_renderer.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.pc_renderer.frame = gui.Rect(600, w.content_rect.y,
                                        500, w.content_rect.height)


       
      

        w.add_child(gui_layout)
        w.add_child(self.pc_renderer)

    def _on_mouse_widget3d(self, event):
            print(event.type)
            return gui.Widget.EventCallbackResult.IGNORED

    def _on_pc_button(self):

        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",

                                 self.window.theme)

        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")

        filedlg.add_filter("", "All files")

        filedlg.set_on_cancel(self._on_pc_filedlg_cancel)

        filedlg.set_on_done(self._on_pc_filedlg_done)

        self.window.show_dialog(filedlg)

    def _on_model_button(self):
        
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",

                                 self.window.theme)

        filedlg.add_filter(".pth", "Trained Model (.pth)")

        filedlg.add_filter("", "All files")

        filedlg.set_on_cancel(self._on_model_filedlg_cancel)

        filedlg.set_on_done(self._on_model_filedlg_done)

        self.window.show_dialog(filedlg)

    def _on_pc_filedlg_cancel(self):

        self.window.close_dialog()

    def _on_model_filedlg_cancel(self):

        self.window.close_dialog()

    def _on_pc_filedlg_done(self, path):

        self._pcfileedit.text_value = path

        self.window.close_dialog()

        self._progress_label.text = "Add trained model to assess the quality"

        self._load_point_cloud_to_renderer(path)

    def _on_model_filedlg_done(self, path):

        self._modelfileedit.text_value = path

        self.window.close_dialog()

        self._progress_label.text = "Set numebr of projections and press OK"

    def _load_point_cloud_to_renderer(self, path):
         
        pc = o3d.io.read_point_cloud(path)
        
        center = np.mean(np.asarray(pc.points), axis=0)
    
        camera_position = center + np.array([0, 0, 1.5*center[1]]) # Adjust the z-coordinate as needed

        up = np.array([0, 5, 0]) # Up direction
        self.pc_renderer.scene.camera.look_at(center, camera_position, up)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        self.pc_renderer.scene.clear_geometry()
        self.pc_renderer.scene.add_geometry('pc', pc, material)

    def _assess_quality(self):

        projections_folder = "test"
        # projections.make_projections(self._pcfileedit.text_value,"test_projections",4, 4, 4, 'default', False)
        images = glob.glob(f'{projections_folder}/*.png')

        transformation = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
        transformed_imgs = torch.zeros([len(images), 3, 224, 224])

        for i in range(len(images)):
            read_image = Image.open(images[i])
            read_image.convert('RGB')
            read_image = transformation(read_image)
            transformed_imgs[i] = read_image

        

        nss_features = feature_extract.get_feature_vector(self._pcfileedit.text_value)

        feature_names = ["l_mean","l_std","l_entropy","a_mean","a_std","a_entropy","b_mean","b_std","b_entropy","curvature_mean","curvature_std","curvature_entropy","curvature_ggd1","curvature_ggd2","curvature_aggd1","curvature_aggd2","curvature_aggd3","curvature_aggd4","curvature_gamma1","curvature_gamma2","anisotropy_mean","anisotropy_std","anisotropy_entropy","anisotropy_ggd1","anisotropy_ggd2","anisotropy_aggd1","anisotropy_aggd2","anisotropy_aggd3","anisotropy_aggd4","anisotropy_gamma1","anisotropy_gamma2","linearity_mean","linearity_std","linearity_entropy","linearity_ggd1","linearity_ggd2","linearity_aggd1","linearity_aggd2","linearity_aggd3","linearity_aggd4","linearity_gamma1","linearity_gamma2","planarity_mean","planarity_std","planarity_entropy","planarity_ggd1","planarity_ggd2","planarity_aggd1","planarity_aggd2","planarity_aggd3","planarity_aggd4","planarity_gamma1","planarity_gamma2","sphericity_mean","sphericity_std","sphericity_entropy","sphericity_ggd1","sphericity_ggd2","sphericity_aggd1","sphericity_aggd2","sphericity_aggd3","sphericity_aggd4","sphericity_gamma1","sphericity_gamma2"]
        
        features_df = pd.DataFrame([nss_features], columns=feature_names)

        # scaler = joblib.load('utils/NSS/sc.joblib')

        # nss_features = scaler.transform(features_df)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MM_NSSInet()
        model.load_state_dict(torch.load(self._modelfileedit.text_value))

        model = model.to(device)
        
        model.eval()

        print('Begin inference.')
        with torch.no_grad():
            transformed_imgs = transformed_imgs.to(device).unsqueeze(0)   

            # nss_features_values = nss_features.astype(float).val
            nss_features_tensor = torch.tensor(nss_features, dtype=torch.float).squeeze() 
            nss_features_tensor = nss_features_tensor.to(device).unsqueeze(0)
            outputs = model(transformed_imgs,nss_features_tensor)
            score = outputs.item()
        
        print('Predicted quality score: ' + str(score))


    def _on_ok(self):
        self._assess_quality()






# This class is essentially the same as window.show_message_box(),

# so for something this simple just use that, but it illustrates making a

# dialog.





def main():

    # We need to initialize the application, which finds the necessary shaders for

    # rendering and prepares the cross-platform window abstraction.

    gui.Application.instance.initialize()


    w = ExampleWindow()


    # Run the event loop. This will not return until the last window is closed.

    gui.Application.instance.run()



if __name__ == "__main__":

    main()