# ----------------------------------------------------------------------------

# -                        Open3D: www.open3d.org                            -

# ----------------------------------------------------------------------------

# Copyright (c) 2018-2023 www.open3d.org

# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os.path
import random


basedir = os.path.dirname(os.path.realpath(__file__))



class ExampleWindow:

    MENU_CHECKABLE = 1

    MENU_DISABLED = 2

    MENU_QUIT = 3


    def __init__(self):

        self.window = gui.Application.instance.create_window("Test", 600, 868)

        w = self.window  # for more concise code


        # Rather than specifying sizes in pixels, which may vary in size based

        # on the monitor, especially on macOS which has 220 dpi monitors, use

        # the em-size. This way sizings will be proportional to the font size,

        # which will create a more visually consistent size across platforms.

        em = w.theme.font_size

        layout = gui.Vert(0, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))



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

        # add to the top-level (vertical) layout

        layout.add_child(pcfileedit_layout)


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

        # add to the top-level (vertical) layout

        layout.add_child(modelfileedit_layout)

             


        

        


        # number editors for the x and y projections

        #x buttons
        x_proj = gui.NumberEdit(gui.NumberEdit.INT)
        x_proj.int_value = 1
        x_proj.set_limits(1, 100)  # value coerced to 1

        #y buttons
        y_proj = gui.NumberEdit(gui.NumberEdit.INT)
        y_proj.set_limits(1,100)
        y_proj.int_value = 1

        #create layout
        projlayout = gui.Horiz(0, gui.Margins(1 * em, 1 * em, 1 * em,

                                        1 * em))

        projlayout.add_child(gui.Label("Number of x-projections"))
        projlayout.add_child(x_proj)
        projlayout.add_fixed(2 * em)
        projlayout.add_child(gui.Label("Number of y-projections"))
        projlayout.add_child(y_proj)

        layout.add_child(projlayout)


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

        layout.add_child(prog_layout)


        #add point cloud visualiser

        self.scene = gui.SceneWidget()

        self.scene.scene = rendering.Open3DScene(self.window.renderer)
       
        self.scene.scene.set_background([1, 1, 1, 1])
        
        self.scene.scene.scene.set_sun_light(

            [-1, -1, -1],  # direction

            [1, 1, 1],  # color

            100000)  # intensity

        self.scene.scene.scene.enable_sun_light(True)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],

                                                   [10, 10, 10])

        self.scene.setup_camera(60, bbox, [0, 0, 0])
        self._id =1

        self.scene.frame = gui.Rect(500, w.content_rect.y,
                                        900, w.content_rect.height)

        layout.add_child(self.scene)

        button_layout = gui.Horiz()

        ok_button = gui.Button("Ok")

        ok_button.set_on_clicked(self._on_ok)

        button_layout.add_fixed((self.window.size.width - 60)/2)
        button_layout.add_child(ok_button)


        layout.add_child(button_layout)




        w.add_child(layout)



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

    def _on_model_filedlg_done(self, path):

        self._modelfileedit.text_value = path

        self.window.close_dialog()

        self._progress_label.text = "Set numebr of projections and press OK"





    def _on_ok(self):

        gui.Application.instance.quit()





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