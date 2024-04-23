# ----------------------------------------------------------------------------

# -                        Open3D: www.open3d.org                            -

# ----------------------------------------------------------------------------

# Copyright (c) 2018-2023 www.open3d.org

# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------


import open3d.visualization.gui as gui

import os.path


basedir = os.path.dirname(os.path.realpath(__file__))



class ExampleWindow:

    MENU_CHECKABLE = 1

    MENU_DISABLED = 2

    MENU_QUIT = 3


    def __init__(self):

        self.window = gui.Application.instance.create_window("Test", 400, 768)

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

        pcfileedit_layout = gui.Horiz()

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

        modelfileedit_layout = gui.Horiz()

        modelfileedit_layout.add_child(gui.Label("Select point cloud file"))

        modelfileedit_layout.add_child(self._modelfileedit)

        modelfileedit_layout.add_fixed(2 * em)

        modelfileedit_layout.add_child(modelfilebutton)

        # add to the top-level (vertical) layout

        layout.add_child(modelfileedit_layout)

        # Create a collapsible vertical widget, which takes up enough vertical

        # space for all its children when open, but only enough for text when

        # closed. This is useful for property pages, so the user can hide sets

        # of properties they rarely use. All layouts take a spacing parameter,

        # which is the spacinging between items in the widget, and a margins

        # parameter, which specifies the spacing of the left, top, right,

        # bottom margins. (This acts like the 'padding' property in CSS.)

        collapse = gui.CollapsableVert("Widgets", 0.33 * em,

                                       gui.Margins(em, 0, 0, 0))

        self._label = gui.Label("Lorem ipsum dolor")

        self._label.text_color = gui.Color(1.0, 0.5, 0.0)

        collapse.add_child(self._label)


        




       


        

        


        # Add two number editors, one for integers and one for floating point

        # Number editor can clamp numbers to a range, although this is more

        # useful for integers than for floating point.

        x_proj = gui.NumberEdit(gui.NumberEdit.INT)

        x_proj.int_value = 0

        x_proj.set_limits(1, 100)  # value coerced to 1


        y_proj = gui.NumberEdit(gui.NumberEdit.INT)

        projlayout = gui.Horiz()

        projlayout.add_child(gui.Label("int"))

        projlayout.add_child(x_proj)

        projlayout.add_fixed(em)  # manual spacing (could set it in Horiz() ctor)

        projlayout.add_child(gui.Label("double"))

        projlayout.add_child(y_proj)

        layout.add_child(projlayout)


        # Create a progress bar. It ranges from 0.0 to 1.0.

        self._progress = gui.ProgressBar()

        self._progress.value = 0

        self._progress.value = self._progress.value + 0.08  # 0.25 + 0.08 = 33%

        prog_layout = gui.Horiz(em)

        prog_layout.add_child(gui.Label("Progress..."))

        prog_layout.add_child(self._progress)

        layout.add_child(prog_layout)


        # Create a slider. It acts very similar to NumberEdit except that the

        # user moves a slider and cannot type the number.

        slider = gui.Slider(gui.Slider.INT)

        slider.set_limits(5, 13)

        slider.set_on_value_changed(self._on_slider)

        collapse.add_child(slider)


        # Create a text editor. The placeholder text (if not empty) will be

        # displayed when there is no text, as concise help, or visible tooltip.

        tedit = gui.TextEdit()

        tedit.placeholder_text = "Edit me some text here"


        # on_text_changed fires whenever the user changes the text (but not if

        # the text_value property is assigned to).

        tedit.set_on_text_changed(self._on_text_changed)


        # on_value_changed fires whenever the user signals that they are finished

        # editing the text, either by pressing return or by clicking outside of

        # the text editor, thus losing text focus.

        tedit.set_on_value_changed(self._on_value_changed)

        collapse.add_child(tedit)


        # Create a widget for showing/editing a 3D vector

        vedit = gui.VectorEdit()

        vedit.vector_value = [1, 2, 3]

        vedit.set_on_value_changed(self._on_vedit)

        collapse.add_child(vedit)


        # Create a VGrid layout. This layout specifies the number of columns

        # (two, in this case), and will place the first child in the first

        # column, the second in the second, the third in the first, the fourth

        # in the second, etc.

        # So:

        #      2 cols             3 cols                  4 cols

        #   |  1  |  2  |   |  1  |  2  |  3  |   |  1  |  2  |  3  |  4  |

        #   |  3  |  4  |   |  4  |  5  |  6  |   |  5  |  6  |  7  |  8  |

        #   |  5  |  6  |   |  7  |  8  |  9  |   |  9  | 10  | 11  | 12  |

        #   |    ...    |   |       ...       |   |         ...           |

        vgrid = gui.VGrid(2)

        vgrid.add_child(gui.Label("Trees"))

        vgrid.add_child(gui.Label("12 items"))

        vgrid.add_child(gui.Label("People"))

        vgrid.add_child(gui.Label("2 (93% certainty)"))

        vgrid.add_child(gui.Label("Cars"))

        vgrid.add_child(gui.Label("5 (87% certainty)"))

        collapse.add_child(vgrid)


        # Create a tab control. This is really a set of N layouts on top of each

        # other, but with only one selected.

        tabs = gui.TabControl()

        tab1 = gui.Vert()

        tab1.add_child(gui.Checkbox("Enable option 1"))

        tab1.add_child(gui.Checkbox("Enable option 2"))

        tab1.add_child(gui.Checkbox("Enable option 3"))

        tabs.add_tab("Options", tab1)

        tab2 = gui.Vert()

        tab2.add_child(gui.Label("No plugins detected"))

        tab2.add_stretch()

        tabs.add_tab("Plugins", tab2)

        tab3 = gui.RadioButton(gui.RadioButton.VERT)

        tab3.set_items(["Apple", "Orange"])


        def vt_changed(idx):

            print(f"current cargo: {tab3.selected_value}")


        tab3.set_on_selection_changed(vt_changed)

        tabs.add_tab("Cargo", tab3)

        tab4 = gui.RadioButton(gui.RadioButton.HORIZ)

        tab4.set_items(["Air plane", "Train", "Bus"])


        def hz_changed(idx):

            print(f"current traffic plan: {tab4.selected_value}")


        tab4.set_on_selection_changed(hz_changed)

        tabs.add_tab("Traffic", tab4)

        collapse.add_child(tabs)


        # Quit button. (Typically this is a menu item)

        button_layout = gui.Horiz()

        ok_button = gui.Button("Ok")

        ok_button.set_on_clicked(self._on_ok)

        button_layout.add_stretch()

        button_layout.add_child(ok_button)


        layout.add_child(collapse)

        layout.add_child(button_layout)


        # We're done, set the window's layout

        w.add_child(layout)


    def _on_pc_button(self):

        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",

                                 self.window.theme)

        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")

        filedlg.add_filter("", "All files")

        filedlg.set_on_cancel(self._on_filedlg_cancel)

        filedlg.set_on_done(self._on_filedlg_done)

        self.window.show_dialog(filedlg)

    def _on_model_button(self):
        
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",

                                 self.window.theme)

        filedlg.add_filter(".pth", "Trained Model (.pth)")

        filedlg.add_filter("", "All files")

        filedlg.set_on_cancel(self._on_filedlg_cancel)

        filedlg.set_on_done(self._on_filedlg_done)

        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):

        self.window.close_dialog()


    def _on_filedlg_done(self, path):

        self._pcfileedit.text_value = path

        self.window.close_dialog()


    def _on_cb(self, is_checked):

        if is_checked:

            text = "Sorry, effects are unimplemented"

        else:

            text = "Good choice"


        self.show_message_dialog("There might be a problem...", text)


    def _on_switch(self, is_on):

        if is_on:

            print("Camera would now be running")

        else:

            print("Camera would now be off")


    # This function is essentially the same as window.show_message_box(),

    # so for something this simple just use that, but it illustrates making a

    # dialog.

    def show_message_dialog(self, title, message):

        # A Dialog is just a widget, so you make its child a layout just like

        # a Window.

        dlg = gui.Dialog(title)


        # Add the message text

        em = self.window.theme.font_size

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        dlg_layout.add_child(gui.Label(message))


        # Add the Ok button. We need to define a callback function to handle

        # the click.

        ok_button = gui.Button("Ok")

        ok_button.set_on_clicked(self._on_dialog_ok)


        # We want the Ok button to be an the right side, so we need to add

        # a stretch item to the layout, otherwise the button will be the size

        # of the entire row. A stretch item takes up as much space as it can,

        # which forces the button to be its minimum size.

        button_layout = gui.Horiz()

        button_layout.add_stretch()

        button_layout.add_child(ok_button)


        # Add the button layout,

        dlg_layout.add_child(button_layout)

        # ... then add the layout as the child of the Dialog

        dlg.add_child(dlg_layout)

        # ... and now we can show the dialog

        self.window.show_dialog(dlg)


    def _on_dialog_ok(self):

        self.window.close_dialog()


    def _on_color(self, new_color):

        self._label.text_color = new_color


    def _on_combo(self, new_val, new_idx):

        print(new_idx, new_val)


    def _on_list(self, new_val, is_dbl_click):

        print(new_val)


    def _on_tree(self, new_item_id):

        print(new_item_id)


    def _on_slider(self, new_val):

        self._progress.value = new_val / 20.0


    def _on_text_changed(self, new_text):

        print("edit:", new_text)


    def _on_value_changed(self, new_text):

        print("value:", new_text)


    def _on_vedit(self, new_val):

        print(new_val)


    def _on_ok(self):

        gui.Application.instance.quit()


    def _on_menu_checkable(self):

        gui.Application.instance.menubar.set_checked(

            ExampleWindow.MENU_CHECKABLE,

            not gui.Application.instance.menubar.is_checked(

                ExampleWindow.MENU_CHECKABLE))


    def _on_menu_quit(self):

        gui.Application.instance.quit()



# This class is essentially the same as window.show_message_box(),

# so for something this simple just use that, but it illustrates making a

# dialog.

class MessageBox:


    def __init__(self, title, message):

        self._window = None


        # A Dialog is just a widget, so you make its child a layout just like

        # a Window.

        dlg = gui.Dialog(title)


        # Add the message text

        em = self.window.theme.font_size

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        dlg_layout.add_child(gui.Label(message))


        # Add the Ok button. We need to define a callback function to handle

        # the click.

        ok_button = gui.Button("Ok")

        ok_button.set_on_clicked(self._on_ok)


        # We want the Ok button to be an the right side, so we need to add

        # a stretch item to the layout, otherwise the button will be the size

        # of the entire row. A stretch item takes up as much space as it can,

        # which forces the button to be its minimum size.

        button_layout = gui.Horiz()

        button_layout.add_stretch()

        button_layout.add_child(ok_button)


        # Add the button layout,

        dlg_layout.add_child(button_layout)

        # ... then add the layout as the child of the Dialog

        dlg.add_child(dlg_layout)


    def show(self, window):

        self._window = window


    def _on_ok(self):

        self._window.close_dialog()



def main():

    # We need to initialize the application, which finds the necessary shaders for

    # rendering and prepares the cross-platform window abstraction.

    gui.Application.instance.initialize()


    w = ExampleWindow()


    # Run the event loop. This will not return until the last window is closed.

    gui.Application.instance.run()



if __name__ == "__main__":

    main()