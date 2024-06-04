import os
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from tkinter import filedialog, font
import open3d as o3d
import glob
import torch
import open3d.visualization.rendering as rendering
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from tkinter.messagebox import showinfo
from utils import projections
from utils.NSS import feature_extract, nss_functions, feature_functions
from models.main_model import MM_NSSInet

class ExampleWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3D Point Cloud Quality Assessment")
        self.geometry("800x1000")
        ctk.set_appearance_mode("dark")  # Dark mode
        ctk.set_default_color_theme("blue")  # Blue theme
        self.font = ctk.CTkFont(family='Helvetica', size=18)
        self.setup_gui()

    def setup_gui(self):
        self.grid_columnconfigure(1, weight=1)

        frame_style = {"corner_radius": 0, "fg_color": "#333333"}  # Adjust frame colors and corner radius

        # Point Cloud File Selection
        pc_file_label = ctk.CTkLabel(self, text="Point Cloud File:", font=self.font)
        pc_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.pc_file_edit = ctk.CTkEntry(self, width=120, **frame_style, font=self.font)
        self.pc_file_edit.grid(row=0, column=1, padx=10, pady=10, sticky="we")

        pc_file_button = ctk.CTkButton(self, text="Browse", command=self.on_pc_button, **frame_style, font=self.font)
        pc_file_button.grid(row=0, column=2, padx=10, pady=10)

        # Model File Selection
        model_file_label = ctk.CTkLabel(self, text="Model File:", font=self.font)
        model_file_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        model_files = self.get_model_files("pretrained_models")

        # Ensure the list is not empty
        if not model_files:
            model_files = ["No model files found"]

        self.model_file_dropdown = ctk.CTkOptionMenu(self, values=model_files, **frame_style, font=self.font)
        self.model_file_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky="we")

        # Calculate Button
        calculate_button = ctk.CTkButton(self, text="Calculate Quality", command=self.on_ok, **frame_style, font=self.font)
        calculate_button.grid(row=2, column=0, columnspan=3, padx=20, pady=20)

        self.log_label = ctk.CTkLabel(self, text="", height=50, wraplength=1000, **frame_style, font=self.font)
        self.log_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="we")

    def get_model_files(self, directory):
        try:
            # List all .pth files in the given directory
            files = [f for f in os.listdir(directory) if f.endswith('.pth')]
            return files
        except Exception as e:
            print(f"Error accessing directory: {e}")
            return []

    def on_pc_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.ply")])
        if file_path:
            self.pc_file_edit.delete(0, ctk.END)
            self.pc_file_edit.insert(0, file_path)

    def log_message(self, message):
        self.log_label.configure(text=message)
        self.log_label.update()

    def _assess_quality(self):
        projections_folder = "PCQA_tool_projections"
        self.log_message('Generating projections')
        if(self.pc_file_edit.get() == ""):
            self.log_message("no ply selected")
            return
        if(self.model_file_dropdown == "No model files found"):
            self.log_message("no model selected")
            return

        projections.make_projections(self.pc_file_edit.get(), projections_folder, 4, 4, 2, 'default', False)

        images = glob.glob(f'{projections_folder}/*.png')

        first_image = Image.open(images[0])
        self.photo = ImageTk.PhotoImage(first_image)
        self.point_cloud_canvas = ctk.CTkCanvas(self, bg="gray90", width=0.9*self.log_label.winfo_width(), height=0.9*self.log_label.winfo_width())
        self.point_cloud_canvas.grid(row=4, column=0, columnspan=3, padx=20, pady=20)
        self.point_cloud_canvas.delete("all")  # Clear the canvas
        self.point_cloud_canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.log_message('Transforming projections')

        transformation = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transformed_imgs = torch.zeros([len(images), 3, 224, 224])

        for i in range(len(images)):
            read_image = Image.open(images[i]).convert('RGB')
            read_image = transformation(read_image)
            transformed_imgs[i] = read_image

        self.log_message('Generating NSS features')

        # Assuming feature_extract.get_feature_vector and MM_NSSInet are defined elsewhere
        nss_features = feature_extract.get_feature_vector(self.pc_file_edit.get())

        feature_names = ["l_mean", "l_std", "l_entropy", "a_mean", "a_std", "a_entropy", "b_mean", "b_std", "b_entropy",
                         "curvature_mean", "curvature_std", "curvature_entropy", "curvature_ggd1", "curvature_ggd2",
                         "curvature_aggd1", "curvature_aggd2", "curvature_aggd3", "curvature_aggd4", "curvature_gamma1",
                         "curvature_gamma2", "anisotropy_mean", "anisotropy_std", "anisotropy_entropy",
                         "anisotropy_ggd1", "anisotropy_ggd2", "anisotropy_aggd1", "anisotropy_aggd2",
                         "anisotropy_aggd3", "anisotropy_aggd4", "anisotropy_gamma1", "anisotropy_gamma2",
                         "linearity_mean", "linearity_std", "linearity_entropy", "linearity_ggd1", "linearity_ggd2",
                         "linearity_aggd1", "linearity_aggd2", "linearity_aggd3", "linearity_aggd4", "linearity_gamma1",
                         "linearity_gamma2", "planarity_mean", "planarity_std", "planarity_entropy", "planarity_ggd1",
                         "planarity_ggd2", "planarity_aggd1", "planarity_aggd2", "planarity_aggd3", "planarity_aggd4",
                         "planarity_gamma1", "planarity_gamma2", "sphericity_mean", "sphericity_std",
                         "sphericity_entropy", "sphericity_ggd1", "sphericity_ggd2", "sphericity_aggd1",
                         "sphericity_aggd2", "sphericity_aggd3", "sphericity_aggd4", "sphericity_gamma1",
                         "sphericity_gamma2"]

        features_df = pd.DataFrame([nss_features], columns=feature_names)
        selected_model = self.model_file_dropdown.get()
        if (selected_model == "WPC.pth"):
            scaler_params = np.load('WPC/scaler_params.npy')
        elif(selected_model == "WPC2.pth"):
            scaler_params = np.load('WPC2/scaler_params.npy')
        elif(selected_model == "SJTU.pth"):
            scaler_params = np.load('SJTU/scaler_params.npy')
        else:
            scaler_params = np.load('WPC/scaler_params.npy')

        self.log_message('Scaling NSS features')

        scaler_loaded = MinMaxScaler()
        scaler_loaded.min_ = scaler_params[0]
        scaler_loaded.scale_ = scaler_params[1]

        nss_features = scaler_loaded.transform(features_df)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MM_NSSInet()
        model.load_state_dict(torch.load(os.path.join("pretrained_models", self.model_file_dropdown.get())))
        model = model.to(device)
        model.eval()

        self.log_message('Begin inference')
        with torch.no_grad():
            transformed_imgs = transformed_imgs.to(device).unsqueeze(0)
            nss_features_tensor = torch.tensor(nss_features, dtype=torch.float).squeeze()
            nss_features_tensor = nss_features_tensor.to(device).unsqueeze(0)
            outputs = model(transformed_imgs, nss_features_tensor)
            score = outputs.item()

        self.log_message('Predicted quality score: ' + str(score))

    def on_ok(self):
        self._assess_quality()

if __name__ == "__main__":
    app = ExampleWindow()
    app.mainloop()
