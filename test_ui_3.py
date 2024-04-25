import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from tkinter import filedialog
import glob
import torch
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
        self.geometry("800x300")
        ctk.set_appearance_mode("dark")  # Dark mode
        ctk.set_default_color_theme("blue")  # Blue theme

        self.setup_gui()

    def setup_gui(self):
        self.grid_columnconfigure(1, weight=1)

        frame_style = {"corner_radius": 10, "fg_color": "#333333"}  # Adjust frame colors and corner radius

        # Point Cloud File Selection
        pc_file_label = ctk.CTkLabel(self, text="Point Cloud File:")
        pc_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.pc_file_edit = ctk.CTkEntry(self, width=120, **frame_style)
        self.pc_file_edit.grid(row=0, column=1, padx=10, pady=10, sticky="we")

        pc_file_button = ctk.CTkButton(self, text="Browse", command=self.on_pc_button, **frame_style)
        pc_file_button.grid(row=0, column=2, padx=10, pady=10)

        # Model File Selection
        model_file_label = ctk.CTkLabel(self, text="Model File:")
        model_file_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.model_file_edit = ctk.CTkEntry(self, width=120, **frame_style)
        self.model_file_edit.grid(row=1, column=1, padx=10, pady=10, sticky="we")

        model_file_button = ctk.CTkButton(self, text="Browse", command=self.on_model_button, **frame_style)
        model_file_button.grid(row=1, column=2, padx=10, pady=10)

        # Calculate Button
        calculate_button = ctk.CTkButton(self, text="Calculate Quality", command=self.on_ok, **frame_style)
        calculate_button.grid(row=2, column=0, columnspan=3, padx=20, pady=20)


    def on_pc_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.obj *.ply *.stl")])
        if file_path:
            self.pc_file_edit.delete(0, ctk.END)
            self.pc_file_edit.insert(0, file_path)

    def on_model_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
        if file_path:
            self.model_file_edit.delete(0, ctk.END)
            self.model_file_edit.insert(0, file_path)

    def _assess_quality(self):
        projections_folder = "test"
        projections.make_projections(self.pc_file_edit.get(),projections_folder,4, 4, 2, 'default', False)

        images = glob.glob(f'{projections_folder}/*.png')

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
        scaler_params = np.load('scaler_params.npy')

        scaler_loaded = MinMaxScaler()
        scaler_loaded.min_ = scaler_params[0]
        scaler_loaded.scale_ = scaler_params[1]

        nss_features = scaler_loaded.transform(features_df)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MM_NSSInet()
        model.load_state_dict(torch.load(self.model_file_edit.get()))
        model = model.to(device)
        model.eval()

        print('Begin inference.')
        with torch.no_grad():
            transformed_imgs = transformed_imgs.to(device).unsqueeze(0)
            nss_features_tensor = torch.tensor(nss_features, dtype=torch.float).squeeze()
            nss_features_tensor = nss_features_tensor.to(device).unsqueeze(0)
            outputs = model(transformed_imgs, nss_features_tensor)
            score = outputs.item()

        print('Predicted quality score:', score)
        showinfo("Quality Score", f"Predicted quality score: {score}")

    def on_ok(self):
        self._assess_quality()  

if __name__ == "__main__":
    app = ExampleWindow()
    app.mainloop()
