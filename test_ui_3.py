import tkinter as tk
from tkinter import ttk
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

class ExampleWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Test")
        self.geometry("1100x500")

        style = ttk.Style()
        style.configure("TButton", padding=(8, 16), font=("Arial", 14))
        style.map("TButton",
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '!disabled', '#003d80'), ('active', '#0056b3')])

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        pc_file_frame = ttk.Frame(main_frame)
        pc_file_frame.pack(pady=10, padx=10, fill=tk.X)

        pc_file_label = ttk.Label(pc_file_frame, text="Select point cloud file")
        pc_file_label.pack(side=tk.LEFT, padx=(0, 10))

        self.pc_file_edit = ttk.Entry(pc_file_frame, width=40)
        self.pc_file_edit.pack(side=tk.LEFT, fill=tk.X, expand=True)

        pc_file_button = ttk.Button(pc_file_frame, text="...", command=self.on_pc_button)
        pc_file_button.pack(side=tk.LEFT)

        model_file_frame = ttk.Frame(main_frame)
        model_file_frame.pack(pady=10, padx=10, fill=tk.X)

        model_file_label = ttk.Label(model_file_frame, text="Select model file")
        model_file_label.pack(side=tk.LEFT, padx=(0, 10))

        self.model_file_edit = ttk.Entry(model_file_frame, width=40)
        self.model_file_edit.pack(side=tk.LEFT, fill=tk.X, expand=True)

        model_file_button = ttk.Button(model_file_frame, text="...", command=self.on_model_button)
        model_file_button.pack(side=tk.LEFT)

        proj_frame = ttk.Frame(main_frame)
        proj_frame.pack(pady=10, padx=10, fill=tk.X)

        x_proj_label = ttk.Label(proj_frame, text="Number of x-projections")
        x_proj_label.pack(side=tk.LEFT, padx=(0, 10))

        self.x_proj_edit = ttk.Entry(proj_frame, width=10)
        self.x_proj_edit.pack(side=tk.LEFT)

        y_proj_label = ttk.Label(proj_frame, text="Number of y-projections")
        y_proj_label.pack(side=tk.LEFT, padx=(20, 10))

        self.y_proj_edit = ttk.Entry(proj_frame, width=10)
        self.y_proj_edit.pack(side=tk.LEFT)

        ok_button = ttk.Button(main_frame, text="Calculate Quality", command=self.on_ok)
        ok_button.pack(pady=10)

    def on_pc_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.obj *.ply *.stl")])
        if file_path:
            self.pc_file_edit.delete(0, tk.END)
            self.pc_file_edit.insert(0, file_path)

    def on_model_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
        if file_path:
            self.model_file_edit.delete(0, tk.END)
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
