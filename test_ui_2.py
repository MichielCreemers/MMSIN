import sys
import glob
from torchvision import transforms
from PIL import Image
from utils import projections
from utils.NSS import feature_extract, nss_functions, feature_functions
from models.main_model import MM_NSSInet
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QMessageBox
import numpy as np

class CustomFileDialog(QFileDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 16px;
            }
            QPushButton {
                padding: 8px 16px;
                font-size: 14px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
        """)

class ExampleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test")
        self.setGeometry(100, 100, 1100, 500)

        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 16px;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #aaa;
                border-radius: 4px;
            }
            QPushButton {
                padding: 8px 16px;
                font-size: 14px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
            QSlider::groove:vertical {
                background: #ddd;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: #007bff;
                border: 1px solid #aaa;
                width: 16px;
                height: 16px;
                margin: -8px 0;
                border-radius: 8px;
            }
        """)

        self.setup_gui()

    def setup_gui(self):
        layout = QVBoxLayout()

        # File-chooser widget for point cloud file
        pc_file_layout = QHBoxLayout()
        pc_file_label = QLabel("Select point cloud file")
        self.pc_file_edit = QLineEdit()
        pc_file_button = QPushButton("...")
        pc_file_button.clicked.connect(self.on_pc_button)

        pc_file_layout.addWidget(pc_file_label)
        pc_file_layout.addWidget(self.pc_file_edit)
        pc_file_layout.addWidget(pc_file_button)

        layout.addLayout(pc_file_layout)

        # File-chooser widget for model file
        model_file_layout = QHBoxLayout()
        model_file_label = QLabel("Select model file")
        self.model_file_edit = QLineEdit()
        model_file_button = QPushButton("...")
        model_file_button.clicked.connect(self.on_model_button)

        model_file_layout.addWidget(model_file_label)
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(model_file_button)

        layout.addLayout(model_file_layout)

        # Number editors for x and y projections
        proj_layout = QHBoxLayout()
        x_proj_label = QLabel("Number of x-projections")
        self.x_proj_edit = QLineEdit()
        y_proj_label = QLabel("Number of y-projections")
        self.y_proj_edit = QLineEdit()

        proj_layout.addWidget(x_proj_label)
        proj_layout.addWidget(self.x_proj_edit)
        proj_layout.addWidget(y_proj_label)
        proj_layout.addWidget(self.y_proj_edit)

        layout.addLayout(proj_layout)

        # Progress bar
        self.progress_slider = QSlider()
        self.progress_slider.setOrientation(1)
        layout.addWidget(self.progress_slider)

        # Confirmation button
        ok_button = QPushButton("Calculate Quality")
        ok_button.clicked.connect(self.on_ok)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def on_pc_button(self):
        dialog = CustomFileDialog(self, "Select point cloud file")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Point Cloud Files (*.obj *.ply *.stl)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.pc_file_edit.setText(selected_files[0])

    def on_model_button(self):
        dialog = CustomFileDialog(self, "Select model file")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Model Files (*.pth)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.model_file_edit.setText(selected_files[0])

    def _assess_quality(self):
        projections_folder = "test"
        images = glob.glob(f'{projections_folder}/*.png')

        transformation = transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

        transformed_imgs = torch.zeros([len(images), 3, 224, 224])

        for i in range(len(images)):
            read_image = Image.open(images[i]).convert('RGB')
            read_image = transformation(read_image)
            transformed_imgs[i] = read_image

        nss_features = feature_extract.get_feature_vector(self.pc_file_edit.text())

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
        model.load_state_dict(torch.load(self.model_file_edit.text()))
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
        QMessageBox.information(self, "Quality Score", f"Predicted quality score: {score}", QMessageBox.Ok)


    def on_ok(self):
        self._assess_quality()  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExampleWindow()
    window.show()
    sys.exit(app.exec_())