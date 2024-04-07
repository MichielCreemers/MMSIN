#!/bin/bash
#assuming that the anaconda environment is already selected in terminal

# Run the Python script
python utils/projections.py --pc_path "test_data/soldier.ply" \
                             --image_path "test_data/soldier_projections_linux" \
                             --x_projections 4 \
                             --y_projections 8

