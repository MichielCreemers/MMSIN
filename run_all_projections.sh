#!/bin/bash

target_pc_directory="SJTU/point_clouds"
target_image_directory="SJTU/projections"

for file in $(find "$target_pc_directory" -type f -name "*.ply" | sort)
do 
    base_file="$(basename $file)"
    echo "$base_file"

    target_projection_directory="${target_image_directory}/${base_file}"
    echo "$target_projection_directory"

    python utils/projections.py --pc_path "$file" \
                             --image_path "$target_projection_directory" \
                             --x_projections 4 \
                             --y_projections 4 \
                             --point_size 2

done

#!/bin/bash
