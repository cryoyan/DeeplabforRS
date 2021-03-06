#!/bin/bash

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

root=$(python2 ${para_py} -p ${para_file} working_root)
eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
augscript=${eo_dir}/image_augment.py
#test_dir=EbolingUAV_deeplab_7

SECONDS=0

image_name_prefix=UAV_DOM_Eboling
label_name_prefix=raster_class_train_polygons_digitize_gps_0.15m_v5

#echo $root
#echo $eo_dir
#exit

# current folder (without path)
test_dir=${PWD##*/}

# only augmentation on gully images (move non-gully image to temp folder)
mkdir -p split_images_temp
mkdir -p split_labels_temp
for id in $(seq 10 24); do
	echo $id
	mv split_images/${image_name_prefix}_${id}_*.tif  split_images_temp/.
	mv split_labels/${label_name_prefix}_${id}_*.tif  split_labels_temp/.
done

#update list
${eo_dir}/bash_script/get_list.sh

#exit 
#backup 
cp list/image_list.txt list/image_list_without_augmentation.txt
cp list/label_list.txt list/label_list_without_augmentation.txt

#augment training images
${augscript} list/image_list.txt -o ${root}/${test_dir}/split_images

#augment training lables
${augscript} --is_ground_truth list/label_list.txt -o ${root}/${test_dir}/split_labels
# force the groud truth only have 0 and 1 after augmentation; could loss some pixels
for item in $(ls ${root}/${test_dir}/split_labels/*.tif)
do
    gdal_calc.py -A $item  --outfile=${item} --calc="A==1"  --debug --type='Byte' --overwrite
done

# mv file back
mv split_images_temp/* split_images/.
mv split_labels_temp/* split_labels/.

# update list file
${eo_dir}/bash_script/get_list.sh

duration=$SECONDS
echo "$(date): time cost of preparing training images augmentation: ${duration} seconds">>"time_cost.txt"