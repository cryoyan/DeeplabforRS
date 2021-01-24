#!/usr/bin/env python
# Filename: pixel_evaluation 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 December, 2019
"""

from optparse import OptionParser
import os,sys
import rasterio
import basic_src.basic as basic
import basic_src.io_function as io_function
import parameters
import gdal
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def pixel_evaluation_result(result_shp,val_shp,inf_img,pixel_accuracy_txt=None):
    """
    evaluate the result based on pixel
    :param result_shp: result shape file contains detected polygons
    :param val_shp: shape file contains validation polygons
    :param inf_img: tiff file used for inference
    :return: True is successful, False otherwise
    """
    basic.outputlogMessage("Pixel_evaluation")
    

    ds_raster = rasterio.open(inf_img)
    bounds = ds_raster.bounds
    left= bounds.left
    bottom = bounds.bottom
    right = bounds.right
    top = bounds.top
    
    result_file_base = os.path.basename(result_shp)    
    result_file_name = os.path.splitext(result_file_base)[0]
    result_tif=result_file_name+'.tif'
    
    val_file_base = os.path.basename(val_shp)
    val_file_name = os.path.splitext(val_file_base)[0]
    val_tif=val_file_name+'.tif'
    
    res = parameters.get_input_image_rescale() 
    
    os.system('gdal_rasterize -burn 1 -tr %f %f -te %f %f %f %f -ot Byte -of GTiff %s %s'%(res, res, left, bottom, right, top, result_shp, result_tif))
    os.system('gdal_rasterize -burn 1 -tr %f %f -te %f %f %f %f -ot Byte -of GTiff %s %s'%(res, res, left, bottom, right, top, val_shp, val_tif))

    pre_file = gdal.Open(result_tif, gdal.GA_ReadOnly)
    pre_array = pre_file.GetRasterBand(1).ReadAsArray()
    
    val_file = gdal.Open(val_tif, gdal.GA_ReadOnly)
    val_array = val_file.GetRasterBand(1).ReadAsArray()
    
    y_pre=pre_array.flatten()
    y_val=val_array.flatten()
    
     # calculate precision, recall, F1score, IOU_score
    Pixel_precision = precision_score(y_val, y_pre, labels=[1,0], average=None)
   
    Pixel_recall = recall_score(y_val, y_pre,labels=[1,0], average=None)

    Pixel_F1score = f1_score(y_val, y_pre,labels=[1,0], average=None)

    #Pixel_IOU_score = jaccard_similarity_score(y_val,y_pre)
    
    mean_precision = np.mean(Pixel_precision)
    mean_recall = np.mean(Pixel_recall)
    mean_F1score = np.mean(Pixel_F1score)

    # confusion matrix
    matrix = np.array2string(confusion_matrix(y_val,y_pre, labels=[1,0]))

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(y_val,y_pre,labels=[1,0]).reshape(-1)

    #calculate IOU for the class 0
    Pixel_IOU_score = np.true_divide(tp,(tp+fp+fn))
    Pixel_IOU_0 = np.true_divide(tn,(tn+fp+fn))
    mean_IOU_score = np.mean([Pixel_IOU_score,Pixel_IOU_0])

    # classification report for precision, recall f1-score and accuracy
    matrix_report = classification_report(y_val,y_pre,labels=[1,0])
    
    #output evaluation reslult
    if pixel_accuracy_txt is None:
        pixel_accuracy_txt = "pixel_accuracy_report.txt"
    f_obj = open(pixel_accuracy_txt,'w')
    f_obj.writelines('true_pos_count: %d\n'%tp)
    f_obj.writelines('false_pos_count: %d\n'% fp)
    f_obj.writelines('false_neg_count: %d\n'%fn)
    f_obj.writelines('true_neg_count: %d\n'%tn)
    f_obj.writelines('precision_1_0: %.6f %.6f\n'%(Pixel_precision[0],Pixel_precision[1]))
    f_obj.writelines('recall_1_0: %.6f %.6f\n'%(Pixel_recall[0],Pixel_recall[1]))
    f_obj.writelines('F1score_1_0: %.6f %.6f\n'%(Pixel_F1score[0],Pixel_F1score[1]))
    f_obj.writelines('IOUscore_1_0: %.6f  %.6f\n'%(Pixel_IOU_score,Pixel_IOU_0))
    f_obj.writelines('mean_precision: %.6f\n'%mean_precision)
    f_obj.writelines('mean_recall: %.6f\n'%mean_recall)
    f_obj.writelines('mean_F1score: %.6f\n'%mean_F1score)
    f_obj.writelines('mean_IOUscore: %.6f\n'%mean_IOU_score)
    f_obj.writelines('\nconfusion matrix : \n')
    f_obj.write(str(matrix))
    f_obj.writelines('\nClassification report : \n')
    f_obj.write(str(matrix_report))
    f_obj.close()

    pass

def main(options, args):
    input_shp = args[0]
    input_tif = args[1]

    # evaluation result
    multi_val_files = parameters.get_string_parameters_None_if_absence('','validation_shape_list')
    if multi_val_files is None:
        val_path = parameters.get_validation_shape()
    else:
        val_path = io_function.get_path_from_txt_list_index(multi_val_files)
        # try to change the home folder path if the file does not exist
        val_path = io_function.get_file_path_new_home_folder(val_path)

    if os.path.isfile(val_path):
        basic.outputlogMessage('Start evaluation, input: %s, validation file: %s'%(input_shp, val_path))
        pixel_evaluation_result(input_shp, val_path, input_tif)
    else:
        basic.outputlogMessage("warning, validation polygon (%s) not exist, skip evaluation"%val_path)



if __name__=='__main__':
    usage = "usage: %prog [options] input_path "
    parser = OptionParser(usage=usage, version="1.0 2017-7-24")
    parser.description = 'Introduction: evaluate the mapping results based on pixel '
    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    ## set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
