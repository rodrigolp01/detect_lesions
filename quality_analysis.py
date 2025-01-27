import cv2
import numpy as np
import skimage
from skimage import img_as_float
from sewar.full_ref import mse as sewar_mse
from sewar.full_ref import rmse as sewar_rmse
from sewar.full_ref import psnr as sewar_psnr
from sewar.full_ref import ssim as sewar_ssim
import pandas as pd
import os


def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def mse2(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def mse3(imageA, imageB):
    #imageA = img_as_float(imageA)
    #imageB = img_as_float(imageB)
    return skimage.metrics.mean_squared_error(imageA, imageB)

def mse4(imageA, imageB):
     return sewar_mse(imageA, imageB)

def psnr(imageA, imageB):
    #imageA = img_as_float(imageA)
    #imageB = img_as_float(imageB)
    return skimage.metrics.peak_signal_noise_ratio(imageA, imageB)

def psnr2(imageA, imageB):
    #imageA = img_as_float(imageA)
    #imageB = img_as_float(imageB)
    return sewar_psnr(imageA, imageB)

def ssim(imageA, imageB):
     #imageA = img_as_float(imageA)
     #imageB = img_as_float(imageB)
     return skimage.metrics.structural_similarity(imageA, imageB, data_range=imageB.max() - imageB.min())

def ssim2(imageA, imageB):
     #imageA = img_as_float(imageA)
     #imageB = img_as_float(imageB)
     return sewar_ssim(imageA, imageB)

def nrmse(imageA, imageB):
     #imageA = img_as_float(imageA)
     #imageB = img_as_float(imageB)
     return skimage.metrics.normalized_root_mse(imageA, imageB)

def rmse(imageA, imageB):
     #imageA = img_as_float(imageA)
     #imageB = img_as_float(imageB)
     return sewar_rmse(imageA, imageB)


if __name__ == '__main__':
     
     #orig_img_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast_folds/train/0/22580218_5530d5782fc89dd7_MG_L_CC_ANON.png'
     #processed_img_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_enh/train/0/22580218_5530d5782fc89dd7_MG_L_CC_ANON.png'
     #orig_img_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast_folds/train/0/53582540_3e73f1c0670cfb0a_MG_R_ML_ANON.png'
     #processed_img_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_enh/train/0/53582540_3e73f1c0670cfb0a_MG_R_ML_ANON.png'

     #orig_path_inbreast = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast_folds/train/2'
     processed_path_raw = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_raw/train/2'
     processed_path_denoise = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds/train/2'
     processed_path_denoise_enh = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_enh/train/2'
     processed_path_denoise_enh_gamma = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_gamma_enh/train/2'
     processed_path_denoise_segmented = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_segmented_smooth/train/2'

     inbreast_img_names = ['20586934_6c613a14b80a8591_MG_L_CC_ANON.png','20586986_6c613a14b80a8591_MG_L_ML_ANON.png',
                           '20587054_b6a4f750c6df4f90_MG_R_CC_ANON.png','20587080_b6a4f750c6df4f90_MG_R_ML_ANON.png', 
                           '20588536_bf1a6aaadb05e3df_MG_L_ML_ANON.png','20588562_bf1a6aaadb05e3df_MG_L_CC_ANON.png']
     
     mias_img_names = ['mdb102.png','mdb023.png',
                           'mdb120.png','mdb125.png', 
                           'mdb184.png','mdb231.png']
     
     cbis_img_names = ['1-000-P_00363.jpg','1-001-P_01818.jpg',
                           '1-002-P_01818.jpg','1-008-P_01361.jpg', 
                           '1-009-P_00962.jpg','1-011-P_01504.jpg']
     
     mini_img_names = ['A_1399_1.RIGHT_CC.png','A_1402_1.LEFT_MLO.png',
                           'A_1465_1.RIGHT_MLO.png','A_1469_1.LEFT_CC.png', 
                           'A_1489_1.LEFT_CC.png', 'A_1500_1.LEFT_CC.png']
     
     img_names = inbreast_img_names + mias_img_names + cbis_img_names + mini_img_names

     output_csv_file = 'quality_analysis.csv'
     df = pd.DataFrame()
     df['images'] = img_names

     mse_denoise_list = []
     psnr_denoise_list = []
     ssim_denoise_list = []
     rmse_denoise_list = []

     mse_enh_list = []
     psnr_enh_list = []
     ssim_enh_list = []
     rmse_enh_list = []

     mse_gamma_list = []
     psnr_gamma_list = []
     ssim_gamma_list = []
     rmse_gamma_list = []

     mse_seg_list = []
     psnr_seg_list = []
     ssim_seg_list = []
     rmse_seg_list = []

     for img in img_names:

          print(img)

          raw_img = cv2.imread(processed_path_raw + '/' + img)
          denoise_img = cv2.imread(processed_path_denoise + '/' + img)
          enh_img = cv2.imread(processed_path_denoise_enh + '/' + img)
          gamma_img = cv2.imread(processed_path_denoise_enh_gamma + '/' + img)
          seg_img = cv2.imread(processed_path_denoise_segmented + '/' + img)

          raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
          denoise_img = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2GRAY)
          enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)
          gamma_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
          seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

          mse_error1 = mse4(raw_img, denoise_img)
          mse_denoise_list.append(mse_error1)

          mse_error2 = mse4(raw_img, enh_img)
          mse_enh_list.append(mse_error2)

          mse_error3 = mse4(raw_img, gamma_img)
          mse_gamma_list.append(mse_error3)

          mse_error4 = mse4(raw_img, seg_img)
          mse_seg_list.append(mse_error4)

          psnr_error1 = psnr2(raw_img, denoise_img)
          psnr_denoise_list.append(psnr_error1)

          psnr_error2 = psnr2(raw_img, enh_img)
          psnr_enh_list.append(psnr_error2)

          psnr_error3 = psnr2(raw_img, gamma_img)
          psnr_gamma_list.append(psnr_error3)

          psnr_error4 = psnr2(raw_img, seg_img)
          psnr_seg_list.append(psnr_error4)

          ssim_error1 = ssim2(raw_img, denoise_img)[0]
          ssim_denoise_list.append(ssim_error1)

          ssim_error2 = ssim2(raw_img, enh_img)[0]
          ssim_enh_list.append(ssim_error2)

          ssim_error3 = ssim2(raw_img, gamma_img)[0]
          ssim_gamma_list.append(ssim_error3)

          ssim_error4 = ssim2(raw_img, seg_img)[0]
          ssim_seg_list.append(ssim_error4)

          rmse_error1 = rmse(raw_img, denoise_img)
          rmse_denoise_list.append(rmse_error1)

          rmse_error2 = rmse(raw_img, enh_img)
          rmse_enh_list.append(rmse_error2)

          rmse_error3 = rmse(raw_img, gamma_img)
          rmse_gamma_list.append(rmse_error3)

          rmse_error4 = rmse(raw_img, seg_img)
          rmse_seg_list.append(rmse_error4)

     df['mse_denoise'] = mse_denoise_list
     df['mse_enh'] = mse_enh_list
     df['mse_gamma'] = mse_gamma_list
     df['mse_seg'] = mse_seg_list
     df['psnr_denoise'] = psnr_denoise_list
     df['psnr_enh'] = psnr_enh_list
     df['psnr_gamma'] = psnr_gamma_list
     df['psnr_seg'] = psnr_seg_list
     df['ssim_denoise'] = ssim_denoise_list
     df['ssim_enh'] = ssim_enh_list
     df['ssim_gamma'] = ssim_gamma_list
     df['ssim_seg'] = ssim_seg_list
     df['rmse_denoise'] = rmse_denoise_list
     df['rmse_enh'] = rmse_enh_list
     df['rmse_gamma'] = rmse_gamma_list
     df['rmse_seg'] = rmse_seg_list

     #with open(os.path.join(output_csv_file), 'w') as f:
     #   for key in df.keys():
     #       f.write("%s,%s\n"%(key,df[key]))
     df.to_csv(output_csv_file, index=False)   
