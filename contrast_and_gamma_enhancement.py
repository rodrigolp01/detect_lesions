from contrast_enhancement import do_contrast_enhancement, do_gamma_and_contrast_enhancement

class_list = ['0','1','2','3']
input_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_folds_4cl_cc'
output_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_folds_4cl_cc_enh2'

for cl in class_list:
    train_input_path = input_path + '/' + 'train' + '/' + cl
    train_output_path = output_path + '/' + 'train' + '/' + cl
    #do_contrast_enhancement(train_input_path, train_output_path)
    do_gamma_and_contrast_enhancement(train_input_path, train_output_path)

    valid_input_path = input_path + '/' + 'valid' + '/' + cl
    valid_output_path = output_path + '/' + 'valid' + '/' + cl
    #do_contrast_enhancement(valid_input_path, valid_output_path)
    do_gamma_and_contrast_enhancement(valid_input_path, valid_output_path)

    test_input_path = input_path + '/' + 'test' + '/' + cl
    test_output_path = output_path + '/' + 'test' + '/' + cl
    #do_contrast_enhancement(test_input_path, test_output_path)
    do_gamma_and_contrast_enhancement(test_input_path, test_output_path)
