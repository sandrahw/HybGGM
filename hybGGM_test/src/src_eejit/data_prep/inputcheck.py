import numpy as np
import matplotlib.pyplot as plt

input_old = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/subsamples/full_input_subsample_0.npy')
# target_old = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/full_target_deltawtd.npy')
target_subsample = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/subsamples/full_target_deltawtd_subsample_0.npy')
inpsel = input_old[0, -1:, :, :]
# tarsel = target_old[0, :, :]
tarsel_subsample = target_subsample[0, :, :]


plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(inpsel[0, 0,:, :])
plt.colorbar()
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(tarsel_subsample[0, :, :])
plt.title('Target')
plt.colorbar()
plt.subplot(1, 3, 3)
diff = inpsel[0, 0, :, :] - tarsel_subsample[0, :, :]
plt.imshow(diff)
plt.colorbar()
plt.title('Difference')
min(diff.flatten()), max(diff.flatten())
min(inpsel.flatten()), max(inpsel.flatten())
min(tarsel_subsample.flatten()), max(tarsel_subsample.flatten())



input_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/subsamples/full_input_subsample_0.npy')
target_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/full_target_deltawtd.npy')
target_subsample_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/subsamples/full_target_deltawtd_subsample_0.npy')

input_new_wtd_sel = input_new_wtd[0, -1:, :, :]
tarsel_subsample = target_subsample_new_wtd[0, :, :]
# tarsel_subsample = target_new_wtd[0, :, :]

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(input_new_wtd_sel[0, 0,:, :])
plt.colorbar()
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(tarsel_subsample[0, :, :])
plt.title('Target')
plt.colorbar()
plt.subplot(1, 3, 3)
diff = input_new_wtd_sel[0, 0, :, :] - tarsel_subsample[0, :, :]
plt.imshow(diff)
plt.colorbar()
plt.title('Difference')

'''correlation between inputs and target'''
params_modflow = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
                    'bed_conductance_used', 
                    'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
                    'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
                    'surface_water_bed_elevation_used',
                    'surface_water_elevation', 'net_RCH', 
                    'bottom_lowermost_layer', 'bottom_uppermost_layer', 
                    'drain_conductance', 
                    'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
                    'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
                    'top_uppermost_layer',
                    'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer',
                    'globgm-wtd']

#create list with 1-21 based on len input_new_wtd[1]
inpl = list(range(0, len(input_new_wtd[1])))
for param,i in zip(params_modflow[:], inpl[:]):
    # print(param, i)
    #calculate correlation between input layer and target layer
    inp = input_new_wtd[0, i, :,:]
    # tar = input_new_wtd[0, -1, :, :]
    tar = target_subsample_new_wtd[0, :, :]
    corr = np.corrcoef(inp[0,:,:].flatten(), tar[0,:,:].flatten())[0,1]
    print('Correlation between %s and deltawtd: %s' % (param, corr))

    corr = np.corrcoef(inp[0,:,:], tar[0,:,:])
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(inp[0,:,:])
    plt.colorbar(shrink=0.8)
    plt.clim(min(inp.flatten()), max(inp.flatten()))
    plt.title('Input %s' % param)
    plt.subplot(1, 3, 2)
    plt.imshow(tar[0,:,:])
    plt.colorbar(shrink=0.8)
    plt.clim(min(tar.flatten()), max(tar.flatten()))
    plt.title('Target')
    plt.subplot(1, 3, 3)
    plt.imshow(corr)
    plt.colorbar(shrink=0.8)
    plt.clim(-1, 1)
    plt.title('average correlation %s' % np.nanmean(corr))
    plt.tight_layout
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/src/data_prep/corr_%s.png' % param)



#check if validation target is the same as the actual target
target_validation = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/target_validation.npy')
input_validation = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/input_validation.npy')

input_deltawtd = input_validation[0, -1, :, :]
target_deltawtd = target_validation[0, :, :]    
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(input_deltawtd[0,:,:])
plt.colorbar()  
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(target_deltawtd[0,:,:])
plt.colorbar()
plt.title('Target') 
plt.subplot(1, 3, 3)    
diff = input_deltawtd[0,:,:] - target_deltawtd[0,:,:]
plt.imshow(diff)
plt.colorbar()

#check the same for the new wtd data training set
target_validation_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/target_validation.npy')
input_validation_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/input_validation.npy')
input_deltawtd_new_wtd = input_validation_new_wtd[0, -1, :, :] #should be wtd
target_deltawtd_new_wtd = target_validation_new_wtd[0, :, :] #should be delta wtd

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(input_deltawtd_new_wtd[0,:,:])
plt.colorbar()
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(target_deltawtd_new_wtd[0,:,:])
plt.colorbar()
plt.title('Target')
plt.subplot(1, 3, 3)
diff = input_deltawtd_new_wtd[0,:,:] - target_deltawtd_new_wtd[0,:,:]
plt.imshow(diff)
plt.colorbar()
plt.title('Difference')


###check if new data for wtd run is the same for wtd-1 and wtd
input_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/input_training.npy')
target_new_wtd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/target_training.npy')

input_new_wtd_sel = input_new_wtd[0, -1:, :, :]
tarsel = target_new_wtd[0, :, :]
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(input_new_wtd_sel[0, 0,:, :])
plt.colorbar()
plt.title('Input')
plt.subplot(1, 3, 2)
plt.imshow(tarsel[0, :, :])
plt.title('Target')
plt.colorbar()
plt.subplot(1, 3, 3)
diff = input_new_wtd_sel[0, 0, :, :] - tarsel[0, :, :]
plt.imshow(diff)
plt.colorbar()

#transformed vs non-transformed data
input_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/inp_training_norm_arr.npy')
target_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/tar_training_norm_arr.npy')
input = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/input_training.npy')
target = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/target_training.npy')    



plt.figure()
plt.subplot(1,2,1)
plt.imshow(input_transformerd[0, -1, 0,:, :])
plt.colorbar()
plt.title('Input transformed')
plt.subplot(1,2,2)
plt.imshow(input[0, -1,0, :, :])
plt.colorbar()
plt.title('Input non-transformed')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(target_transformerd[0, 0, :, :])
plt.colorbar()
plt.title('Target transformed')
plt.subplot(1,2,2)
plt.imshow(target[0, 0, :, :])
plt.colorbar()  
plt.title('Target non-transformed')


input_old = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/input_training.npy')
# target_old = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180/full_target_deltawtd.npy')
target_subsample = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/target_training.npy')
inpsel = input_old[0, -1:,0, :, :]
# tarsel = target_old[0, :, :]
tarsel_subsample = target_subsample[0, :, :]

limitedInpSel = ['abstraction_uppermost_layer',
                 'bed_conductance_used',
                 'drain_elevation_uppermost_layer',
                 'surface_water_bed_elevation_used',
                 'surface_water_elevation', 
                 'net_RCH',
                 'bottom_uppermost_layer',
                 'drain_conductance',
                 'horizontal_conductivity_uppermost_layer',
                 'primary_storage_coefficient_uppermost_layer',
                 'vertical_conductivity_uppermost_layer',
                 'globgm-wtd'] 
inprange = list(range(0, len(limitedInpSel)))
for i,p in zip(inprange[:], limitedInpSel[:]):
    inpsel = input_old[0, i,0, :, :]
    corr = np.corrcoef(inpsel.flatten(), tarsel_subsample.flatten())[0,1]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(inpsel)
    plt.colorbar()
    plt.title('Input %s, \n corr %2f' %(p, corr)) 
    plt.subplot(1, 3, 2)    
    plt.imshow(tarsel_subsample[0, :, :])
    plt.title('Target wtd')
    plt.colorbar()
    plt.tight_layout
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/src/data_prep/corr_wtd_with_%s.png' % p)

    
input_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/inp_training_norm_arr.npy')
target_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010/tar_training_norm_arr.npy')
limitedInpSel = ['abstraction_uppermost_layer',
                 'bed_conductance_used',
                 'drain_elevation_uppermost_layer',
                 'surface_water_bed_elevation_used',
                 'surface_water_elevation', z
                 'net_RCH',
                 'bottom_uppermost_layer',
                 'drain_conductance',
                 'horizontal_conductivity_uppermost_layer',
                 'primary_storage_coefficient_uppermost_layer',
                 'vertical_conductivity_uppermost_layer',
                 'globgm-wtd']

inprange = list(range(0, len(limitedInpSel)))
for i,p in zip(inprange[:], limitedInpSel[:]):
    inpsel = input_transformerd[0, i,0, :, :]
    corr = np.corrcoef(inpsel.flatten(), target_transformerd[0, 0, :, :].flatten())[0,1]
    print('Correlation between transformed %s and wtd: %s' % (p, corr))
    mininp = min(inpsel.flatten())
    maxinp = max(inpsel.flatten())
    mintar = min(target_transformerd[0, 0, :, :].flatten())
    maxtar = max(target_transformerd[0, 0, :, :].flatten())
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(inpsel)
    plt.colorbar()
    plt.title('Input %s, \n min: %2f, max: %2f' %(p, mininp, maxinp)) 
    plt.subplot(1, 3, 2)    
    plt.imshow(target_transformerd[0, 0, :, :])
    plt.title('Target wtd \n min: %2f, max: %2f' %(mintar, maxtar))
    plt.colorbar()
    plt.tight_layout
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/src/data_prep/corr_wtd_with_%s_transformed.png' % p)


    
input_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/inp_training_norm_arr.npy')
target_transformerd = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/tar_training_norm_arr.npy')
target_train = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/target_training.npy')

plt.figure()
plt.imshow(input_transformerd[0, -1, 0, :, :])
plt.colorbar()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(target_train[0, 0, :, :])
plt.colorbar()
plt.title('Target train delta wtd')
plt.subplot(1, 2, 2)
plt.imshow(target_transformerd[0, 0, :, :])
plt.colorbar()
plt.title('Target transformed delta wtd')

mean = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/full_out_var_mean.npy')
std = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/full_out_var_std.npy')

y_new_denorm = target_transformerd*std+ mean

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(y_new_denorm[0, 0, :, :])
plt.colorbar()
plt.title('Target denorm delta wtd')
plt.subplot(1, 2, 2)
plt.imshow(target_train[0, 0, :, :])
plt.colorbar()
plt.title('Target train delta wtd')




import numpy as np

#order of input data
# limitedInpSel = ['abstraction_uppermost_layer',
#                  'bed_conductance_used',
#                  'drain_elevation_uppermost_layer',
#                  'surface_water_bed_elevation_used',
#                  'surface_water_elevation', 
#                  'net_RCH',
#                  'bottom_uppermost_layer',
#                  'drain_conductance',
#                  'horizontal_conductivity_uppermost_layer',
#                  'primary_storage_coefficient_uppermost_layer',
#                  'vertical_conductivity_uppermost_layer',
#                  'globgm-wtd'] 
input_training = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/input_training.npy')
target_training = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/target_training.npy')

input_subsample_fullrun = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/subsamples/full_input_subsample_0.npy')
target_subsample_fullrun = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_101010_deltawtd/subsamples/full_target_deltawtd_subsample_0.npy')
