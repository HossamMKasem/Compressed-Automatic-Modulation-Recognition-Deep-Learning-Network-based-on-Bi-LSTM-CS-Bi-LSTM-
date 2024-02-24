require ('image')
f=torch.load('output_5.bin')
ff=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/3-Natural/1-R/1-0.5/combined_test_half_initial.bin')

image.save('1-sample_Modified_ST_VDSR_Translated.png',image.toDisplayTensor(f[441]))
image.save('1-Input_Modified_ST_VDSR_Translated.png',image.toDisplayTensor(ff.data[441]))
image.save('1-Target_Modified_ST_VDSR_Translated.png',image.toDisplayTensor(ff.label[441]))

