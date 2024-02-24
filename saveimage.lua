require 'image'
local testdataset={}
testdataset=torch.load('combined_testLR.bin')
testdataset_data=testdataset.data
testdataset_label=testdataset.label

ss1=testdataset_label:size()[1]
ss2=testdataset_label:size()[2]
ss3=testdataset_label:size()[3]
ss4=testdataset_label:size()[4]

s1=testdataset_data:size()[1]
s2=testdataset_data:size()[2]
s3=testdataset_data:size()[3]
s4=testdataset_data:size()[4]




--local image_tensor=torch.Tensor(s1,s2,s3,s4)
local image_tensordata=torch.zeros(3,24,24)
local image_tensorlabel=torch.zeros(3,48,48)


--for i=1,s do
temp1=testdataset_data[5]
temp2=testdataset_label[5]
image_tensordata:copy(temp1)
image_tensorlabel:copy(temp2)
image.save('test1LR.png', image.toDisplayTensor(image_tensordata))
image.save('test1HR.png', image.toDisplayTensor(image_tensorlabel))

local output={}
output=torch.load('sampleoutput.bin')
output_data=output

sss1=output_data:size()[1]
sss2=output_data:size()[2]
sss3=output_data:size()[3]
sss4=output_data:size()[4]

--local image_tensor=torch.Tensor(s1,s2,s3,s4)
local image_tensorout=torch.zeros(3,48,48)
--for i=1,s do
temp3=output_data[5]
image_tensorout:copy(temp3)
image.save('sampleoutput.png', image.toDisplayTensor(image_tensorout))

local targets={}
targets=torch.load('targets.bin')
targets_data=targets

ssss1=targets_data:size()[1]
ssss2=targets_data:size()[2]
ssss3=targets_data:size()[3]
ssss4=targets_data:size()[4]

--local image_tensor=torch.Tensor(s1,s2,s3,s4)
local image_tensortargets=torch.zeros(3,48,48)
--for i=1,s do
temp4=targets_data[5]
image_tensortargets:copy(temp4)
image.save('sampletargets.png', image.toDisplayTensor(image_tensortargets))

require 'image'
require 'nn'

-- PSNR 
true_frame=image.load('sampletargets.png')
pred=image.load('sampleoutput.png')
local eps = 0.0001
  -- if true_frame:size(1) == 1 then true_frame = true_frame[1] end
  -- if pred:size(1) == 1 then pred = pred[1] end

   local prediction_error = 0
   for i = 1, pred:size(2) do
          for j = 1, pred:size(3) do
            for c = 1, pred:size(1) do
            -- put image from -1 to 1 to 0 and 255
            prediction_error = prediction_error +
              (pred[c][i][j] - true_frame[c][i][j])^2
            end
          end
   end
   --MSE
   prediction_error=128*128*prediction_error/(pred:size(1)*pred:size(2)*pred:size(3))

   --PSNR
   if prediction_error>eps then
      prediction_error = 10*torch.log((255*255)/ prediction_error)/torch.log(10)
   else
      prediction_error = 10*torch.log((255*255)/ eps)/torch.log(10)
   end
   print('PSNR='..prediction_error..'dB')
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
-- SSIM
img1=true_frame;
img2=pred;

if img1:size(1) > 2 then
    img1 = image.rgb2y(img1)
    img1 = img1[1]
    img2 = image.rgb2y(img2)
    img2 = img2[1]
   end



   -- place images between 0 and 255.
   img1:add(1):div(2):mul(255)
   img2:add(1):div(2):mul(255)

   local K1 = 0.01;
   local K2 = 0.03;
   local L = 255;

   local C1 = (K1*L)^2;
   local C2 = (K2*L)^2;
   local window = image.gaussian(11, 1.5/11,0.0708);

   local window = window:div(torch.sum(window));

   local mu1 = image.convolve(img1, window, 'full')
   local mu2 = image.convolve(img2, window, 'full')

   local mu1_sq = torch.cmul(mu1,mu1);
   local mu2_sq = torch.cmul(mu2,mu2);
   local mu1_mu2 = torch.cmul(mu1,mu2);

   local sigma1_sq = image.convolve(torch.cmul(img1,img1),window,'full')-mu1_sq
   local sigma2_sq = image.convolve(torch.cmul(img2,img2),window,'full')-mu2_sq
   local sigma12 =  image.convolve(torch.cmul(img1,img2),window,'full')-mu1_mu2

   local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)) ,
     torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
   local mssim = torch.mean(ssim_map);

print('SSIM='..mssim.. 'of 1')





