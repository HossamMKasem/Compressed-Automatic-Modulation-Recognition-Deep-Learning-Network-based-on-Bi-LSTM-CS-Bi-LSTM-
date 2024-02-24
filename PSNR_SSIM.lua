require 'image'
combined_testLR=torch.load('combined_testLR.bin')
combined_testLR_data=combined_testLR.data
s1=combined_testLR_data:size()[1]
no_image=s1
local PSNR_valus=torch.zeros(no_image,1)
local SSIM_valus=torch.zeros(no_image,1)
total_PSNR_values=0
total_SSIM_values=0




true_frame=image.load('1-Target_Modified_ST_VDSR_Translated.png')
pred=image.load ('1-sample_Modified_ST_VDSR_Translated.png')
-- PSNR 

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
   
   
   PSNR_valus=prediction_error
 total_PSNR_values=total_PSNR_values+prediction_error
 --  print('PSNR='..prediction_error..'dB')
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

-- print('SSIM='..mssim.. 'of 1')
SSIM_valus=mssim
total_SSIM_values=total_SSIM_values+mssim



 average_PSNR=total_PSNR_values/no_image;
average_SSIM=total_SSIM_values/no_image;
print('PSNR='..average_PSNR..'dB')
print('SSIM='..average_SSIM.. 'of 1')

torch.save('PSNR_valus.bin',PSNR_valus)
torch.save('average_PSNR.bin',average_PSNR)
torch.save('SSIM_valus.bin',SSIM_valus)
torch.save('average_SSIM.bin',average_SSIM)