require 'image'
require 'nn'
true_frame=image.load('1-Target_Modified_ST_VDSR_Translated.png')
pred=image.load('1-sample_Modified_ST_VDSR_Translated.png')
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
   print(prediction_error)











