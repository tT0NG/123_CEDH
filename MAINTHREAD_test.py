import sys
sys.path.insert(0, "./model")

import argparse
import time
import glob
from torch.autograd import Variable
import torch
from utils.utils import *
import urllib


print("*****************************************\n"
      "This is the test code for NTIRE19-Dehaze\n"
      "Pre-request: python-3.6 and pytorch-1.0\n"
      "*****************************************\n"
      "Note: the original model is trained on GPU.\n"
      "The conversion between CPU and GPU model can generate the precesion error.\n"
      "Please keep the original setups and evaluate the network on GPU to reproduce the exact results.\n"
      "Please ignore UserWarning if there is any, which is caused by pytorch updating issues.")

parser = argparse.ArgumentParser(description="Pytorch 123_COLOR_model Evaluation")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda? Default is True")
parser.add_argument("--model", type=str, default="COLOR_model", help="model path")
parser.add_argument("--test", type=str, default="testset", help="testset path")
opt = parser.parse_args()
cuda = opt.cuda
device_label = 'GPU' if opt.cuda else 'CPU'

if cuda and not torch.cuda.is_available():
    raise Exception(">>No GPU found, please run without --cuda")

if not cuda:
    print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
          ">>To run on GPU, please run the script with --cuda option")

save_path = 'result_{}_{}'.format(opt.model,device_label)
checkdirctexist(save_path)

model_path = os.path.join('model', "{}.pth".format(opt.model))

if not os.path.exists(model_path):
    print("*The trained model is not downloaded*\nPlease visit: https://drive.google.com/file/d/1UXsd1hob9XhNxSIitd1KLoI-UnqL9THQ/view?usp=sharing\n to download the trained model and copy the pth file to ./model folder\n")

model = torch.load(model_path)["model"]

image_list = glob.glob(os.path.join(opt.test, '*.png'))

print(">>Start testing...\n"
      "\t\t Model: {0}\n"
      "\t\t Test on: {1}\n"
      "\t\t Results save in: {2}".format(opt.model, opt.test, save_path))

avg_elapsed_time = 0.0
count = 0
for image_name in image_list:
    count += 1
    print(">>Processing ./{}".format(image_name))
    im_input, W, H = get_image_for_test(image_name)

    with torch.no_grad():
        im_input = Variable(torch.from_numpy(im_input).float())
        if cuda:
            model = model.cuda()
            model.train(False)
            im_input = im_input.cuda()
        else:
            im_input = im_input.cpu()
            model = model.cpu()
        model.train(False)
        model.eval()
        start_time = time.time()
        # feeding forward
        im_output = model(im_input)
        # compute running time
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

    im_output = im_output.cpu()
    im_output_forsave = get_image_for_save(im_output)
    path, filename = os.path.split(image_name)
    im_output_forsave = im_output_forsave[0:H, 0:W, :]
    cv2.imwrite(os.path.join(save_path, filename), im_output_forsave)

print(">>Finished!"
      "It takes average {}s for processing single image on {}\n"
      "Results are saved at ./{}".format(avg_elapsed_time / count, device_label, save_path))

