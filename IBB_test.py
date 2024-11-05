from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

def main(cfg):

    irisRec = irisRecognition(cfg)


    # Get the list of images to process
    filename_list = []
    image_list = []
    
    extensions = ["bmp", "png", "gif", "jpg", "jpeg", "tiff", "tif"]
    for ext in extensions:
        for filename in glob.glob("./data/*." + ext):
            im = Image.fromarray(np.array(Image.open(filename).convert("RGB"))[:, :, 0], "L")

            image_list.append(im)
            filename_list.append(os.path.basename(filename))

 

    polar_mask_list = []
    polar_image_list = []
    for filename in glob.glob("./images_polar/*.png" ):
        pol_im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        polar_image_list.append(pol_im)
        

    for filename in glob.glob("./masks_polar/*.png" ):
        pol_mask = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        polar_mask_list.append(pol_mask)
    
    #polar_mask_list = []
    #polar_image_list = []
    #for ext in extensions:
     #  for filename in glob.glob("./data/*." + ext):
      #      pol_im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
       #     pol_im = cv2.resize(pol_im, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
            #polar_image_list.append(pol_im)
   
    #for filename in glob.glob("./masks/*.png" ):
     #   pol_mask = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
      #  polar_mask_list.append(pol_mask)


    print(len(polar_image_list))
    print(len(polar_mask_list))


    for P,R in zip([4], [2]):#,8, 12,16,24],  [1,1,1.5,2,3]):#,12,16,24], [1,1,1.5,2,3]):
    
        code_list = []
        for im_polar,mask_polar in zip(polar_image_list,polar_mask_list):
            

            # human-driven BSIF encoding:
            #code = irisRec.extractCode(im_polar)
            # TODO:
            code = irisRec.extractIBBCode(im_polar, mask_polar, R=R, P=P)  #[, mask]) < masks are up to you where to use them

            code_list.append(code)

        with open('results4/resultsIBB' +"P"+ str(P) +"R"+str(R) +"W"+ '.txt', 'w') as f:
            for code1,fn1,i in zip(code_list,filename_list,range(len(code_list))):
                for code2,fn2,j in zip(code_list,filename_list,range(len(code_list))):
                    if i < j:
                        #score, shift = irisRec.matchCodesEfficient(code1, code2, mask1, mask2)
                        # TODO: 
                        score = irisRec.matchIBBCodes(code1, code2) #[, mask1, mask2]) < masks are up to you where to use them
                        #print("{} <-> {} : {:.3f} (mutual rot: {:.2f} deg)".format(fn1,fn2,score,360*shift/irisRec.polar_width))
                        print(fn1, fn2, score)
                        
                        #f.write("{} <-> {} : {:.3f} (mutual rot: {:.2f} deg)".format(fn1,fn2,score,360*shift/irisRec.polar_width))
                        f.write(str(fn1)+" "+ str(fn2)+" "+ str(score))
                        f.write("\n")
    return None     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))