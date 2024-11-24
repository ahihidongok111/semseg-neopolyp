import argparse
from segmentation_models_pytorch import UnetPlusPlus
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def load_model(model_path):
    model = UnetPlusPlus(
        encoder_name="efficientnet-b7",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3     
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model

def mask_to_rgb(mask):
    color_dict= {0: (0, 0, 0),
                 1: (255, 0, 0),
                 2: (0, 255, 0)}
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)    

def img_segmentation(model, img_path, transform, device):
    img_size = 256
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (img_size, img_size))
    transformed = transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_segmented.png", mask_rgb) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path of the test image"
    )
    args = parser.parse_args()

    img_path = args.image_path
    model_path = "checkpoint/unet_sce_40_epochs.pth"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path).to(device)

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    img_segmentation(model, img_path, transform, device)
    print("Inference completed! The segmented image is saved in test_segmented.png.")




    
    



