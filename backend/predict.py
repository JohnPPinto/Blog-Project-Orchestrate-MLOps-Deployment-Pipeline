import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from BLIP.models.blip import blip_decoder
from BLIP.models.blip_vqa import blip_vqa

IMAGE_SIZE = 384
CAPTION_MODEL_PATH = 'model_base_capfilt_large.pth'
VQA_MODEL_PATH = 'model_base_vqa_capfilt_large.pth'

# Setting torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image, image_size, device):
    # Creating a RGB image instance
    raw_image = image.convert('RGB')

    # Transforming the image as per the model
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), 
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    tensor_img = transform(raw_image).unsqueeze(0).to(device)
    
    return tensor_img

def predict(image,
            type='caption',
            decode_method='beam',
            question = None,
            img_size=IMAGE_SIZE, 
            device=device):
    """
    Predict caption or answer for any image.

    Parameters:
        image: A PIL opened image.
        type: A string for the type of prediction, either 'caption' or 'vqa'.
        decode_method (default: 'beam'): A decoding method for the predicted caption text, 
                                         either 'beam' or 'nucleus'.
        question: (default: None): A string containing question for the vqa model.
        img_size (default: 384): A int for the input image size that the model accepts.
        device (default: device): A torch.device function for loading the data on 'CPU' or 'GPU'.

    Returns:
        Dict( prediction: A string containing the predicted caption or answer for the image. )
    """
    # Loading the image
    tensor_img = load_image(image=image, 
                            image_size=img_size, 
                            device=device)

    # Loading the model
    if type == 'caption':
        model = blip_decoder(pretrained=CAPTION_MODEL_PATH,
                             image_size=img_size,
                             vit='base')
    elif type == 'vqa':
        model = blip_vqa(pretrained=VQA_MODEL_PATH,
                         image_size=img_size,
                         vit='base')
    model.eval()
    model = model.to(device)

    # Performing prediction
    with torch.no_grad():
        
        # Predicting caption
        if type == 'caption':
        
            # Beam Search
            if decode_method == 'beam':
                caption = model.generate(tensor_img, 
                                         sample=False, 
                                         num_beams=3, 
                                         max_length=20, 
                                         min_length=5)[0]
        
            # Nucleus sampling
            elif decode_method == 'nucleus':
                caption = model.generate(tensor_img, 
                                         sample=True, 
                                         top_p=0.9, 
                                         max_length=20, 
                                         min_length=5)[0]
            else:
                return '[ERROR] Check your decode method. Check the docstring for any clarification'
            
            return {'prediction': caption.capitalize()}
        
        # Predicting VQA
        elif type == 'vqa' and question is not None:
            answer = model(tensor_img, 
                           question, 
                           train=False, 
                           inference='generate')[0]

            return {'prediction': answer.capitalize()}
        
        else:
            return '[ERROR] Check your type or questions. Check the docstring for any clarification'
        
if __name__ == '__main__':
    IMAGE_PATH = 'DIVO1.jpg'
    image = Image.open(IMAGE_PATH)

    # Predicting the image
    caption = predict(image=image,
                      type='caption',
                      decode_method='beam')
    print(caption)
