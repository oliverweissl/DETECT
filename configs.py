gan_facial_ckpt_path = "local_models/generators/stylegan2-ffhq-1024x1024.pkl"
gan_imagenet_ckpt_path = "local_models/generators/imagenet128.pkl"
gan_afhq_ckpt_path = "local_models/generators/stylegan3-r-afhqv2-512x512.pkl"
gan_lsun_dog_ckpt_path = "local_models/generators/stylegan2-lsundog-256x256.pkl"
gan_cat_ckpt_path = "local_models/generators/stylegan2-afhqcat-512x512.pkl"
gan_car_ckpt_path = "local_models/generators/stylegan2-car-config-f.pkl"

generate_image_base_dir = "."

segmentation_facial_ckpt_path = "local_models/segmentation/face_segmentation.pth"

sut_facial_parallel_path = '/tmp/pycharm_project_181/local_models/classifiers/checkpoints/resnet_celeb_40_parallel.pth'
sut_facial_path = '/tmp/pycharm_project_181/local_models/classifiers/checkpoints/resnet_celeb_40_single.pth'
sut_facial_large_parallel_path = '/tmp/pycharm_project_181/local_models/classifiers/checkpoints/swag_celeb_40_parallel.pth'
sut_facial_large_path = '/tmp/pycharm_project_181/local_models/classifiers/checkpoints/swag_celeb_40_single.pth'


sut_dog_path = 'local_models/classifiers/checkpoints/dogs_triplet_rexnet_150_best_model.pth'

celeba_attributes = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]

celeba_attributes_dict = {
    0: '5_o_Clock_Shadow',
    1: 'Arched_Eyebrows',
    2: 'Attractive',
    3: 'Bags_Under_Eyes',
    4: 'Bald',
    5: 'Bangs',
    6: 'Big_Lips',
    7: 'Big_Nose',
    8: 'Black_Hair',
    9: 'Blond_Hair',
    10: 'Blurry',
    11: 'Brown_Hair',
    12: 'Bushy_Eyebrows',
    13: 'Chubby',
    14: 'Double_Chin',
    15: 'Eyeglasses',
    16: 'Goatee',
    17: 'Gray_Hair',
    18: 'Heavy_Makeup',
    19: 'High_Cheekbones',
    20: 'Male',
    21: 'Mouth_Slightly_Open',
    22: 'Mustache',
    23: 'Narrow_Eyes',
    24: 'No_Beard',
    25: 'Oval_Face',
    26: 'Pale_Skin',
    27: 'Pointy_Nose',
    28: 'Receding_Hairline',
    29: 'Rosy_Cheeks',
    30: 'Sideburns',
    31: 'Smiling',
    32: 'Straight_Hair',
    33: 'Wavy_Hair',
    34: 'Wearing_Earrings',
    35: 'Wearing_Hat',
    36: 'Wearing_Lipstick',
    37: 'Wearing_Necklace',
    38: 'Wearing_Necktie',
    39: 'Young'
}
dog_rexnet_dict = {0: 'dog - vizsla',
                   1: 'dog - labradoodle',
                   2: 'dog - borzoi',
                   3: 'dog - beagle',
                   4: 'dog - cavalier king charles spaniel',
                   5: 'dog - japanese spaniel',
                   6: 'dog - bull mastiff',
                   7: 'dog - elk hound',
                   8: 'dog - rottweiler',
                   9: 'dog - german shepherd',
                   10: 'dog - pekinese',
                   11: 'dog - shih-tzu',
                   12: 'dog - corgi',
                   13: 'dog - chihuahua',
                   14: 'dog - collie',
                   15: 'dog - mex hairless',
                   16: 'dog - sheepdog-dog - shetland',
                   17: 'dog - border collie',
                   18: 'dog - irish spaniel',
                   19: 'dog - cocker',
                   20: 'dog - clumber',
                   21: 'dog - yorkie',
                   22: 'dog - basset',
                   23: 'dog - afghan',
                   24: 'dog - labrador',
                   25: 'dog - bernese mountain',
                   26: 'dog - saint bernard',
                   27: 'dog - poodle',
                   28: 'dog - rhodesian',
                   29: 'dog - bluetick',
                   30: 'dog - bichon frise',
                   31: 'dog - doberman',
                   32: 'dog - cockapoo',
                   33: 'dog - golden retriever',
                   34: 'dog - bermaise',
                   35: 'dog - boxer',
                   36: 'dog - american hairless',
                   37: 'dog - akita',
                   38: 'dog - shar pei',
                   39: 'dog - irish wolfhound',
                   40: 'dog - maltese',
                   41: 'dog - dachshund',
                   42: 'dog - greyhound',
                   43: 'dog - airedale',
                   44: 'dog - blenheim',
                   45: 'dog - french bulldog',
                   46: 'dog - siberian husky',
                   47: 'dog - basenji',
                   48: 'dog - cairn',
                   49: 'dog - bloodhound',
                   50: 'dog - dalmatian',
                   51: 'dog - dingo',
                   52: 'dog - pomeranian',
                   53: 'dog - dhole',
                   54: 'dog - groenendael',
                   55: 'dog - schnauzer',
                   56: 'dog - great dane',
                   57: 'dog - american spaniel',
                   58: 'dog - labrador retriever',
                   59: 'dog - pit bull',
                   60: 'dog - chow',
                   61: 'dog - scotch terrier',
                   62: 'dog - lhasa',
                   63: 'dog - shiba inu',
                   64: 'dog - great perenees',
                   65: 'dog - newfoundland',
                   66: 'dog - chinese crested',
                   67: 'dog - havanese',
                   68: 'dog - african wild dog',
                   69: 'dog - bull terrier',
                   70: 'dog - boston terrier',
                   71: 'dog - bulldog',
                   72: 'dog - pug',
                   73: 'dog - coyote',
                   74: 'dog - komondor',
                   75: 'dog - malinois',
                   76: 'dog - aspin',
                   77: 'dog - flat coated retriever'}
cat_rexnet_dict = {0: 'cat - egyptian mau', 1: 'cat - bengal',
                   2: 'cat - maine coon', 3: 'cat - russian blue',
                   4: 'cat - bombay', 5: 'cat - scottish fold',
                   6: 'cat - birman', 7: 'cat - british shorthair',
                   8: 'cat - abyssinian', 9: 'cat - sphynx',
                   10: 'cat - ragdoll', 11: 'cat - siamese',
                   12: 'cat - american shorthair', 13: 'cat - persian'}

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
class ToTensorIfNeeded:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype in (np.float32, np.float64):
                x = np.clip(x * 255, 0, 255).astype(np.uint8)
            x = Image.fromarray(x)  # Convert NumPy to PIL first
        return transforms.ToTensor()(x)

preprocess_imagenet_classifier = transforms.Compose([
    ToTensorIfNeeded(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# preprocess are the same since backbones are identical
preprocess_celeb_classifier = transforms.Compose([
                ToTensorIfNeeded(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
preprocess_celeb_large_classifier = transforms.Compose([
                ToTensorIfNeeded(),
                transforms.Resize((518, 518)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

preprocess_rexnet_classifier = transforms.Compose([
                ToTensorIfNeeded(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])