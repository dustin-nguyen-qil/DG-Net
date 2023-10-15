from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from utils import get_config, get_target_images, get_target_loader
from PIL import Image
from trainer import DGNet_Trainer, to_gray
from datasets.dataloader import vid_dataset_loader
from visual_tools.test_folder import recover, fliplr
from tqdm.auto import tqdm
import pickle, random 
from datasets.dataset import VidDataset

checkpoint_gen = 'outputs/E0.5new_reid0.5_w30000/checkpoints/gen_00100000.pt'
checkpoint_id = 'outputs/E0.5new_reid0.5_w30000/checkpoints/id_00100000.pt'
config = get_config('outputs/E0.5new_reid0.5_w30000/config.yaml')
output_folder = 'outputs/vccr'
seed = 10

trainer = DGNet_Trainer(config)

state_dict_gen = torch.load(checkpoint_gen)
trainer.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
trainer.gen_b = trainer.gen_a

state_dict_id = torch.load(checkpoint_id)
trainer.id_a.load_state_dict(state_dict_id['a'])
trainer.id_b = trainer.id_a

trainer.cuda()
trainer.eval()

encode = trainer.gen_a.encode
style_encode = trainer.gen_a.encode
id_encode = trainer.id_a
decode = trainer.gen_a.decode

data_transforms = transforms.Compose([
        transforms.Resize(( config['crop_image_height'], config['crop_image_width']), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path = 'data/vccr/train.pkl'

vid_dataset = VidDataset(data_path, data_transforms)
vid_loader_a = DataLoader(vid_dataset, batch_size=1, shuffle=False)

torch.manual_seed(seed)

"""
    How many times per tracklet?
    New evaluation protocol?
"""

with torch.inference_mode():
    for tracklet_a in tqdm(vid_loader_a):
        imgs, img_paths, p_id, cam_id, clothes_id = tracklet_a
        target_images = get_target_images(data_path, p_id)
        target_loader = get_target_loader(target_images, data_transforms)
        for img in imgs:
            pass 






