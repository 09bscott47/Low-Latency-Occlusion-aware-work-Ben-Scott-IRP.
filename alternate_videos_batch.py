# we can just have multiple dataloaders and just alternate between them.

import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
# device='cpu'
# replay_buffer = torch.load('full_replay_buffer_no_img.pt')
# print("h")
#
# # Usage:
# for frame in replay_buffer:
#     img = frame['img'].to(device)
#     mem = frame['mem_embedding'].to(device)
#     boxes = frame['gt_boxes']
#     ids = frame['trk_ids']

# ──────────────────────────────────
# 1) Your IAFE module
# ─────────────────────────────────────────────
# class DummyIAFE(nn.Module):
#     def __init__(self, feature_dim, embed_dim):
#         super().__init__()
#         self.fc    = nn.Linear(embed_dim, feature_dim)
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, memory_embeddings, feature_map):
#         # memory_embeddings: [N, D], feature_map: [B, C, H, W]
#         enhancement = self.fc(memory_embeddings)    # [N, C]
#         enhancement = enhancement.mean(0)           # [C]
#         enhancement = enhancement.view(1, -1, 1, 1) # [1, C, 1, 1]
#         return feature_map + self.alpha * enhancement
######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D] with N = H*W
        return x, (H, W)

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, height * width, dim))

    def forward(self, x):
        return x + self.pos

class CrossSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, identity_embed):
        B, N, D = x.shape
        identity = identity_embed.expand(B, 1, D)  # [B, 1, D]
        identity = identity.repeat(1, N, 1)        # [B, N, D]

        # Cross-Attention
        z1, _ = self.cross_attn(query=x, key=identity, value=identity)
        x = self.norm1(x + z1)

        # Self-Attention
        z2, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + z2)

        # Feedforward
        x = self.norm3(x + self.ffn(x))
        return x

class IAFETransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, num_heads=4, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.layers = nn.ModuleList([
            CrossSelfAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.deproj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, feature_map, memory_bank):
        """
        feature_map: [1, C, H, W]
        memory_bank: dict[int, Tensor[C]]
        """
        B, C, H, W = feature_map.shape
        x, (Hp, Wp) = self.patch_embed(feature_map)  # [B, N, D]
        x = PositionalEncoding2D(x.shape[-1], Hp, Wp)(x)

        # enhancement_total = torch.zeros_like(feature_map)
        # #
        # if memory_bank:
        #     for identity_vec in memory_bank.values():
        #         identity_embed = identity_vec.unsqueeze(0)  # [1, D]
        #         x_id = x.clone()
        #
        #         for layer in self.layers:
        #             x_id = layer(x_id, identity_embed)  # [B, N, D]
        #
        #         x_id = x_id.transpose(1, 2).reshape(B, -1, Hp, Wp)  # [B, D, Hp, Wp]
        #         enhancement = self.deproj(x_id)  # [B, C, H, W]
        #         enhancement_total += enhancement
        # else:
        #     # Fallback: no memory, no enhancement
        #     enhancement_total = torch.zeros_like(feature_map)
        #
        # return feature_map + enhancement_total

        if memory_bank:
            identity_embed = torch.stack(list(memory_bank.values())).mean(dim=0, keepdim=True)  # [1, D]
        else:
            identity_embed = torch.zeros((1, x.shape[-1]), device=feature_map.device)

        for layer in self.layers:
            x = layer(x, identity_embed)  # [B, N, D]

        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)  # [B, D, Hp, Wp]
        enhancement = self.deproj(x)  # [B, C, H, W]

        return feature_map + enhancement

# ───────────────────────────────────────────────
# 2) Build your custom Dataset
# ───────────────────────────────────────────────────────
class VisDroneYoloDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=1280):
        self.imgs = sorted(Path(img_dir).rglob("*.jpg"))
        self.lbls = [Path(lbl_dir) / p.name.replace(".jpg", ".txt") for p in self.imgs]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")

        img = self._resize(img, self.imgsz)  #    FIXED

        img_tensor = TF.to_tensor(img)

        boxes = []
        if self.lbls[idx].exists():
            for line in open(self.lbls[idx]):
                cls, trk_id, xc, yc, w, h = map(float, line.split())
                boxes.append([cls, trk_id, xc, yc, w, h])
        target = torch.tensor(boxes) if boxes else torch.zeros((0,6))

        if target.numel():
            cls = target[:,0].long()
            trk_id = target[:, 1].long()
            xc, yc, w_, h_ = target[:,2:].unbind(1)
            # x1 = xc - w_/2; y1 = yc - h_/2
            # x2 = xc + w_/2; y2 = yc + h_/2
            boxes = torch.stack([xc, yc, w_, h_], dim=1)
        else:
            cls   = torch.zeros(0, dtype=torch.long)
            trk_id = torch.zeros(0, dtype=torch.long)
            boxes = torch.zeros((0,4), dtype=torch.float32)

        return img_tensor, {'cls':cls,'trk_id':trk_id,'boxes':boxes, 'img_size':torch.tensor([self.imgsz,self.imgsz])}


    @staticmethod
    def _resize(img, new_size):
        return img.resize((new_size, new_size), Image.BILINEAR)


# ──────────────────────────────────────────────────
# 3) Instantiate YOLO hook in IAFE
# ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################ NEW 24/06
from ultralytics import YOLO

# 1. Instantiate your base YOLO model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.

# 2. Load weights into the YOLO submodel (not the wrapper)
model.model.load_state_dict(torch.load('yolov8_finetune_weights_epoch_3.pt', map_location='cpu'))
model.model.eval()
# 3. Instantiate your IAFE module (with same config as training)
iafe = IAFETransformer(
    in_channels=64,
    embed_dim=64,
    patch_size=8,#was 8. 4 was tried also
    num_heads=4,
    num_layers=4
)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name:50s} {param.numel():,}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() )

print(f"Total trainable parameters: {count_parameters(model):,}")


# # 4. Load IAFE weights
# iafe.load_state_dict(torch.load('iafe_weights_alternate_3_epoch_9.pt', map_location='cpu')) ##########################
iafe.train()
# def iafe_hook(module, inp, out):
#     # memory_embedding = torch.randn(5, 64, device=out.device)
#     enhanced = iafe(out,memory_bank)
#     feature_cache['feature_map'] = out.detach()
#
#     return enhanced
#
# hook_handle=model.model.model[4].register_forward_hook(iafe_hook)
##################################################



# load and freeze YOLO
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8l.pt')
model.model.eval()#

for p in model.model.parameters():
    p.requires_grad = True

# figure out channel count at layer 4
feat_store = {}
def _probe_hook(m, inp, out):
    feat_store['F'] = out#.detach()#####################################################
h = model.model.model[4].register_forward_hook(_probe_hook)
# results = model(torch.randn(1, 3, 640, 640))  #    handles training/inference logic safely

results = model.model(torch.randn(1, 3, 1280, 1280))  #   

# _ = model.model(torch.randn(1,3,640,640))
h.remove()


C = feat_store['F'].shape[1]
# iafe = DummyIAFE(feature_dim=C, embed_dim=64).to(device)
# iafe = IAFETransformer(in_channels=C, embed_dim=64, patch_size=8).to(device)
# enhanced_map = iafe(feature_map, memory_bank)

# iafe.train()
# Save a copy of the initial IAFE weights (deep copy)
# iafe.load_state_dict(torch.load(r'iafe_weights_alternate_3_epoch_7.pt', map_location='cpu')) #####################################
iafe.train()#eval()
def iafe_hook(module, inp, out):
    # memory_embedding = torch.randn(5, 64, device=out.device)
    enhanced = iafe(out,memory_bank)
    feature_cache['feature_map'] = out.detach()

    return enhanced

hook_handle=model.model.model[4].register_forward_hook(iafe_hook)
print(f"loading worked,{model.model.model[4]}")
# initial_iafe_weights = {k: v.clone().detach() for k, v in iafe.state_dict().items()}


import torch

def get_tracklet_embeddings_from_feature_map(feature_map, tracklet_boxes, img_size=1280):
    """
    Extracts per-tracklet Re-ID vectors by cropping YOLO feature map using GT boxes.

    Args:
        feature_map (Tensor): [1, C, H, W] YOLO feature map for a single image
        tracklet_boxes (Dict[int, Tensor]): {track_id: box} in normalized [xc, yc, w, h]
        img_size (int): assumed square input image size (e.g. 640 or 1280)

    Returns:
        Dict[int, Tensor]: {track_id: reid_embedding}, each embedding is [C] normalized
    """
    assert feature_map.shape[0] == 1, "Function only supports batch size 1."
    _, C, H, W = feature_map.shape
    fmap = feature_map[0]  # [C, H, W]

    tracklet_embeddings = {}

    for track_id, box in tracklet_boxes.items():
        # Convert [xc, yc, w, h] → [x1, y1, x2, y2] in absolute coords
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * W)
        y1 = int((yc - bh / 2) * H)
        x2 = int((xc + bw / 2) * W)
        y2 = int((yc + bh / 2) * H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            emb = torch.zeros(C, device=feature_map.device)
        else:
            crop = fmap[:, y1:y2, x1:x2]  # [C, h_crop, w_crop]
            emb = crop.mean(dim=(1, 2))  # [C]
            emb = emb / (emb.norm(p=2) + 1e-6)

        tracklet_embeddings[track_id] = emb

    return tracklet_embeddings


feature_cache = {}

# def iafe_hook(module, inp, out):
#     # memory_embedding = torch.randn(5, 64, device=out.device)
#     enhanced = iafe(out,memory_bank)
#     feature_cache['feature_map'] = out.detach()
#
#     return enhanced
#
# hook_handle = model.model.model[4].register_forward_hook(iafe_hook)

# Right after you register the IAFE hook:
raw_store = {}

# 1) grab the Detect module (last in the seq)
detect_module = model.model.model[-1]
# model.model.train()

# 2) define a hook that saves its output (pre-NMS Tensors)
def detect_hook(module, inp, out):
    # 'out' here is the list/tuple of raw head outputs
    raw_store['preds'] = out

# 3) register it
detect_handle = detect_module.register_forward_hook(detect_hook)

# ────────────────────────────────────────────────
# 4) Prepare DataLoaders
# ───────────────────────────────────────────────────────────
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\86"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\86"
dataset_86  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_86 = DataLoader(
    dataset_86,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
#################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\117"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\117"

dataset_117  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_117 = DataLoader(
    dataset_117,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
#################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\137"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\137"
dataset_137  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_137 = DataLoader(
    dataset_137,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\182"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\182"
dataset_182  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_182 = DataLoader(
    dataset_182,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\268"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\268"

dataset_268  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_268 = DataLoader(
    dataset_268,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\305"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\305"

dataset_305  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_305 = DataLoader(
    dataset_305,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
#################################################
###############################
IMG_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_images\yolo_images\train\imgs_train\339"
LBL_DIR = r"C:\Users\cjsco\Downloads\training_smart\yolo_labels_w_ids\yolo_labels_w_ids\train\lbls\339"

dataset_339  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280) ####### We can have multiple datsets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
collate_fn=lambda x: list(zip(*x))
dataloader_339 = DataLoader(
    dataset_339,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)
#################################################
optimizer = torch.optim.Adam(iafe.parameters(), lr=1e-4)

def convert_batch_to_loss_format(batch_tuple):
    batch_idx = []
    cls_list = []
    trk_id_list=[]
    bboxes_list = []

    for i, sample in enumerate(batch_tuple):
        cls = sample['cls'].long()  # (N,)
        trk_id=sample['trk_id'].long()
        boxes = sample['boxes'].float()  # (N, 4), assumed normalized [x1, y1, x2, y2]
        num_boxes = cls.shape[0]

        if num_boxes == 0:
            continue  # skip empty samples

        batch_idx.append(torch.full((num_boxes,), i, dtype=torch.long))
        cls_list.append(cls)
        trk_id_list.append(trk_id)
        bboxes_list.append(boxes)

    if not cls_list:  # all images had no GTs
        return {
            'batch_idx': torch.empty(0, dtype=torch.long),
            'cls': torch.empty(0, dtype=torch.long),
            'trk_id': torch.empty(0, dtype=torch.long),
            'bboxes': torch.empty(0, 4, dtype=torch.float),
        }

    return {
        'batch_idx': torch.cat(batch_idx),
        'cls': torch.cat(cls_list),
        'trk_id': torch.cat(trk_id_list),
        'bboxes': torch.cat(bboxes_list),
    }

print("tried plotting detections abvove")


print("training:")
# ────────────────────────────────────────────
# 5) Training loop
# ───────────────────────────────────────────────────────────
replay_buffer = []
from ultralytics.utils.loss import v8DetectionLoss
memory_bank={} #dkkkkkkkkkkk if here is the correct place!!!!!!!!!!!!!!!!!!!
from itertools import cycle

dataloaders = [dataloader_86,dataloader_117,dataloader_137,dataloader_182,dataloader_268,dataloader_305,dataloader_339]
iters = [cycle(dl) for dl in dataloaders]
total_training_steps = 100
for epoch in range(10):
    memory_bank_86 = {}
    memory_bank_268 = {}
    memory_bank_137 = {}
    memory_bank_182 = {}
    memory_bank_305 = {}
    memory_bank_117 = {}
    memory_bank_339= {}
    mem_banks=[memory_bank_86,memory_bank_268,memory_bank_137,memory_bank_182,memory_bank_305,memory_bank_117,memory_bank_339]
    epoch_loss=0
    for i in range(total_training_steps):
        # loss_total=torch.tensor([0.00])
        optimizer.zero_grad() #############
        for step in range(len(dataloaders)):
            # optimizer.zero_grad() ##############
            idx = step % len(dataloaders)  # or use round-robin logic
            imgs_list, targets_list = next(iters[idx])
            memory_bank=mem_banks[idx]
            imgs = torch.stack(imgs_list, 0).to(device)
            for t in targets_list:
                t['cls'] = t['cls'].to(device)
                t['trk_id'] = t['trk_id'].to(device)
                t['boxes'] = t['boxes'].to(device)
                t['img_size'] = t['img_size'].to(device)

            _ = model.model(imgs)  # fires both hooks: just one model originally ###############################

            raw_preds = raw_store['preds']  # pure Tensors


            head_outputs = raw_preds[1]

            loss_fn = v8DetectionLoss(model.model)

            batch = convert_batch_to_loss_format(targets_list)
            # batch['images'] = imgs  #    Add this ######################################### new 20/06 COMENTED OUT, NEED THIS IF WANT TO PLOTTTTTTTTTT
            loss_output = loss_fn(raw_preds, batch)  # batch_obj
            loss_total = loss_output[0]  # this is differentiable
            loss_total.backward(retain_graph=False)

            loss_additive = loss_total.detach().item() ######### added .detach()
            epoch_loss += loss_additive
            raw_store['preds'] = None     ####################
            del raw_preds, loss_output, loss_total #######################
            torch.cuda.empty_cache()  # Optional ############################

            # loss_total.backward()
            # optimizer.zero_grad()
            # optimizer.step()  ############################################################################################
            # print("stepping")
            ##Now, I want to update E_{t,k} with ground truth BBs.
            with torch.no_grad(): ############################### added this
                bboxes_from_targets = [t['boxes'] for t in targets_list]
                trk_box_dict = {}
                for t in targets_list:
                    trk_ids = t['trk_id']
                    boxes = t['boxes']
                    if trk_ids.numel() == 0:
                        continue
                    for track_id, box in zip(trk_ids.tolist(), boxes):
                        trk_box_dict[track_id] = box  # if track_id seen multiple times, last one wins

                mem_embed_R = get_tracklet_embeddings_from_feature_map(feature_cache['feature_map'], trk_box_dict)

                ema_decay = 0.95
                # Update memory via EMA
                for tid, emb in mem_embed_R.items():
                    if tid in memory_bank:
                        memory_bank[tid] = ema_decay * memory_bank[tid] + (1 - ema_decay) * emb
                    else:
                        memory_bank[tid] = emb.clone()
                mem_banks[idx]=memory_bank

        # loss_total.backward()

        optimizer.step() #####################
        print(i)


    torch.save(iafe.state_dict(),
               f'iafe_weights_alternate_yolo_fine_tuned_{epoch}.pt')  ############################################################################################################
    print(f"Epoch {epoch + 1:02d}  loss={epoch_loss:.4f}")




print("part 6")
# ───────────────────────────────────────────────
# 6) Clean up if I ever want to run un-patched model
# ──────────────────────────────────────────────────
hook_handle.remove()
#############################################
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.patches as patches


print("proper plot below:")

