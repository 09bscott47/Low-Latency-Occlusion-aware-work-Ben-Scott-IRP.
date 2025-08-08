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

# ───────────────────────────────────────────────────────────────────────────
# 1) Your IAFE module
# ───────────────────────────────────────────────────────────────────────────
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

        enhancement_total = torch.zeros_like(feature_map)

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
        #
        if memory_bank:
            identity_embed = torch.stack(list(memory_bank.values())).mean(dim=0, keepdim=True)  # [1, D]
        else:
            identity_embed = torch.zeros((1, x.shape[-1]), device=feature_map.device)

        for layer in self.layers:
            x = layer(x, identity_embed)  # [B, N, D]

        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)  # [B, D, Hp, Wp]
        enhancement = self.deproj(x)  # [B, C, H, W]

        return feature_map + enhancement

# ───────────────────────────────────────────────────────────────────────────
# 2) Build your custom Dataset
# ───────────────────────────────────────────────────────────────────────────
class VisDroneYoloDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=1280):
        self.imgs = sorted(Path(img_dir).rglob("*.jpg"))
        self.lbls = [Path(lbl_dir) / p.name.replace(".jpg", ".txt") for p in self.imgs]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")

        img = self._resize(img, self.imgsz)  #   FIXED

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


# ───────────────────────────────────────────────────────────────────────────
# 3) Instantiate YOLO hook in IAFE
# ───────────────────────────────────────────────────────────────────────────
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
    patch_size=8,#was 8
    num_heads=4,
    num_layers=4
)

# # 4. Load IAFE weights
iafe.load_state_dict(torch.load('iafe_weights_alternate_yolo_fine_tuned_5.pt', map_location='cpu'))
iafe.eval()
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
# results = model(torch.randn(1, 3, 640, 640))  #   handles training/inference logic safely

results = model.model(torch.randn(1, 3, 1280, 1280))  #  

# _ = model.model(torch.randn(1,3,640,640))
h.remove()


C = feat_store['F'].shape[1]
# iafe = DummyIAFE(feature_dim=C, embed_dim=64).to(device)
# iafe = IAFETransformer(in_channels=C, embed_dim=64, patch_size=8).to(device)
# enhanced_map = iafe(feature_map, memory_bank)

# iafe.train()
# Save a copy of the initial IAFE weights (deep copy)
# iafe.load_state_dict(torch.load(r'C:\Users\cjsco\Downloads\iafe_weights_fixed.pt', map_location='cpu')) #####################################
iafe.eval()
def iafe_hook(module, inp, out):
    # memory_embedding = torch.randn(5, 64, device=out.device)
    enhanced = iafe(out,memory_bank)
    feature_cache['feature_map'] = out.detach()

    return enhanced

hook_handle=model.model.model[4].register_forward_hook(iafe_hook)
print("loading worked")
initial_iafe_weights = {k: v.clone().detach() for k, v in iafe.state_dict().items()}


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

# ───────────────────────────────────────────────────────────────────────────
# 4) Prepare DataLoader
# ───────────────────────────────────────────────────────────────────────────
IMG_DIR = r"C:\Users\cjsco\Downloads\VisDrone2019-MOT-test-dev\VisDrone2019-MOT-test-dev\yolo_images\train\uav0000009_03358_v"#uav0000119_02301_v#uav0000120_04775_v
LBL_DIR = r"C:\Users\cjsco\Downloads\VisDrone2019-MOT-test-dev\VisDrone2019-MOT-test-dev\yolo_labels_w_ids\train\uav0000009_03358_v"#uav0000119_02301_v#uav0000120_04775_v
#uav0000161_00000_v
dataset  = VisDroneYoloDataset(IMG_DIR, LBL_DIR, imgsz=1280)
collate_fn=lambda x: list(zip(*x))
dataloader = DataLoader(
    dataset,
    batch_size=1, #was 4 #############################################
    shuffle=False,
    collate_fn=lambda batch: list(zip(*batch))
)

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
# Load weights
# model = YOLO('yolov8n.pt')
# iafe = IAFETransformer(
#     in_channels=64,
#     embed_dim=64,
#     patch_size=8,
#     num_heads=4,
#     num_layers=4
# )
# checkpoint = torch.load('detector_with_iafe.pth', map_location='cpu')
# model.model.load_state_dict(checkpoint['yolo'])
# iafe.load_state_dict(checkpoint['iafe'])
###############################################################
###############################################################
# model = YOLO('yolo_with_iafe/weights/best.pt')
#
# # Rebuild IAFE
# iafe = IAFETransformer(
#     in_channels=64,
#     embed_dim=64,
#     patch_size=8,
#     num_heads=4,
#     num_layers=4
# )
#
# # Load IAFE weights
# iafe.load_state_dict(torch.load('iafe_weights.pt', map_location='cpu'))
# iafe.eval()

# def iafe_hook(module, inp, out):
#     # memory_embedding = torch.randn(5, 64, device=out.device)
#     enhanced = iafe(out,memory_bank)
#     feature_cache['feature_map'] = out.detach()
#
#     return enhanced
#
# hook_handle=model.model.model[4].register_forward_hook(iafe_hook)

print("training:")
# ───────────────────────────────────────────────────────────────────────────
# 5) Training loop
# ───────────────────────────────────────────────────────────────────────────
replay_buffer = []
from ultralytics.utils.loss import v8DetectionLoss
memory_bank={} #dkkkkkkkkkkk if here is the correct place!!!!!!!!!!!!!!!!!!!!
with torch.no_grad():
    for epoch in range(1):
        memory_bank = {}
        epoch_loss=0
        i=0
        for imgs_list, targets_list in dataloader:
            imgs = torch.stack(imgs_list, 0).to(device)
            # move targets to device
            for t in targets_list:
                t['cls']      = t['cls'].to(device)
                t['trk_id']   = t['trk_id'].to(device)
                t['boxes']    = t['boxes'].to(device)
                t['img_size'] = t['img_size'].to(device)

            _ = model.model(imgs)  # fires both hooks: just one model originally ###############################

            raw_preds = raw_store['preds']  # pure Tensors

            head_outputs = raw_preds[1]

            loss_fn = v8DetectionLoss(model.model)

            batch = convert_batch_to_loss_format(targets_list)
            #batch['images'] = imgs  #  Add this ######################################### new 20/06 COMENTED OUT, NEED THIS IF WANT TO PLOTTTTTTTTTT
            loss_output = loss_fn(raw_preds, batch)#batch_obj
            #print(loss_output)
            # print(loss_output[0].requires_grad)
            optimizer.zero_grad()
            loss_total = loss_output[0]  # this is differentiable
            loss_additive=loss_total.item()
            epoch_loss+=loss_additive
            #loss_total.backward()
            i+=1
            #optimizer.step() ############################################################################################
            print("stepping",i)
            ##Now, I want to update E_{t,k} with ground truth BBs.
            bboxes_from_targets = [t['boxes'] for t in targets_list]
            trk_box_dict = {}
            for t in targets_list:
                trk_ids = t['trk_id']
                boxes = t['boxes']
                if trk_ids.numel() == 0:
                    continue
                for track_id, box in zip(trk_ids.tolist(), boxes):
                    trk_box_dict[track_id] = box  # if track_id seen multiple times, last one wins

            mem_embed_R=get_tracklet_embeddings_from_feature_map(feature_cache['feature_map'],trk_box_dict)

            ema_decay=0.95
            # # Update memory via EMA
            # for tid, emb in mem_embed_R.items():
            #     if tid in memory_bank:
            #         memory_bank[tid] = 0 * memory_bank[tid] + 0 * emb ######### CHANGED ##############
            #     else:
            #         memory_bank[tid] = emb.clone()
            # Update memory via EMA
            for tid, emb in mem_embed_R.items():
                if tid in memory_bank:
                    memory_bank[tid] = ema_decay * memory_bank[tid] + (1 - ema_decay) * emb
                else:
                    memory_bank[tid] = emb.clone()
            ############################################################### above needs to be uncommented
            #print("wow")
            # frame_data = {
            #     #'img': imgs.cpu(),  # [3, H, W]
            #     'mem_embedding': {k: v.detach().cpu() for k, v in memory_bank.items()},
            #     'gt_boxes': targets_list[0]['boxes'].cpu(),  # [N, 4]
            #     'trk_ids': targets_list[0]['trk_id'].cpu()  # [N]
            # }
            #replay_buffer.append(frame_data)

        #torch.save(iafe.state_dict(), 'iafe_weights3.pt') #######################################################################################################################
        # torch.save(model.model.state_dict(), 'yolo_weights2.pt')

            #print("replay buffer")
        # if epoch==0:
        #     torch.save(replay_buffer, 'full_replay_buffer_no_img.pt_uav0000086_00000_v.pt')
            #cheking iafe weights are changing:

            # (tensor(3.4484, grad_fn= < MulBackward0 >), ['cls:', tensor(1.7561, grad_fn= < MulBackward0 >), 'box:', tensor(
            #     0.0129, grad_fn= < MulBackward0 >), 'dfl:', tensor(1.6794, grad_fn= < MulBackward0 >)])
            # (tensor(3.4070, grad_fn= < MulBackward0 >), ['cls:', tensor(1.7293, grad_fn= < MulBackward0 >), 'box:', tensor(
            #     0.0128, grad_fn= < MulBackward0 >), 'dfl:', tensor(1.6649, grad_fn= < MulBackward0 >)])




            # for name, param in iafe.named_parameters():
            #     if param.requires_grad:
            #         current = param.detach()
            #         initial = initial_iafe_weights[name]
            #         if not torch.allclose(current, initial):
            #             print(f" {name} has changed.")
            #         else:
            #             print(f" {name} has NOT changed.")

        print(f"Epoch {epoch+1:02d}  loss={epoch_loss:.4f}")


# ───────────────────────────────────────────────────────────────────────────
# 6) Clean up if you ever want to run the un-patched model
# ───────────────────────────────────────────────────────────────────────────
hook_handle.remove()
#############################################
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.patches as patches


print("proper plot below:")
# def plot_batch_with_boxes(images, batch):
#     batch_idx = batch['batch_idx']
#     bboxes = batch['bboxes']
#     cls = batch['cls']
#     # img_sizes = batch['img_size']
#
#     num_imgs = len(images)
#     fig, axs = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5)) #5,5
#
#     if num_imgs == 1:
#         axs = [axs]
#
#     for i in range(num_imgs):
#         img = images[i].permute(1, 2, 0).numpy()
#         axs[i].imshow(img)
#         axs[i].set_title(f"Image {i}")
#         axs[i].axis("off")
#
#         mask = batch_idx == i
#         boxes = bboxes[mask]
#         labels = cls[mask]
#
#         # scale from [0,1] to pixels
#         h, w = (640,640)
#         boxes_pixel = boxes.clone()
#         boxes_pixel[:, 0] *= w  # xc
#         boxes_pixel[:, 1] *= h  # yc
#         boxes_pixel[:, 2] *= w  # w
#         boxes_pixel[:, 3] *= h  # h
#
#         for box, label in zip(boxes_pixel, labels):
#             xc, yc, bw, bh = box.tolist()
#             x1 = xc - bw / 2
#             y1 = yc - bh / 2
#             rect = patches.Rectangle(
#                 (x1, y1), bw, bh,
#                 linewidth=2,
#                 edgecolor='lime',
#                 facecolor='none'
#             )
#             axs[i].add_patch(rect)
#             axs[i].text(x1, y1 - 2, str(label.item()), color='white',
#                         bbox=dict(facecolor='green', alpha=0.5))
#
#     plt.tight_layout()
#     plt.show()
#
# images, targets_list = next(iter(dataloader))
# batch = convert_batch_to_loss_format(targets_list)
# plot_batch_with_boxes(images, batch)
