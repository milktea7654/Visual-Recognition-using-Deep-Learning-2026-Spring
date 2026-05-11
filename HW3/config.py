"""config.py – Hyper-parameters for HW3 (ConvNeXt-Base + FPN + Mask R-CNN)."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = "data/train"
TEST_DIR = "data/test_release"
TEST_ID_MAP = "data/test_image_name_to_ids.json"
OUTPUT_DIR = "output"
MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

NUM_CLASSES = 4              # foreground classes (class1..class4)
NUM_CLASSES_WITH_BG = NUM_CLASSES + 1  # +1 background → fed to MaskRCNN

CROP_SIZE = 768              # random-crop tile size (training + sliding-window inference)
SLIDING_WINDOW_OVERLAP = 0.5 # 0.5 → every cell covered by ≥2 tiles, reduces boundary misses

VAL_SPLIT = 0.2              # 80/20 train/val split of data/train/
SEED = 42












# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

USE_ALBUMENTATIONS = True       # heavy aug (rot/flip/colour/elastic/noise)
USE_COPY_PASTE = True           # paste random instances from other images
COPY_PASTE_MAX_OBJECTS = 5      # max instances to paste per image
COPY_PASTE_PROB = 0.5           # probability of applying copy-paste

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
# Choose: "convnext_base" (torchvision V1) or "convnextv2_base" (HF V2 GRN)
BACKBONE_NAME = "convnextv2_base"

BACKBONE_OUT_CHANNELS = 256                       # FPN output channels
BACKBONE_IN_CHANNELS  = [128, 256, 512, 1024]     # both V1 and V2 base widths
# torchvision convnext_base feature-extractor node names
BACKBONE_RETURN_NODES = {
    "features.1": "0",   # stride  4
    "features.3": "1",   # stride  8
    "features.5": "2",   # stride 16
    "features.7": "3",   # stride 32
}

# ---------------------------------------------------------------------------












# Mask R-CNN heads
# ---------------------------------------------------------------------------

# Cell size analysis: p5=17px p50=26px p90=42px p99=65px max=318px
# Old anchors (16,32,64,128,256) waste capacity on 128/256 (>p99)
RPN_ANCHOR_SIZES = ((8,), (16,), (24,), (32,), (64,))
RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(RPN_ANCHOR_SIZES)
BOX_DETECTIONS_PER_IMG = 500  # up from 300; prevents miss in dense regions

# Multi-scale training: list of min sizes randomly sampled per iter
MULTI_SCALE_MIN_SIZES = (512, 640, 768, 896, 1024)
MULTI_SCALE_MAX_SIZE = 1333
INFERENCE_MIN_SIZE = 1024    # larger inference resolution for big images
INFERENCE_MAX_SIZE = 1333

# Higher mask-head resolution for sharper boundaries
MASK_ROI_OUTPUT_SIZE = 28      # default torchvision value is 14
# Deeper mask head: default torchvision is 4-conv layers; 6 gives sharper instance edges
MASK_HEAD_LAYERS = 6

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

EPOCHS = 300
BATCH_SIZE = 2
NUM_WORKERS = 8
USE_AMP = True

OPTIMIZER = "AdamW"
LR = 4e-5          # scaled for batch=1; 8e-4 is too aggressive and diverges
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0    # tighter clip to prevent box/objectness loss explosion

# Cosine LR with linear warmup
SCHEDULER = "CosineAnnealingLR"
WARMUP_EPOCHS = 8
WARMUP_FACTOR = 0.01
LR_MIN = 1e-6

# Class-balanced sampling (WeightedRandomSampler)
USE_CLASS_BALANCED_SAMPLER = True

# Exponential Moving Average (EMA) of model weights
USE_EMA = True
EMA_DECAY = 0.999

# Cascade Mask R-CNN: 3 IoU stages (0.5 → 0.6 → 0.7)
# Adds ~26M params vs standard Mask R-CNN (total ~133M, still under 200M)
# Refines bounding boxes iteratively → better AP especially at IoU > 0.5
USE_CASCADE = True

# ---------------------------------------------------------------------------
# Evaluation / Checkpointing
# ---------------------------------------------------------------------------

VAL_SCORE_THRESHOLD = 0.01
SAVE_EVERY = 0              # 0 = only best.pt + last.pt
EARLY_STOP_PATIENCE = 50    # stop if AP50 does not improve for N epochs

# Full-train mode: use all 209 images (no val split), save every N epochs.
# Best for final submission – no val feedback but more training data.
USE_FULL_TRAIN = True
SW_EVAL_EVERY = 5   # sliding-window val every N epochs (0=disabled); only when not USE_FULL_TRAIN

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

USE_TTA = True           # Test-Time Augmentation: hflip, vflip + scale (5× inference)
SCALE_TTA_FACTORS = (0.8, 1.2)   # additional scale TTA (resize tile, predict, map back)
USE_WBF = True           # Weighted Box Fusion: fuse overlapping tile/TTA boxes
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.0
WBF_WEIGHTS = [1, 1, 1]
SOFT_NMS_SIGMA = 0.5              # Gaussian soft-NMS sigma (used when USE_WBF=False)
SOFT_NMS_MIN_SCORE = 0.001        # minimum score after soft-NMS decay

