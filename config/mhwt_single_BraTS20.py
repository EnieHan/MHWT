from yacs.config import CfgNode as CN

# config definition
_C = CN()
_C.SEED = 42

# dataset config
_C.DATASET = CN()
_C.DATASET.ROOT = ''  # the root of dataset, need to modify
_C.DATASET.CHALLENGE = 'singlecoil'
_C.DATASET.SAMPLE_RATE = 1

_C.TRANSFORMS = CN()
_C.TRANSFORMS.MASKTYPE = 'random'  # "random" or "equispaced"
_C.TRANSFORMS.CENTER_FRACTIONS = [0.08]
_C.TRANSFORMS.ACCELERATIONS = [4]  # 8
_C.TRANSFORMS.MASK_PATH = './dataset/mask/xx.hdf5'  # Need to modify

# model config
_C.MODEL = CN()
_C.MODEL.IMG_SIZE = (192, 192)
_C.MODEL.WINDOW_SIZE = (16, 16, 8, 8, 4, 4)
_C.MODEL.INPUT_DIM = 1   # the channel of input
_C.MODEL.HEAD_HIDDEN_DIM = 120  # the hidden dim of Head
_C.MODEL.MLP_RATIO = 2  # the MLP RATIO Of transformer
_C.MODEL.DEPTHS = (8, 8, 8, 8, 8, 8)  # the hidden dim of Head
_C.MODEL.NUM_HEADS = (4, 4, 4, 4, 4, 4)  # the head's num of multi head attention
_C.MODEL.INTERVAL = (0, 2, 0, 2, 0, 2)
_C.MODEL.RESI_CONNECTION = "1conv"

# the solver config
_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.DEVICE_IDS = [0]  # if [] use cpu, else gpu
_C.SOLVER.LR = 2e-4
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.LR_DROP = []
_C.SOLVER.BATCH_SIZE = int  # Need to modify
_C.SOLVER.NUM_WORKERS = int  # Need to modify
_C.SOLVER.PRINT_FREQ = int  # Need to modify
_C.SOLVER.SAVE_MODAL = int  # Need to modify
_C.SOLVER.PRINT_LOSS = int  # Need to modify

# the others config
_C.RESUME = ''  # model resume path
_C.OUTPUTDIR = './save/weights'  # the model output dir
_C.OUTPUTFILEDIR = './save/file'
_C.LOGDIR = './save/log'
_C.TEST_OUTPUTDIR = './save/test'
# the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = int  # the train epochs, need to modify
_C.USE_MULTI_MODEL = False

# loss
_C.LOSS = CN()
_C.LOSS.TYPE = 'THREE'  # THREE
_C.LOSS.USE_MM_LOSS = False  # 是否使用多模态LOSS
_C.LOSS.ALPHA = 15
_C.LOSS.BETA = 0.1
_C.LOSS.GAMMA = 0.025
_C.LOSS.CHARBONNIER_EPS = 1e-9
