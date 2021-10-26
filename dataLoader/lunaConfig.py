'''
    masked_files: masked -> resampled, pad = -3000
    preprocessed: normalized -> masked -> resampled, pad = 170
    modeNorm_files: masked -> resampled -> modeNorm, pad = 0
'''
class LunaConfig(object):
    # DATA_DIR = "./LUNA16/masked_files/"
    # DATA_DIR = "./LUNA16/preprocessed/"
    DATA_DIR = "./LUNA16/modeNorm_files/"

    TRAIN_DATA_DIR = ['subset0/',
                      'subset1/',
                      'subset2/',
                      'subset3/',
                      'subset4/',
                      'subset5/',
                      'subset6/',
                      'subset7/']
    VAL_DATA_DIR = ['subset8/']
    TEST_DATA_DIR = ['subset9/']
    POS_LABEL_FILE = 'annotations.csv'
    POS_LABEL_EXCLUDE_FILE = 'annotations_excluded.csv'
    BLACK_LIST = []

    ANCHORS = [5., 10., 20.]
    # ANCHORS = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    CHANNEL = 1
    CROP_SIZE = [96, 96, 96]
    STRIDE = 4
    MAX_STRIDE = 16
    NUM_NEG = 800
    TH_NEG = 0.02
    TH_POS_TRAIN = 0.5
    TH_POS_VAL = 1
    NUM_HARD = 2
    BOUND_SIZE = 12
    RESO = 1
    SIZE_LIM = 2.5  # 3 #6. #mm
    SIZE_LIM2 = 10  # 30
    SIZE_LIM3 = 20  # 40
    AUG_SCALE = True
    R_RAND_CROP = 0.3
    PAD_VALUE = 0 # -3000, 0, 170
    AUGTYPE = {"flip": True, "swap": False, "scale": True, "rotate": False}

    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5

    SIDE_LEN = 144
    MARGIN = 32

    ORIGIN_SCALE = False

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
