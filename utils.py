import sys
import mindspore.dataset as ds
import numpy as np
from mindspore.dataset.transforms import TypeCast
import mindspore as ms

if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle


AUDIO = b'covarep'
VISUAL = b'facet'
TEXT = b'glove'
LABEL = b'label'
TRAIN = b'train'
VALID = b'valid'
TEST = b'test'

def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings

def load_pom(data_path):
    '''
    parse the input args
    '''
    class POM:
        '''
        Dataset for POM
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            # batch size
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'))
    else:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'), encoding='bytes')
    pom_train, pom_valid, pom_test = pom_data[TRAIN], pom_data[VALID], pom_data[TEST]

    train_audio, train_visual, train_text, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL], pom_train[TEXT], pom_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[TEXT], pom_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[TEXT], pom_test[LABEL]

    # remove possible NaN values
    train_audio[np.isnan(train_audio)] = 0
    valid_audio[np.isnan(valid_audio)] = 0
    test_audio[np.isnan(test_audio)] = 0
    train_visual[np.isnan(train_visual)] = 0
    valid_visual[np.isnan(valid_visual)] = 0
    test_visual[np.isnan(test_visual)] = 0

    # code that instantiates the Dataset objects
    train_generator = POM(train_audio, train_visual, train_text, train_labels)
    valid_generator = POM(valid_audio, valid_visual, valid_text, valid_labels)
    test_generator = POM(test_audio, test_visual, test_text, test_labels)

    train_set = ds.GeneratorDataset(train_generator, ["audio", "visual", "text", "label"], shuffle=True)
    valid_set = ds.GeneratorDataset(valid_generator, ["audio", "visual", "text", "label"], shuffle=True)
    test_set = ds.GeneratorDataset(test_generator, ["audio", "visual", "text", "label"], shuffle=True)

    audio_dim, visual_dim, text_dim = 0, 0, 0
    for train_data in train_set.create_dict_iterator():
        audio_dim = train_data["audio"].shape[0]
        print("Audio feature dimension is: {}".format(audio_dim))
        visual_dim = train_data["visual"].shape[0]
        print("Visual feature dimension is: {}".format(visual_dim))
        text_dim = train_data["text"].shape[1]
        print("Text feature dimension is: {}".format(text_dim))
        break
    input_dims = (audio_dim, visual_dim, text_dim)

    return train_set, valid_set, test_set, input_dims


def load_iemocap(data_path, emotion):
    '''
    parse the input args
    '''

    class IEMOCAP:
        '''
        Dataset for IEMOCAP
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'))
    else:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'), encoding='bytes')
    iemocap_train, iemocap_valid, iemocap_test = iemocap_data[emotion][TRAIN], iemocap_data[emotion][VALID], iemocap_data[emotion][TEST]

    train_audio, train_visual, train_text, train_labels \
        = iemocap_train[AUDIO], iemocap_train[VISUAL], iemocap_train[TEXT], iemocap_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = iemocap_valid[AUDIO], iemocap_valid[VISUAL], iemocap_valid[TEXT], iemocap_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = iemocap_test[AUDIO], iemocap_test[VISUAL], iemocap_test[TEXT], iemocap_test[LABEL]

    # remove possible NaN values
    train_audio[np.isnan(train_audio)] = 0
    valid_audio[np.isnan(valid_audio)] = 0
    test_audio[np.isnan(test_audio)] = 0
    train_visual[np.isnan(train_visual)] = 0
    valid_visual[np.isnan(valid_visual)] = 0
    test_visual[np.isnan(test_visual)] = 0

    # code that instantiates the Dataset objects
    train_generator = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_generator = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_generator = IEMOCAP(test_audio, test_visual, test_text, test_labels)

    train_set = ds.GeneratorDataset(train_generator, ["audio", "visual", "text", "label"], shuffle=True)
    valid_set = ds.GeneratorDataset(valid_generator, ["audio", "visual", "text", "label"], shuffle=True)
    test_set = ds.GeneratorDataset(test_generator, ["audio", "visual", "text", "label"], shuffle=True)

    audio_dim, visual_dim, text_dim = 0, 0, 0
    for train_data in train_set.create_dict_iterator():
        audio_dim = train_data["audio"].shape[0]
        print("Audio feature dimension is: {}".format(audio_dim))
        visual_dim = train_data["visual"].shape[0]
        print("Visual feature dimension is: {}".format(visual_dim))
        text_dim = train_data["text"].shape[1]
        print("Text feature dimension is: {}".format(text_dim))
        break
    input_dims = (audio_dim, visual_dim, text_dim)

    return train_set, valid_set, test_set, input_dims


def load_mosi(data_path):
    '''
    parse the input args
    '''

    class MOSI:
        '''
        Dataset for MOSI
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'))
    else:
        mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'), encoding='bytes')
    mosi_train, mosi_valid, mosi_test = mosi_data[TRAIN], mosi_data[VALID], mosi_data[TEST]

    train_audio, train_visual, train_text, train_labels \
        = mosi_train[AUDIO], mosi_train[VISUAL], mosi_train[TEXT], mosi_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = mosi_valid[AUDIO], mosi_valid[VISUAL], mosi_valid[TEXT], mosi_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = mosi_test[AUDIO], mosi_test[VISUAL], mosi_test[TEXT], mosi_test[LABEL]

    print(train_audio.shape)
    print(train_visual.shape)
    print(train_text.shape)
    print(train_labels.shape)

    # remove possible NaN values
    train_audio[np.isnan(train_audio)] = 0
    valid_audio[np.isnan(valid_audio)] = 0
    test_audio[np.isnan(test_audio)] = 0
    train_visual[np.isnan(train_visual)] = 0
    valid_visual[np.isnan(valid_visual)] = 0
    test_visual[np.isnan(test_visual)] = 0

    # code that instantiates the Dataset objects
    train_generator = MOSI(train_audio, train_visual, train_text, train_labels)
    valid_generator = MOSI(valid_audio, valid_visual, valid_text, valid_labels)
    test_generator = MOSI(test_audio, test_visual, test_text, test_labels)

    train_set = ds.GeneratorDataset(train_generator, ["audio", "visual", "text", "label"], shuffle=True)
    valid_set = ds.GeneratorDataset(valid_generator, ["audio", "visual", "text", "label"], shuffle=True)
    test_set = ds.GeneratorDataset(test_generator, ["audio", "visual", "text", "label"], shuffle=True)

    audio_dim, visual_dim, text_dim = 0, 0, 0
    for train_data in train_set.create_dict_iterator():
        audio_dim = train_data["audio"].shape[0]
        print("Audio feature dimension is: {}".format(audio_dim))
        visual_dim = train_data["visual"].shape[0]
        print("Visual feature dimension is: {}".format(visual_dim))
        text_dim = train_data["text"].shape[1]
        print("Text feature dimension is: {}".format(text_dim))
        break
    input_dims = (audio_dim, visual_dim, text_dim)

    return train_set, valid_set, test_set, input_dims
