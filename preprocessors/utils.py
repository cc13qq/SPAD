from utils.config import Config

from .base_preprocessor import BasePreprocessor
from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config, split)
    else:
        return test_preprocessors[config.preprocessor.name](config, split)
