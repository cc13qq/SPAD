from utils import Config

from .base_recorder import BaseRecorder
from .SP_recorder import SPRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'SP': SPRecorder
    }

    return recorders[config.recorder.name](config)
