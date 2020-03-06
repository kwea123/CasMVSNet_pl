from .dtu import DTUDataset
from .tanks import TanksDataset
from .blendedmvs import BlendedMVSDataset

dataset_dict = {'dtu': DTUDataset,
                'tanks': TanksDataset,
                'blendedmvs': BlendedMVSDataset}