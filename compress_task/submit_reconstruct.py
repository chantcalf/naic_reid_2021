import glob
import os

import numpy as np
import torch

try:
    from sub_models import TrainModel
except:
    from .sub_models import TrainModel
MODEL_NAME = './compress.pth'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def decompress_feature(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='uint8')


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(fea.astype('<f4').tostring())
    return True


def reconstruct(byte_rate: str):
    st = torch.load(MODEL_NAME, map_location='cpu')
    select_index = st["encoder.select_index"][0].numpy()
    model = TrainModel([64, 128, 256], select_index)
    model.load_state_dict(st)
    model.to(device)
    model.eval()

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(byte_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)
    byte_rate = int(byte_rate)
    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    with torch.no_grad():
        for compressed_query_fea_path in compressed_query_fea_paths:
            query_basename = get_file_basename(compressed_query_fea_path)
            fea = decompress_feature(compressed_query_fea_path)
            fea = torch.tensor(fea).long().to(device).unsqueeze(0)
            fea = model.decoder(fea, byte_rate)[0].detach().cpu().numpy()
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
            write_feature_file(fea, reconstructed_fea_path)

    print(f'Decode Done {byte_rate}')
