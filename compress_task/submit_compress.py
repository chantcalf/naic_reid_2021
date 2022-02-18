import os
import glob
import zipfile
import numpy as np
import shutil
import torch
from .sub_models import TrainModel
MODEL_NAME = './compress.pth'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def extract_zipfile(dir_input: str, dir_dest: str):
    files = zipfile.ZipFile(dir_input, "r")
    for file in files.namelist():
        if file.find("__MACOSX") >= 0 or file.startswith('.'):
            continue
        else:
            files.extract(file, dir_dest)
    files.close()
    return 1


def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    assert fea.dtype == np.uint8
    with open(path, 'wb') as f:
        f.write(fea.tostring())
    return True


def compress_all(input_path: str, bytes_rate: int):
    st = torch.load(MODEL_NAME, map_location='cpu')
    select_index = st["encoder.select_index"][0].numpy()
    model = TrainModel([64, 128, 256], select_index)
    model.load_state_dict(st)
    model.to(device)
    model.eval()

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(input_path, '*.*'))
    with torch.no_grad():
        for query_fea_path in query_fea_paths:
            query_basename = get_file_basename(query_fea_path)
            fea = read_feature_file(query_fea_path)
            fea = torch.tensor(fea).to(device).unsqueeze(0)
            fea = model.encoder(fea, bytes_rate)[0].detach().cpu().numpy().astype('uint8')
            compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
            compress_feature(fea, bytes_rate, compressed_fea_path)

    print('Encode Done for bytes_rate' + str(bytes_rate))


def compress(test_path: str, byte: str):
    query_fea_dir = 'query_feature'
    extract_zipfile(test_path, query_fea_dir)
    compress_all(query_fea_dir, int(byte))
    shutil.rmtree(query_fea_dir)
    return 1



