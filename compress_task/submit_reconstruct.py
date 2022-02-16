import os
import glob
import numpy as np

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def decompress_feature(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        feature_len = int.from_bytes(f.read(4), byteorder='little', signed=False)
        fea = np.frombuffer(f.read(), dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(feature_len - fea.shape[0], dtype='<f4')], axis=0
    )
    return fea

def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(fea.astype('<f4').tostring())
    return True

def reconstruct(byte_rate: str):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(byte_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea = decompress_feature(compressed_query_fea_path)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Decode Done' + byte_rate)
