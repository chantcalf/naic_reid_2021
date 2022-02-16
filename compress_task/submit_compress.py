import os
import glob
import zipfile
import numpy as np
import shutil

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea

def extract_zipfile(dir_input: str, dir_dest: str):
    files = zipfile.ZipFile(dir_input, "r")
    for file in files.namelist():
        if file.find("__MACOSX")>=0 or file.startswith('.'): continue
        else:
            files.extract(file, dir_dest)
    files.close()
    return 1

def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(int(fea.shape[0]).to_bytes(4, byteorder='little', signed=False))
        f.write(fea.astype('<f4')[:(target_bytes - 4) // 4].tostring())
    return True


def compress_all(input_path: str, bytes_rate: int):

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(input_path, '*.*'))
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        compress_feature(fea, bytes_rate, compressed_fea_path)

    print('Encode Done for bytes_rate' + str(bytes_rate))


def compress(test_path:str, byte: str):
    query_fea_dir = 'query_feature'
    extract_zipfile(test_path, query_fea_dir)
    compress_all(query_fea_dir, int(byte))
    shutil.rmtree(query_fea_dir)
    return 1



