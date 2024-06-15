# import time

from tqdm import tqdm


# import numba


def str2bytes(input_list: list[str]):
    out_list = []
    for i in input_list:
        datas = str(i).encode('utf8')
        out_list.append(datas)
    return out_list


def bytes2str(input_list: list[bytes]):
    out_list = []
    for i in input_list:
        datas = i.decode('utf8')
        out_list.append(datas)
    return out_list


# @numba.jit(nopython=True, nogil=True,)
# def fast_build_binary_data(input_list: list[bytes]):
#     index_binary = b''
#     data_binary = b''
#     offset = 0
#     for i in input_list:
#         datas = i
#         # index_binary = index_binary + offset.to_bytes(length=4, byteorder='big')
#         # data_binary = data_binary + datas
#         index_binary += offset.to_bytes(length=4, byteorder='big')
#         data_binary += datas
#         offset += len(datas)
#     return data_binary, index_binary


# def build_binary_data(input_list: list[bytes]):
#     data_len = len(input_list).to_bytes(length=4, byteorder='big')
#     index_binary = b''
#     data_binary = b''
#     offset = 0
#     for i in tqdm(input_list, leave=False):
#         datas = i
#         # index_binary = index_binary + offset.to_bytes(length=4, byteorder='big')
#         # data_binary = data_binary + datas
#         index_binary += offset.to_bytes(length=4, byteorder='big')
#         data_binary += datas
#         offset += len(datas)
#     # t1=time.time()
#     # data_binary, index_binary = fast_build_binary_data(input_list)
#     # t2 = time.time()
#     # print(t2-t1)
#     index_binary = data_len + index_binary
#     return data_binary, index_binary

def build_binary_data(input_list: list[bytes]):
    data_len = len(input_list).to_bytes(length=4, byteorder='big')

    index_binary = bytearray()
    data_binary = bytearray()

    offset = 0
    for i in tqdm(input_list, leave=False):
        datas = i
        index_binary.extend(offset.to_bytes(length=4, byteorder='big'))
        data_binary.extend(datas)
        offset += len(datas)

    index_binary = data_len + bytes(index_binary)
    return bytes(data_binary), bytes(index_binary)


def read_binary_data(index: int, binary_data: bytes, index_binary: bytes) -> bytes:
    max_index = int.from_bytes(index_binary[:4], byteorder='big')
    assert index + 1 <= max_index
    if index + 1 != max_index:
        start_offset, end_offset = int.from_bytes(index_binary[(index + 1) * 4:(index + 2) * 4],
                                                  byteorder='big'), int.from_bytes(
            index_binary[(index + 2) * 4:(index + 3) * 4], byteorder='big')
        datas = binary_data[start_offset:end_offset]
    else:
        start_offset = int.from_bytes(index_binary[(index + 1) * 4:(index + 2) * 4], byteorder='big')
        datas = binary_data[start_offset:]
        pass
    return datas
