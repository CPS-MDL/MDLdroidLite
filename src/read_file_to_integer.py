import numpy as np


def read_data_to_npy(data, num_channel, num_byte, num_sample=10, signed=True):
    row_length = num_channel * num_byte
    npy = np.zeros((num_sample, num_channel))
    for row_index in range(num_sample):
        for i in range(0, row_length, num_byte):
            s = i + row_index * row_length
            e = s + num_byte
            hexadecimal_string = data[s:e]
            byte_array = bytearray.fromhex(hexadecimal_string)
            value = int.from_bytes(byte_array, byteorder='little', signed=signed)
            npy[row_index, int(i / num_byte)] = value
    # npy = npy.reshape((1,-1) ,order='F')
    return npy


if __name__ == "__main__":

    path_to_data = "/Users/zber/ProgramDev/data_process_jupyter/Hua/data/Jack_10s.txt"

    p_id = "5A0391000A0906870B"
    p_id_length = 13 * 2
    p_length = 917 * 2
    p_start_end = "FFFF"
    f = open(path_to_data, 'r')
    position = 0
    num_package = 0
    num_sample = 10

    acc_start = 45 * 2
    acc_end = 135 * 2
    acc_num_byte = 2 * 2
    acc_num_channel = 3
    acc_row_length = acc_num_channel * acc_num_byte

    ppg_start = 135 * 2
    ppg_end = 615 * 2
    ppg_num_channel = 12
    ppg_num_byte = 4 * 2

    total_acc_npy = None
    total_ppg_npy = None

    while True:
        f.seek(position)
        p_content = f.read(p_length)
        if p_content == '':
            break
        if p_content[:len(p_id)] == p_id and p_content[-6:-2] == p_start_end:
            # do something here
            # read acc data
            acc_data = p_content[acc_start:acc_end]
            acc_npy = read_data_to_npy(acc_data, acc_num_channel, acc_num_byte)
            if total_acc_npy is None:
                total_acc_npy = acc_npy
            else:
                total_acc_npy = np.vstack((total_acc_npy, acc_npy))

            # read ppg data
            ppg_data = p_content[ppg_start: ppg_end]
            ppg_npy = read_data_to_npy(ppg_data, ppg_num_channel, ppg_num_byte, signed=True)
            if total_ppg_npy is None:
                total_ppg_npy = ppg_npy
            else:
                total_ppg_npy = np.vstack((total_ppg_npy, ppg_npy))
            # loop going on
            position += p_length
            num_package += 1
        else:
            position += 1
    # save file
    np.save("/Users/zber/ProgramDev/data_process_jupyter/Hua/data/acc", acc_npy)
    np.save("/Users/zber/ProgramDev/data_process_jupyter/Hua/data/ppg", ppg_npy)

    print("{} pacakge found".format(num_package))
