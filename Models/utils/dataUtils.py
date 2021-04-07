

def noZeroArrayCheck(data_np):
    """

    :param data: numpy array with shape [num_of_images, channels, shape0, shape1]
    :return:
    """
    assert len(data_np.shape) == 4
    num_of_data = data_np.shape[0]
    for i in range( num_of_data):
        if not data_np[i].any():
            print(str(i)+' th data is all zero')
            assert False
    print('All check, succeed !')
def allZeroArrayCheck(data_np):
    """

    :param data: numpy array with shape [num_of_images, channels, shape0, shape1]
    :return:
    """
    assert len(data_np.shape) == 4
    num_of_data = data_np.shape[0]
    for i in range( num_of_data):
        if data_np[i].any():
            print(str(i)+' th data is not all zero')
            assert False
    print('All check, succeed !')
