import pickle
import numpy as np
import os


def get_cifar10_train_data_and_label(root=""):
    def load_file(filename):
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    #这里请读者自行补充更多的数据
    data_batch_1 = load_file(os.path.join(root, 'data_batch_1'))
    data_batch_2 = load_file(os.path.join(root, 'data_batch_2'))
    data_batch_3 = load_file(os.path.join(root, 'data_batch_3'))
    data_batch_4 = load_file(os.path.join(root, 'data_batch_4'))
    data_batch_5 = load_file(os.path.join(root, 'data_batch_5'))
    dataset = []
    labelset = []
    for data in [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]:
        img_data = (data["data"])
        img_label = (data["labels"])
        dataset.append(img_data)
        labelset.append(img_label)
    dataset = np.concatenate(dataset)
    labelset = np.concatenate(labelset)
    return dataset, labelset


def get_cifar10_test_data_and_label(root=""):
    def load_file(filename):
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    data_batch_1 = load_file(os.path.join(root, 'test_batch'))
    dataset = []
    labelset = []
    for data in [data_batch_1]:
        img_data = (data["data"])
        img_label = (data["labels"])
        dataset.append(img_data)
        labelset.append(img_label)
    dataset = np.concatenate(dataset)
    labelset = np.concatenate(labelset)
    return dataset, labelset


def get_CIFAR10_dataset(root=""):
    train_dataset, label_dataset = get_cifar10_train_data_and_label(root=root)
    test_dataset, test_label_dataset = get_cifar10_train_data_and_label(root=root)
    return train_dataset, label_dataset, test_dataset, test_label_dataset


if __name__ == "__main__":
    train_dataset, label_dataset, test_dataset, test_label_dataset = get_CIFAR10_dataset(root="../dataset/cifar-10-batches-py/")

    train_dataset = np.reshape(train_dataset, [len(train_dataset), 3, 32, 32]).astype(np.float32) / 255.
    test_dataset = np.reshape(test_dataset, [len(test_dataset), 3, 32, 32]).astype(np.float32) / 255.
    label_dataset = np.array(label_dataset).astype(int)
    test_label_dataset = np.array(test_label_dataset).astype(int)

    print(train_dataset.shape)
    print(label_dataset.shape)

    print(test_dataset.shape)
    print(test_label_dataset.shape)
