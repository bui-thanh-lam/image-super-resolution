import torch
import random
import PIL.Image as pil_img
import torchvision.transforms as T


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def img_to_tensor(filename):
    img = 0
    try:
        img = pil_img.open(filename).convert('RGB')
    except:
        raise RuntimeError("File not found")
    to_tensor = T.ToTensor()
    tensor = to_tensor(img)
    return tensor.to(DEVICE)


def make_iterator(n_examples=8000, batch_size=64, start_filename=0):
    """Makes iterator through dataset for batch training.

    Args:
        n_examples (int): size of data set.
        batch_size (int): size of each batch.
    
    Returns:
        list: a list of list(str), each element is a list of photos' filename within a batch. For example:
        [['0001.png', '0002.png'], ['0000.png', '0007.png'], ['0006.png', '0008.png'], ['0003.png']] 
        is an iterator of a 5-example-length dataset with batch size 2.
    """
    # Convention: files are named from '1.png', '2.png',..., 'xxxx.png'
    files = list(range(1, n_examples+1))
    random.shuffle(files)
    iterator = []
    for batch in range(n_examples // batch_size):
        new_batch = []
        for offset in range(batch_size):
            new_filename = str(files[batch*batch_size + offset] + start_filename) + '.png'
            new_batch.append(new_filename)
        iterator.append(new_batch)
    return iterator


def prepare_tensors(batch, LR_folder_path, HR_folder_path):
    """Preprare inputs and targets tensor for batch training.

    Args:
        batch (list(str)): a list of files name within a batch.
        LR_folder_path (str): path to folder containing LR images.
        HR_folder_path (str): path to folder containing HR images.

    Returns:
        inputs (tensor: [N, C, H, W]), targets (tensor: [N, C, H, W]).
    """
    inputs_tmp = []
    targets_tmp = []
    for filename in batch:
        input_tmp = img_to_tensor(LR_folder_path + filename)
        target_tmp = img_to_tensor(HR_folder_path + filename)
        if input_tmp != None and target_tmp != None:
            inputs_tmp.append(input_tmp)
            targets_tmp.append(target_tmp)
    inputs = torch.stack(inputs_tmp)
    targets = torch.stack(targets_tmp)
    del inputs_tmp
    del targets_tmp
    return inputs, targets