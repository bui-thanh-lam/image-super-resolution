from utils import *
from models import *
import math
import time

SCALE_FACTOR = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, criterion, optimizer):
    """Main function to train a model.

    Args:
        model (nn.Module): any PyTorch model.
        iterator (list): an iterator throughout dataset.
        criterion (nn Loss function): any PyToch-like (original or custom) loss function.
        optimizer (torch.optim): any PyTorch optimizer.

    Returns:
        loss (float32): loss at current epoch.
    """
    model.train()
    bs = len(iterator[0])
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        inputs, targets = prepare_tensors(batch, 
                                          LR_folder_path=f"./data/train/LRx{str(SCALE_FACTOR)}/", 
                                          HR_folder_path=f"./data/train/HR/")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator):
    """Evaluate a model during training time on validation set.

    Args:
        model (nn.Module): any PyTorch model.
        iterator (list): an iterator throughout dataset.

    Returns:
        psnr (float32): average PSNR on validation set at current epoch.
    """
    model.eval()
    bs = len(iterator[0])
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        inputs, targets = prepare_tensors(batch, 
                                          LR_folder_path=f"./data/validation/LRx{str(SCALE_FACTOR)}/", 
                                          HR_folder_path=f"./data/validation/HR/")
        with torch.no_grad():
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(iterator)
    psnr = 10*math.log(1.0/epoch_loss, 10)
    return psnr


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def benchmark(model, test_set='Set5'):
    """Evaluate a model during training time on validation set.

    Args:
        model (nn.Module): any PyTorch model.
        iterator (list): an iterator throughout dataset.

    Returns:
        psnr (float32): average PSNR on test set.
    """
    N_EXAMPLES = 14 if test_set == 'Set14' else 5
    test_iter = make_iterator(n_examples=N_EXAMPLES, batch_size=1)
    LR_FOLDER_PATH = f"./data/test/{test_set}/LRx{str(model.scale_factor)}/"
    HR_FOLDER_PATH = f"./data/test/{test_set}/HR/"
    model.eval()
    bs = len(test_iter[0])
    epoch_loss = 0
    for i, batch in enumerate(test_iter):
        inputs, targets = prepare_tensors(batch, 
                                          LR_folder_path=LR_FOLDER_PATH, 
                                          HR_folder_path=HR_FOLDER_PATH)
        with torch.no_grad():
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(test_iter)
    psnr = 10*math.log(1.0/epoch_loss, 10)
    return psnr


# Hyper params setting and training
N_EPOCHS = 100
BATCH_SIZE = 64
N_EXAMPLES = 8000
model = SRCNNpp(scale_factor=SCALE_FACTOR).to(DEVICE)
loss_func = CombinedLoss(pixel_loss_coeff=0.8)
# loss_func = nn.MSELoss()

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch+1}")
    start_time = time.time()
    train_iter = make_iterator(n_examples=N_EXAMPLES, batch_size=BATCH_SIZE)
    train_loss = train(model, train_iter, loss_func, optim)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    valid_iter = make_iterator(n_examples=100, batch_size=8, start_filename=8100)
    psnr = evaluate(model, valid_iter)
    print(f"\tTime: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain loss: {train_loss:.4f}")
    print(f"\tValid PSNR: {psnr:.2f}")
    
psnr = benchmark(model, test_set='Set14')
print(f"PSNR on test set: {psnr:.2f}")