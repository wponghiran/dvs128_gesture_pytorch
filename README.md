# Standalone IBM DVS128 Gesture Dataset on PyTorch
Standalone IBM DVS128 Gesture Dataset on PyTorch. Most codes in this repository are extracted from [*Spiking Jelly*](https://spikingjelly.readthedocs.io/zh_CN/latest/spikingjelly.datasets.html#spikingjelly.datasets.dvs128_gesture.DVS128Gesture), which is a neuromorphic simulator repository. This is intended for anyone to experiment with the IBM DVS128 Gesture dataset without solving all dependencies of *Spiking Jelly* that could be daunting sometimes.

Different from [*Tonic*](https://tonic.readthedocs.io/en/latest/reference/generated/tonic.datasets.DVSGesture.html#tonic.datasets.DVSGesture) - another neuromorphic dataset library, the dataset class in *Spiking Jelly* and this repository directly extracts sample from the original IBM dataset. *Tonic* has already pre-processed the original dataset and remove some samples, thus the number of samples in *Tonic* are slightly smaller.

**Example of dataset usage**
```python
from tqdm import tqdm
import torch

device = 'cuda'
batch_size = 4
workers = 4
simulation_steps = 16
dataset_dir = '<ENTER PATH OF YOU DATASET HERE>'
# If dataset doesn't exist, the dataset will be download to the specified location

# Test loading DVS 128 gesture dataset and spliting each sample into N frames
# such that each frame has about the same number of events
print("Loading data - Example mode 1")
dataset_train = DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=simulation_steps, split_by='number')
dataset_test = DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=simulation_steps, split_by='number')
print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

print("Creating data loaders")
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    shuffle=True, num_workers=workers, pin_memory=False)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    shuffle=False, num_workers=workers, pin_memory=False)

for event_reprs, target in tqdm(data_loader, desc='Loading training data'):
    pass
    # Do something
print(event_reprs.shape, target.shape)

for event_reprs, target in tqdm(data_loader_test, desc='Loading testing data'):
    pass
    # Do something
print(event_reprs.shape, target.shape)
```

**Another example of dataset usage in a different mode**
```python
# Test loading DVS 128 gesture dataset and spliting each sample into abritrary number of frames
# such that each frame has about the same duration for instance 3e5 micro second
print("Loading data - Example mode 2")
dataset_train = DVS128Gesture(root=dataset_dir, train=True, data_type='frame', split_by='frame_duration', frame_duration=300000)
dataset_test = DVS128Gesture(root=dataset_dir, train=False, data_type='frame', split_by='frame_duration', frame_duration=300000)
print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

print("Creating data loaders")
# Collate function is needed because each sample may have a different size
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, collate_fn=base_dataset.pad_seq,
    shuffle=True, num_workers=workers, pin_memory=False)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, collate_fn=base_dataset.pad_seq,
    shuffle=False, num_workers=workers, pin_memory=False)

# Suppose we want to measure length of event representation 
train_repr_lens = []
for event_reprs, repr_lens, target in tqdm(data_loader, desc='Loading training data'):
    event_reprs = event_reprs
    target = target
    # Collecting length of event representation when splitting by this method
    train_repr_lens.extend(list(repr_lens))
train_repr_lens = torch.as_tensor(train_repr_lens)
# Print statistic of the event representation length 
print(torch.min(train_repr_lens), torch.max(train_repr_lens), torch.mean(train_repr_lens.float()), torch.std(train_repr_lens.float()))

# Repeat the same thing with test set 
test_repr_lens = []
for event_reprs, repr_lens, target in tqdm(data_loader_test, desc='Loading testing data'):
    # Do something
    event_reprs = event_reprs
    target = target
    event_reprs = event_reprs.float()
    test_repr_lens.extend(list(repr_lens))
test_repr_lens = torch.as_tensor(test_repr_lens)
# Print statistic of the event representation length 
print(torch.min(test_repr_lens), torch.max(test_repr_lens), torch.mean(test_repr_lens.float()), torch.std(test_repr_lens.float()))
```
There are other modes of loading dataset. See files for details.
