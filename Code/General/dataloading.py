# Path to images
root_dir = r"/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary"


# Getting files
data_dir = os.path.join(root_dir,"train") # Modify to val and test for val and test files
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

# print(f"Total image count: {num_total}")
# print(f"Image dimensions: {image_width} x {image_height}")
# print(f"Label names: {class_names}")
# print(f"Label counts: {num_each}")

val_frac = 0 # Change to 1 if validation data set is being loaded
test_frac = 0 # Change to 1 if test data set is being loaded
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(type(train_y))
print(np.shape(train_y))
print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

# Perform basic transforms for model training
class MyCompose(Compose):
    def set_random_state(self, seed=None, state=None):
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            # Change data type to int64
            _seed = int(_seed % np.iinfo(np.int64).max)
            self.R = np.random.RandomState(_seed)
        return self

train_transforms = MyCompose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
])

val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

# Generate PyTorch DataLoader
class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index], self.image_files[index]

# Specify batch size
train_batch_size = 8
val_batch_size = 8
test_batch_size = 8

# Change to val and test based on use case
train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=2)
