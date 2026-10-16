INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
from src.config import STATS, LEGACY_STATS
class CNNDataset(Dataset):
    def __init__(self, data_dir, biome, mode='train', augment=False, patch_size=384, use_lidar_input=False):
        super().__init__()
        self.data_dir = Path(data_dir) / biome
        self.biome = biome
        self.augment = augment
        self.mode = mode
        self.patch_size = patch_size
        self.use_lidar_input = use_lidar_input
        self.legacy = LEGACY_STATS.get(biome, LEGACY_STATS.get('amazonia'))
        self.official = STATS.get(biome, STATS['default'])
        self.input_files = sorted(list(self.data_dir.glob("*_input.npy")))
        rng = random.Random(42)
        rng.shuffle(self.input_files)
        n_total = len(self.input_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        if mode == 'train':
            self.files = self.input_files[:n_train]
        elif mode == 'val':
            self.files = self.input_files[n_train:n_train+n_val]
        else:
            self.files = self.input_files[n_train+n_val:]
        print(f"[{biome.upper()}][{mode}] Loaded {len(self.files)} tiles. SCALE MIGRATION ACTIVE.")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        inp_path = self.files[idx]
        tgt_path = inp_path.parent / inp_path.name.replace("_input.npy", "_target.npy")
        input_np = np.load(inp_path).astype(np.float32)
        target_np = np.load(tgt_path).astype(np.float32)
        o_std, o_mean = self.legacy['std'], self.legacy['mean']
        n_std, n_mean = self.official['std'], self.official['mean']
        input_np[:, :, 0] = (input_np[:, :, 0] * o_std + o_mean - n_mean) / n_std
        target_np[:, :, 0] = (target_np[:, :, 0] * o_std + o_mean - n_mean) / n_std
        if self.use_lidar_input:
            lidar_val_dense = target_np[:, :, 0:1]
            lidar_mask = target_np[:, :, 1:2]
            lidar_sparse = lidar_val_dense * lidar_mask
            input_np = np.concatenate([input_np, lidar_sparse, lidar_mask], axis=-1)
        h, w = input_np.shape[:2]
        th, tw = self.patch_size, self.patch_size
        if h > th or w > tw:
            if self.mode == 'train':
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
            else:
                i = (h - th) // 2
                j = (w - tw) // 2
            input_np = input_np[i:i+th, j:j+tw, :]
            target_np = target_np[i:i+th, j:j+tw, :]
        if self.augment:
            if random.random() > 0.5:
                input_np = np.fliplr(input_np)
                target_np = np.fliplr(target_np)
            if random.random() > 0.5:
                input_np = np.flipud(input_np)
                target_np = np.flipud(target_np)
            k = random.randint(0, 3)
            if k > 0:
                input_np = np.rot90(input_np, k)
                target_np = np.rot90(target_np, k)
        input_tensor = torch.from_numpy(np.ascontiguousarray(input_np).transpose(2, 0, 1))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_np).transpose(2, 0, 1))
        if not self.use_lidar_input and input_tensor.shape[0] > 17:
            input_tensor = input_tensor[:17, :, :]
        return input_tensor, target_tensor
if __name__ == "__main__":
    ds = CNNDataset(f"{INTERNAL_DATA_ROOT}/RESU-NET/data_cnn", "amazonia", mode='train')
    x, y = ds[0]
    print(f"Input Shape: {x.shape}, Target Shape: {y.shape}")
