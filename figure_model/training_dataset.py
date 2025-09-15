"""
OHLC Deep Learning Training Dataset

This module provides training dataset creation and management for multi-timeframe OHLC images.
Supports dynamic loading, memory management, and flexible time horizon prediction.

Features:
- Multi-timeframe image stacking (3min, 15min, 1h)
- Flexible return prediction horizons (1h to 7d)
- Batch loading for memory efficiency
- Data alignment validation
- Time-based filtering
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import gc
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
TIMEFRAMES = ['3min', '15min', '1h']
RETURN_HORIZON_MAP = {
    '1h': '1h', '2h': '2h', '4h': '4h', '8h': '8h',
    '12h': '12h', '16h': '16h', '18h': '18h',
    '1d': '1d', '1d6h': '1d6h', '1d12h': '1d12h',
    '2d': '2d', '3d': '3d', '4d': '4d', '5d': '5d', '6d': '6d', '7d': '7d'
}

# Memory management
MAX_MEMORY_GB = 8
BATCH_LOAD_SIZE = 10000


def find_available_files(data_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Scan available data files"""
    files_info = {'images': {tf: [] for tf in TIMEFRAMES}, 'labels': {}}

    print("📁 Scanning available data files...")

    # Scan image files
    for timeframe in TIMEFRAMES:
        pattern = f"images_{timeframe}_*.pt"
        pt_files = list(data_dir.glob(pattern))
        pattern = f"images_{timeframe}_*.npz"
        npz_files = list(data_dir.glob(pattern))

        files_info['images'][timeframe] = sorted(pt_files + npz_files)
        print(f"  {timeframe}: {len(files_info['images'][timeframe])} image files")

    # Scan label files
    for horizon_key, horizon_str in RETURN_HORIZON_MAP.items():
        pattern = f"labels_{horizon_str}_*.pt"
        pt_files = list(data_dir.glob(pattern))
        pattern = f"labels_{horizon_str}_*.npz"
        npz_files = list(data_dir.glob(pattern))

        files_info['labels'][horizon_key] = sorted(pt_files + npz_files)
        if len(files_info['labels'][horizon_key]) > 0:
            print(f"  {horizon_key}: {len(files_info['labels'][horizon_key])} label files")

    # Remove empty label categories
    files_info['labels'] = {k: v for k, v in files_info['labels'].items() if len(v) > 0}

    return files_info


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename"""
    try:
        # Extract YYYY-MM format date
        import re
        match = re.search(r'(\d{4})-(\d{2})', filename)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            return datetime(year, month, 1)
    except:
        pass
    return None


def filter_files_by_date_range(files: List[Path], start_date: Optional[str], end_date: Optional[str]) -> List[Path]:
    """Filter files by date range"""
    if not start_date and not end_date:
        return files

    start_dt = datetime.strptime(start_date, '%Y-%m') if start_date else None
    end_dt = datetime.strptime(end_date, '%Y-%m') if end_date else None

    filtered_files = []
    for file_path in files:
        file_date = extract_date_from_filename(file_path.name)
        if file_date:
            if start_dt and file_date < start_dt:
                continue
            if end_dt and file_date > end_dt:
                continue
            filtered_files.append(file_path)

    return sorted(filtered_files)


def load_single_file(file_path: Path) -> Dict:
    """Load single file (supports .pt and .npz formats)"""
    try:
        if file_path.suffix == '.pt':
            return torch.load(file_path, weights_only=False)
        elif file_path.suffix == '.npz':
            npz_data = np.load(file_path, allow_pickle=True)
            # Convert npz format to dictionary
            result = {}
            for key in npz_data.files:
                data = npz_data[key]
                if key in ['images', 'labels'] and isinstance(data, np.ndarray):
                    result[key] = torch.from_numpy(data)
                else:
                    result[key] = data.item() if data.ndim == 0 else data
            return result
    except Exception as e:
        print(f"❌ Failed to load file {file_path}: {e}")
        return None


class StackedOHLCDataset(Dataset):
    """Stacked multi-timeframe OHLC image dataset"""

    def __init__(self,
                 data_dir: Path,
                 label_horizon: str = '1d',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 batch_load_size: int = 10000,
                 preload_all: bool = False):
        """
        Initialize dataset

        Args:
            data_dir: Data file directory
            label_horizon: Return time horizon ('1h', '1d', '7d' etc)
            start_date: Start date (format: 'YYYY-MM')
            end_date: End date (format: 'YYYY-MM')
            batch_load_size: Batch loading size
            preload_all: Whether to preload all data to memory
        """
        self.data_dir = Path(data_dir)
        self.label_horizon = label_horizon
        self.batch_load_size = batch_load_size
        self.preload_all = preload_all

        # Validate parameters
        if label_horizon not in RETURN_HORIZON_MAP:
            raise ValueError(f"Unsupported return horizon: {label_horizon}. Available: {list(RETURN_HORIZON_MAP.keys())}")

        # Scan files
        self.files_info = find_available_files(self.data_dir)

        # Filter by date range
        self._filter_files_by_date(start_date, end_date)

        # Validate data alignment
        self._validate_data_alignment()

        # Build index mapping
        self._build_index_mapping()

        # Preload data (if needed)
        if self.preload_all:
            self._preload_all_data()
        else:
            self.loaded_batches = {}  # Cache loaded batches

    def _filter_files_by_date(self, start_date: Optional[str], end_date: Optional[str]):
        """Filter files by date range"""
        print(f"\n📅 Filtering date range: {start_date or 'start'} to {end_date or 'end'}")

        # Filter image files
        for timeframe in TIMEFRAMES:
            original_count = len(self.files_info['images'][timeframe])
            self.files_info['images'][timeframe] = filter_files_by_date_range(
                self.files_info['images'][timeframe], start_date, end_date
            )
            filtered_count = len(self.files_info['images'][timeframe])
            print(f"  {timeframe}: {original_count} -> {filtered_count} files")

        # Filter label files
        if self.label_horizon in self.files_info['labels']:
            original_count = len(self.files_info['labels'][self.label_horizon])
            self.files_info['labels'][self.label_horizon] = filter_files_by_date_range(
                self.files_info['labels'][self.label_horizon], start_date, end_date
            )
            filtered_count = len(self.files_info['labels'][self.label_horizon])
            print(f"  labels({self.label_horizon}): {original_count} -> {filtered_count} files")

    def _validate_data_alignment(self):
        """Validate data alignment"""
        print(f"\n🔍 Validating data alignment...")

        # Check if label files exist
        if self.label_horizon not in self.files_info['labels']:
            raise ValueError(f"No label files found for return horizon '{self.label_horizon}'")

        # Check file count consistency across timeframes
        file_counts = [len(self.files_info['images'][tf]) for tf in TIMEFRAMES]
        label_count = len(self.files_info['labels'][self.label_horizon])

        if len(set(file_counts + [label_count])) != 1:
            print(f"⚠️  Inconsistent file counts:")
            for tf in TIMEFRAMES:
                print(f"    {tf}: {len(self.files_info['images'][tf])} files")
            print(f"    labels({self.label_horizon}): {label_count} files")
            raise ValueError("Data file counts are not aligned")

        print(f"✅ Data alignment validation passed: {file_counts[0]} months of data")

        # Validate filename correspondence
        self._validate_filename_alignment()

    def _validate_filename_alignment(self):
        """Validate filename correspondence"""
        # Extract date identifiers from all image files
        image_dates = set()
        for tf in TIMEFRAMES:
            for file_path in self.files_info['images'][tf]:
                date_str = file_path.name.split('_')[-1].replace('.pt', '').replace('.npz', '')
                image_dates.add(date_str)

        # Extract date identifiers from label files
        label_dates = set()
        for file_path in self.files_info['labels'][self.label_horizon]:
            date_str = file_path.name.split('_')[-1].replace('.pt', '').replace('.npz', '')
            label_dates.add(date_str)

        # Check correspondence
        if image_dates != label_dates:
            missing_in_labels = image_dates - label_dates
            missing_in_images = label_dates - image_dates
            if missing_in_labels:
                print(f"⚠️  Missing dates in label files: {missing_in_labels}")
            if missing_in_images:
                print(f"⚠️  Missing dates in image files: {missing_in_images}")
            raise ValueError("Image and label file dates do not correspond")

        print(f"✅ Filename correspondence validation passed")

    def _build_index_mapping(self):
        """Build index mapping"""
        print(f"\n📊 Building data index mapping...")

        self.file_groups = []  # Each group contains files from the same month
        self.cumulative_sizes = []  # Cumulative sample counts
        self.total_samples = 0

        # Sort files by name to ensure alignment
        sorted_files = {}
        for tf in TIMEFRAMES:
            sorted_files[tf] = sorted(self.files_info['images'][tf], key=lambda x: x.name)
        sorted_label_files = sorted(self.files_info['labels'][self.label_horizon], key=lambda x: x.name)

        # Build file groups
        for i in range(len(sorted_label_files)):
            file_group = {
                'images': {tf: sorted_files[tf][i] for tf in TIMEFRAMES},
                'labels': sorted_label_files[i],
                'samples': None  # Determined during lazy loading
            }
            self.file_groups.append(file_group)

        # Quick scan files to get sample counts
        print("Scanning file sizes...")
        for i, group in enumerate(tqdm(self.file_groups)):
            # Get sample count from label file (usually smallest)
            label_data = load_single_file(group['labels'])
            if label_data and 'labels' in label_data:
                if isinstance(label_data['labels'], torch.Tensor):
                    samples = len(label_data['labels'])
                else:
                    samples = len(label_data['labels'])
                group['samples'] = samples
                self.total_samples += samples
                self.cumulative_sizes.append(self.total_samples)
            else:
                raise ValueError(f"Cannot get sample count from label file: {group['labels']}")

        print(f"✅ Index mapping complete: {len(self.file_groups)} file groups, total {self.total_samples:,} samples")

    def _preload_all_data(self):
        """Preload all data to memory"""
        print(f"\n🔄 Preloading all data to memory...")

        all_images = []
        all_labels = []

        for group in tqdm(self.file_groups, desc="Loading files"):
            # Load image files
            images_data = {}
            for tf in TIMEFRAMES:
                img_data = load_single_file(group['images'][tf])
                if img_data and 'images' in img_data:
                    images_data[tf] = img_data['images']
                else:
                    raise ValueError(f"Failed to load image file: {group['images'][tf]}")

            # Stack images (N, 3, 64, 60)
            stacked_images = torch.stack([
                images_data['3min'],
                images_data['15min'],
                images_data['1h']
            ], dim=1).squeeze(2)  # Remove original channel dimension

            # Load label file
            label_data = load_single_file(group['labels'])
            if label_data and 'labels' in label_data:
                labels = label_data['labels']
            else:
                raise ValueError(f"Failed to load label file: {group['labels']}")

            all_images.append(stacked_images)
            all_labels.append(labels)

        # Concatenate all data
        self.all_images = torch.cat(all_images, dim=0)
        self.all_labels = torch.cat(all_labels, dim=0)

        print(f"✅ Preloading complete: images {self.all_images.shape}, labels {self.all_labels.shape}")
        print(f"   Memory usage: {self.all_images.numel() * 4 / 1024**3:.2f} GB (images) + {self.all_labels.numel() * 4 / 1024**2:.2f} MB (labels)")

    def _find_file_group_for_index(self, idx: int) -> Tuple[int, int]:
        """Find file group and local index for given index"""
        for group_idx, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                start_idx = self.cumulative_sizes[group_idx - 1] if group_idx > 0 else 0
                local_idx = idx - start_idx
                return group_idx, local_idx
        raise IndexError(f"Index {idx} out of range")

    def _load_file_group(self, group_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data for specified file group"""
        if group_idx in self.loaded_batches:
            return self.loaded_batches[group_idx]

        group = self.file_groups[group_idx]

        # Load image files
        images_data = {}
        for tf in TIMEFRAMES:
            img_data = load_single_file(group['images'][tf])
            if img_data and 'images' in img_data:
                images_data[tf] = img_data['images']
            else:
                raise ValueError(f"Failed to load image file: {group['images'][tf]}")

        # Stack images (N, 3, 64, 60)
        stacked_images = torch.stack([
            images_data['3min'],
            images_data['15min'],
            images_data['1h']
        ], dim=1).squeeze(2)  # Remove original channel dimension

        # Load label file
        label_data = load_single_file(group['labels'])
        if label_data and 'labels' in label_data:
            labels = label_data['labels']
        else:
            raise ValueError(f"Failed to load label file: {group['labels']}")

        # Cache data (memory management)
        if len(self.loaded_batches) > 5:  # Cache at most 5 batches
            # Remove oldest cache
            oldest_key = min(self.loaded_batches.keys())
            del self.loaded_batches[oldest_key]
            gc.collect()

        self.loaded_batches[group_idx] = (stacked_images, labels)

        return stacked_images, labels

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload_all:
            return self.all_images[idx], self.all_labels[idx]
        else:
            # Dynamic loading
            group_idx, local_idx = self._find_file_group_for_index(idx)
            stacked_images, labels = self._load_file_group(group_idx)
            return stacked_images[local_idx], labels[local_idx]

    def get_data_info(self) -> Dict:
        """Get dataset information"""
        return {
            'total_samples': self.total_samples,
            'num_files': len(self.file_groups),
            'label_horizon': self.label_horizon,
            'timeframes': TIMEFRAMES,
            'image_shape': '(N, 3, 64, 60)',
            'label_shape': '(N,)',
            'preloaded': self.preload_all
        }


def create_training_dataset(data_dir: str,
                          label_horizon: str = '1d',
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          preload_all: bool = False) -> StackedOHLCDataset:
    """
    Create training dataset

    Args:
        data_dir: Data file directory
        label_horizon: Return time horizon ('1h', '1d', '7d' etc)
        start_date: Start date (format: 'YYYY-MM')
        end_date: End date (format: 'YYYY-MM')
        preload_all: Whether to preload all data to memory

    Returns:
        dataset
    """
    print(f"🚀 Creating training dataset...")
    print(f"Configuration:")
    print(f"  - Data directory: {data_dir}")
    print(f"  - Return horizon: {label_horizon}")
    print(f"  - Time range: {start_date or 'start'} ~ {end_date or 'end'}")
    print(f"  - Preload mode: {'Yes' if preload_all else 'No'}")

    # Create dataset
    dataset = StackedOHLCDataset(
        data_dir=Path(data_dir),
        label_horizon=label_horizon,
        start_date=start_date,
        end_date=end_date,
        preload_all=preload_all
    )

    # Display dataset info
    info = dataset.get_data_info()
    print(f"\n📊 Dataset creation complete:")
    print(f"  - Total samples: {info['total_samples']:,}")
    print(f"  - Number of files: {info['num_files']}")
    print(f"  - Image shape: {info['image_shape']}")
    print(f"  - Label shape: {info['label_shape']}")

    return dataset


def create_data_loaders(dataset: StackedOHLCDataset,
                       train_ratio: float = 0.7,
                       batch_size: int = 256,
                       shuffle: bool = True,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders

    Args:
        dataset: OHLC dataset
        train_ratio: Training data ratio
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        random_seed: Random seed for reproducibility

    Returns:
        train_dataloader, val_dataloader
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"📊 Data loaders created:")
    print(f"  - Training samples: {n_train:,} ({len(train_dataloader)} batches)")
    print(f"  - Validation samples: {n_val:,} ({len(val_dataloader)} batches)")
    print(f"  - Batch size: {batch_size}")

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test dataset creation
    DATA_DIR = '/home/craz/crypto/model_training/ohlc_img_dataset'

    print("Testing OHLC training dataset...")

    try:
        # Test dynamic loading mode
        dataset = create_training_dataset(
            data_dir=DATA_DIR,
            label_horizon='4h',
            start_date='2025-03',
            end_date='2025-06',
            preload_all=False
        )

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset, train_ratio=0.7, batch_size=128
        )

        # Test data loading
        print(f"\nTesting data loading...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
            if batch_idx >= 2:  # Test first 3 batches
                break

        print("✅ Dataset testing completed successfully!")

    except Exception as e:
        print(f"❌ Dataset testing failed: {e}")
        import traceback
        traceback.print_exc()