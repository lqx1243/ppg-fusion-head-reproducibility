"""Original Pool and Dataset bodies, connected to portable constants only."""
from pathlib import Path
import numpy as np
import torch
import study as r
# Dataset takes base designs; public model names are handled by the runner.
from types import SimpleNamespace
r=SimpleNamespace(**{k:getattr(r,k) for k in dir(r) if not k.startswith('__')})
r.MODES=('r0','r1','r2')
STEMS=('x','y','subject','window');FILES=tuple(n+'.npy' for n in STEMS)

def _people(people):
    people = tuple(people)
    r.require(people and all(isinstance(value, (int, np.integer))
                            and not isinstance(value, (bool, np.bool_)) for value in people),
              'nonempty integer participant list required')
    people = tuple(int(value) for value in people)
    r.require(len(people) == len(set(people)) and set(people) <= set(r.ALL_SUBJECTS),
              'duplicate or undeclared participant')
    return people

def _stats(stats):
    r.require(isinstance(stats, dict) and set(stats) == {'channel_mean', 'channel_std', 'label_mean', 'label_std'},
              'invalid normalization keys')
    mean = np.asarray(stats['channel_mean'], dtype=np.float64)
    std = np.asarray(stats['channel_std'], dtype=np.float64)
    r.require(mean.shape == std.shape == (r.CHANNELS,) and np.isfinite(mean).all()
              and np.isfinite(std).all() and np.all(std >= r.STD_FLOOR), 'invalid channel normalization')
    for name in ('label_mean', 'label_std'):
        value = stats[name]
        r.require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value),
                  'invalid label normalization')
    r.require(stats['label_std'] >= r.STD_FLOOR, 'invalid label standard deviation')
    return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None], \
        float(stats['label_mean']), float(stats['label_std'])

class Pool:
    """Three read-only memmap groups plus explicit small pooled metadata."""

    def __init__(self, cache_root=r.DEFAULT_CACHE):
        self.root, self.arrays, self._closed = Path(cache_root), {}, False
        self.counts, self.offsets = {}, {}
        seen, offset = set(), 0
        try:
            r.require(not self.root.is_symlink() and self.root.is_dir(), 'invalid cache root')
            r.require({path.name for path in self.root.iterdir()} == set(r.SOURCE_PARTS), 'unexpected cache root inventory')
            for part in r.SOURCE_PARTS:
                folder = self.root / part
                r.require(folder.is_dir() and not folder.is_symlink()
                          and {path.name for path in folder.iterdir()} == set(FILES), 'unexpected cache part inventory: ' + part)
                arrays = self.arrays[part] = {}
                for stem in STEMS:
                    path = folder / (stem + '.npy')
                    r.require(path.is_file() and not path.is_symlink(), 'non-regular cache member')
                    arrays[stem] = np.load(path, mmap_mode='r', allow_pickle=False)
                count = len(arrays['y'])
                r.require(count == r.SOURCE_COUNTS[part] and count > 0, 'source row count differs: ' + part)
                for stem, array in arrays.items():
                    shape = (count, r.CHANNELS, r.FRAME) if stem == 'x' else (count,)
                    dtype = {'subject': np.int16, 'window': np.int32}.get(stem, np.float32)
                    r.require(array.shape == shape and array.dtype == np.dtype(dtype), 'invalid cache shape/dtype: ' + part + '/' + stem)
                r.require(set(np.unique(arrays['subject']).tolist()) == set(r.SOURCE_SUBJECTS[part]),
                          'historical source participant assignment differs: ' + part)
                r.require(np.all(arrays['window'] >= 0), 'negative source window key')
                keys = set(zip(arrays['subject'].tolist(), arrays['window'].tolist()))
                r.require(len(keys) == count and not seen.intersection(keys), 'duplicate source (subject, window) key')
                seen.update(keys)
                for start in range(0, count, r.BLOCK):
                    r.require(np.isfinite(arrays['x'][start:start + r.BLOCK]).all()
                              and np.isfinite(arrays['y'][start:start + r.BLOCK]).all(), 'nonfinite source values')
                self.counts[part], self.offsets[part] = count, offset
                offset += count
            self._metadata = {'pool_row': np.arange(offset, dtype=np.int64),
                'source_part': np.concatenate([np.full(self.counts[part], index, dtype=np.int8)
                                               for index, part in enumerate(r.SOURCE_PARTS)]),
                'old_row_index': np.concatenate([np.arange(self.counts[part], dtype=np.int64)
                                                 for part in r.SOURCE_PARTS])}
            for name in ('subject', 'window', 'y'):
                self._metadata[name] = np.concatenate([self.arrays[part][name] for part in r.SOURCE_PARTS])
            for array in self._metadata.values():
                array.flags.writeable = False
        except BaseException:
            self.close()
            raise

    def _open(self):
        r.require(not self._closed, 'source pool is closed')

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        self._open()
        return len(self._metadata['pool_row'])

    def _indices(self, indices):
        self._open()
        indices = np.asarray(indices)
        r.require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(indices) > 0,
                  'nonempty integer pool indices required')
        r.require(np.all(indices >= 0) and np.all(indices < len(self))
                  and np.all(np.diff(indices.astype(np.int64)) > 0), 'pool indices must be unique, increasing and in range')
        return indices.astype(np.int64, copy=False)

    def indices(self, people):
        people = _people(people)
        self._open()
        return np.flatnonzero(np.isin(self._metadata['subject'], people)).astype(np.int64)

    def metadata(self, indices):
        indices = self._indices(indices)
        return {name: self._metadata[name][indices].copy() for name in r.META_KEYS}

    def _read_x(self, indices):
        chunks = []
        for part in r.SOURCE_PARTS:
            offset, count = self.offsets[part], self.counts[part]
            local = indices[(indices >= offset) & (indices < offset + count)] - offset
            if len(local):
                chunks.append(np.asarray(self.arrays[part]['x'][local]))
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def fit_stats(self, indices):
        """Population moments from these explicit training rows, never other roles."""
        indices = self._indices(indices)
        totals, squares = np.zeros(r.CHANNELS, np.float64), np.zeros(r.CHANNELS, np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            for start in range(0, len(indices), r.BLOCK):
                block = np.asarray(self._read_x(indices[start:start + r.BLOCK]), dtype=np.float64)
                totals += block.sum(axis=(0, 2))
                squares += np.square(block).sum(axis=(0, 2))
            mean = totals / (len(indices) * r.FRAME)
            std = np.sqrt(np.maximum(squares / (len(indices) * r.FRAME) - mean ** 2, r.STD_FLOOR ** 2))
        labels = np.asarray(self._metadata['y'][indices], dtype=np.float64)
        stats = {'channel_mean': mean.tolist(), 'channel_std': std.tolist(),
                 'label_mean': float(labels.mean()), 'label_std': max(float(labels.std(ddof=0)), r.STD_FLOOR)}
        _stats(stats)
        return stats

    def dataset(self, people, stats, mode):
        return Dataset(self, self.indices(people), stats, mode)

    def close(self):
        for arrays in self.arrays.values():
            for array in arrays.values():
                mapping = getattr(array, '_mmap', None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
        self._closed = True

class Dataset(torch.utils.data.Dataset):
    """Borrow the Pool; sampler indices select rows without reordering metadata."""

    def __init__(self, pool, indices, stats, mode):
        r.require(mode in r.MODES, 'unknown wrist LOSO model')
        self.pool, self.indices, self.mode = pool, pool._indices(indices).copy(), mode
        self.mean, self.std, self.label_mean, self.label_std = _stats(stats)
        self.channels, self._closed = (1 if mode == 'r0' else r.CHANNELS), False

    def _open(self):
        r.require(not self._closed, 'dataset is closed')
        self.pool._open()

    def __len__(self):
        self._open()
        return len(self.indices)

    def __getitem__(self, index):
        self._open()
        r.require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
                  and 0 <= index < len(self.indices), 'invalid dataset row index')
        row = int(self.indices[index])
        for part in r.SOURCE_PARTS:
            offset = self.pool.offsets[part]
            if offset <= row < offset + self.pool.counts[part]:
                arrays, local = self.pool.arrays[part], row - offset
                break
        x = np.array(arrays['x'][local, :self.channels], dtype=np.float32, copy=True)
        x = (x - self.mean[:self.channels]) / self.std[:self.channels]
        y = (float(arrays['y'][local]) - self.label_mean) / self.label_std
        r.require(np.isfinite(x).all() and np.isfinite(y), 'nonfinite normalized row')
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def close(self):
        self._closed = True
