"""Portable study definitions; original model and selection function bodies retained."""
import math, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/study.json').read_text())
MODES=tuple(CONFIG['model_order']); SEEDS=tuple(CONFIG['seeds']); ENDPOINTS=tuple(CONFIG['endpoints'])
MAX_EPOCHS=40; FRAME=512; CHANNELS=4; BLOCK=256; STD_FLOOR=1e-6
ALL_SUBJECTS=tuple(range(1,16)); SOURCE_PARTS=('train','val','test')
SOURCE_COUNTS={'train':38964,'val':12879,'test':12854}
SOURCE_SUBJECTS={k:tuple(int(s[1:]) for s in v) for k,v in json.loads((ROOT/'splits/storage_partitions.json').read_text()).items()}
FOLDS=json.loads((ROOT/'splits/loso.json').read_text())
FOLDS={f'outer_S{person}':FOLDS[f'outer_S{person}'] for person in range(1,16)}
HYPERPARAMS=CONFIG['hyperparams']
PARAMETERS={k:v['parameters'] for k,v in CONFIG['models'].items()}
MACS={k:v['conv_linear_macs'] for k,v in CONFIG['models'].items()}
DEFAULT_CACHE=ROOT/'work/cache'
META_KEYS=('pool_row','source_part','old_row_index','subject','window','y')
def require(condition,message):
    if not condition: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def model_spec(mode):
    require(mode in MODES, 'undeclared model mode')
    base, head = mode.split('_')
    return {'base': base, 'head': head, 'input_channels': 1 if base == 'r0' else 4,
            'widths': [11, 22, 44, 44] if base == 'r2' else [16, 32, 64, 64],
            'encoder_count': 2 if base == 'r2' else 1,
            'head_features': 88 if base == 'r2' else 64,
            'head_hidden': 32 if head == 'mlp' else None,
            'parameters': PARAMETERS[mode], 'conv_linear_macs': MACS[mode]}

class Encoder(nn.Module):
    def __init__(self, channels, widths):
        super().__init__()
        a, b, c, d = widths
        self.net = nn.Sequential(
            nn.Conv1d(channels, a, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(a, b, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(b, c, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(c, d, 7, stride=2, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    """Same historical convolution recipes, fresh study and six explicit heads."""
    def __init__(self, mode):
        super().__init__()
        info = model_spec(mode)
        self.mode = mode
        self.groups = ((0,), (1, 2, 3)) if info['base'] == 'r2' else (
            (tuple(range(info['input_channels']))),)
        # Encoder creation precedes head creation: same-base/head pairs share
        # encoder initialization with the same torch seed.
        self.encoders = nn.ModuleList(Encoder(len(group), tuple(info['widths'])) for group in self.groups)
        self.head = (nn.Sequential(nn.Linear(info['head_features'], 32), nn.ReLU(), nn.Linear(32, 1))
                     if info['head'] == 'mlp' else nn.Linear(info['head_features'], 1))

    def forward(self, x):
        require(x.device.type == 'cpu', 'CPU tensors required')
        info = model_spec(self.mode)
        require(x.ndim == 3 and x.shape[1:] == (info['input_channels'], FRAME), 'invalid model input shape')
        return self.head(torch.cat([encoder(x[:, group, :]) for group, encoder in
                                    zip(self.groups, self.encoders)], dim=1)).squeeze(-1)

def build_model(mode):
    model = Model(mode).cpu()
    require(sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[mode],
            'model parameter count differs')
    return model

def nominal_macs(mode):
    """Analytic Conv1d + Linear MACs/window; excludes bias/ReLU/GAP and I/O."""
    info = model_spec(mode)
    widths = info['widths']
    groups = (1, 3) if info['base'] == 'r2' else (info['input_channels'],)
    result = 0
    for channels in groups:
        length = FRAME
        for output, kernel, padding in zip(widths, (15, 15, 7, 7), (7, 7, 3, 3)):
            length = (length + 2 * padding - kernel) // 2 + 1
            result += length * output * channels * kernel
            channels = output
    features = info['head_features']
    result += features * 32 + 32 if info['head'] == 'mlp' else features
    require(result == MACS[mode], 'nominal MAC declaration differs')
    return result

def selection_mae(raw, labels):
    """Inner-only selection uses unrounded float32 clipped MAE, as before."""
    raw, labels = np.asarray(raw), np.asarray(labels)
    require(raw.dtype == labels.dtype == np.dtype('float32') and raw.ndim == 1
            and raw.shape == labels.shape and raw.size and np.isfinite(raw).all()
            and np.isfinite(labels).all(), 'finite matching float32 selection vectors required')
    return float(np.mean(np.abs(np.clip(raw, 30, 220) - labels), dtype=np.float32))

def select_checkpoints(history):
    """Return two selected epoch numbers, using inner history only.

    Each range replays strict improvement (>1e-4 bpm); first retained epoch wins
    ties/near-ties. Full forty-epoch history is mandatory, with no early stop.
    The function never receives outer/test arrays or scores.
    """
    require(isinstance(history, (list, tuple)) and len(history) == MAX_EPOCHS,
            'exactly forty ordered epochs required')
    best, best_epoch, selected = math.inf, 0, {}
    for epoch, row in enumerate(history, 1):
        require(isinstance(row, dict) and type(row.get('epoch')) is int and row['epoch'] == epoch,
                'nonsequential epoch history')
        require(not any('outer' in key.lower() or 'test' in key.lower() for key in row),
                'outer/test information forbidden in checkpoint history')
        value = row.get('inner_mae_bpm')
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                'finite nonnegative inner MAE required')
        if value < best - HYPERPARAMS['improvement_min']:
            best, best_epoch = float(value), epoch
        if epoch in ENDPOINTS:
            selected[epoch] = {'selected_epoch': best_epoch, 'inner_mae_bpm': best}
    return selected
