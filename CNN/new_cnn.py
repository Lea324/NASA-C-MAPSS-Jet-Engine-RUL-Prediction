# 1D-CNN for RUL prediction on NASA C-MAPSS FD001
# Focus: push test MAE down via deep ensemble + TTA + all-data training
import os, time, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR      = r"C:\Engine URL Prediction\Aircraft Engine Dataset\FD001"
SEQ_LEN       = 30
WINDOW_STRIDE = 2
RUL_CLIP      = 125.0
OVER_PEN      = 5.0
UNDER_PEN     = 1.0

BATCH_SIZE   = 256
MAX_EPOCHS   = 250
PATIENCE     = 20
N_ENSEMBLE   = 8         # deep ensemble size for the final test prediction
N_TTA        = 5         # test-time augmentation copies per window
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y_test   = (pd.read_csv(os.path.join(DATA_DIR, "RUL_targets.csv"))["true_RUL"]
            .astype(float).to_numpy())

drop_cols = [c for c in ["unit_number", "time_in_cycles", "max_cycle", "RUL", "health_stage"]
             if c in train_df.columns]
FEATURES = [c for c in train_df.columns if c not in drop_cols]

# ---- Feature selection: drop sensors whose std is near-zero ----------
raw = train_df[FEATURES].to_numpy(np.float32)
keep_mask = raw.std(axis=0) > 1e-4
FEATURES = [f for f, k in zip(FEATURES, keep_mask) if k]
print(f"Kept {len(FEATURES)} features: {FEATURES}")

# ---- Scaling ---------------------------------------------------------
scaler = StandardScaler().fit(train_df[FEATURES].to_numpy(np.float32))
train_df[FEATURES] = scaler.transform(train_df[FEATURES].to_numpy(np.float32))
test_df[FEATURES]  = scaler.transform(test_df[FEATURES].to_numpy(np.float32))

# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------
def build_windows(df, feature_cols, seq_len, rul_clip=125.0,
                  last_only=False, stride=1):
    """Sliding windows -> (n_windows, seq_len, n_features)."""
    Xs, ys, gs = [], [], []
    has_rul = "RUL" in df.columns

    for unit, grp in df.groupby("unit_number", sort=True):
        grp = grp.sort_values("time_in_cycles")
        feats = grp[feature_cols].to_numpy(np.float32)
        rul = np.minimum(grp["RUL"].to_numpy(np.float32), rul_clip) if has_rul else None

        if len(feats) < seq_len:
            pad = seq_len - len(feats)
            feats = np.pad(feats, ((pad, 0), (0, 0)), mode="edge")
            if rul is not None:
                rul = np.pad(rul, (pad, 0), mode="edge")

        if last_only:
            Xs.append(feats[-seq_len:])
            gs.append(unit)
        else:
            for end in range(seq_len, len(feats) + 1, stride):
                Xs.append(feats[end - seq_len:end])
                ys.append(rul[end - 1])
                gs.append(unit)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, np.float32) if ys else None
    return X, y, np.asarray(gs)

X_train, y_train, groups_train = build_windows(
    train_df, FEATURES, SEQ_LEN, RUL_CLIP, last_only=False, stride=WINDOW_STRIDE)
X_test, _, _ = build_windows(
    test_df, FEATURES, SEQ_LEN, RUL_CLIP, last_only=True)

print(f"Train windows: {X_train.shape} | Test engines: {X_test.shape}")

# ------------------------------------------------------------------
# Dataset with augmentation (training only)
# ------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)                 # sensor noise
            scale = 1.0 + 0.05 * torch.randn(x.shape[1])       # per-channel scale
            x = x * scale
            shift = int(torch.randint(-3, 4, (1,)).item())     # time shift
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)
            if torch.rand(1).item() < 0.3:                     # time cutout
                mask_len = int(torch.randint(2, 6, (1,)).item())
                start = int(torch.randint(0, max(1, x.shape[0] - mask_len), (1,)).item())
                x[start:start + mask_len, :] = 0.0
        return x if self.y is None else (x, self.y[i])

# ------------------------------------------------------------------
# Conservative loss
# ------------------------------------------------------------------
class ConservativeLoss(nn.Module):
    def __init__(self, over_penalty=5.0, under_penalty=1.0):
        super().__init__()
        self.over, self.under = over_penalty, under_penalty

    def forward(self, pred, target):
        err = pred - target
        return torch.where(err > 0, self.over * err ** 2, self.under * err ** 2).mean()

# ------------------------------------------------------------------
# Model: residual dilated 1D CNN
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(ch)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(x + h)

class RULCNN(nn.Module):
    def __init__(self, n_features, channels=(32, 64, 64),
                 kernel_size=3, dropout=0.3, input_drop=0.1):
        super().__init__()
        self.input_drop = nn.Dropout(input_drop)
        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        # stack of residual dilated blocks + downsampling
        blocks = []
        for i, ch in enumerate(channels):
            if i > 0:                                   # downsample between stages
                blocks.append(nn.MaxPool1d(2))
                if channels[i] != channels[i - 1]:
                    blocks.append(nn.Conv1d(channels[i - 1], ch, 1))
            blocks.append(ResBlock(ch, kernel_size, dilation=1, dropout=dropout))
            blocks.append(ResBlock(ch, kernel_size, dilation=2, dropout=dropout))
        self.body = nn.Sequential(*blocks)
        # head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # (B, seq_len, n_features)
        x = self.input_drop(x).transpose(1, 2)
        x = self.stem(x)
        x = self.body(x)
        return self.head(x).squeeze(-1)

# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch_size=512):
    model.eval()
    loader = DataLoader(WindowDataset(X), batch_size=batch_size, shuffle=False)
    return np.concatenate([model(xb.to(DEVICE)).cpu().numpy() for xb in loader])

@torch.no_grad()
def predict_tta(model, X, n_tta=5, batch_size=512):
    """Average predictions over n_tta jittered copies of each window."""
    model.eval()
    preds = [predict(model, X, batch_size)]
    for _ in range(n_tta - 1):
        Xj = X.copy()
        # small noise + small time shift (consistent across features)
        Xj += 0.005 * np.random.randn(*Xj.shape).astype(np.float32)
        shifts = np.random.randint(-2, 3, size=Xj.shape[0])
        for i, s in enumerate(shifts):
            if s != 0:
                Xj[i] = np.roll(Xj[i], shifts=s, axis=0)
        preds.append(predict(model, Xj, batch_size))
    return np.mean(preds, axis=0)

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def fit(config, X_tr, y_tr, X_va, y_va, seed=SEED, verbose=False):
    """Train one CNN with early stopping on validation MAE."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = RULCNN(
        n_features=X_tr.shape[2],
        channels=config["channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        input_drop=config["input_drop"],
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(),
                              lr=config["lr"],
                              weight_decay=config["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_EPOCHS)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        optim, [warmup, cosine], milestones=[5])

    loss_fn = ConservativeLoss(OVER_PEN, UNDER_PEN)
    train_loader = DataLoader(
        WindowDataset(X_tr, y_tr, augment=True),
        batch_size=config["batch_size"], shuffle=True, drop_last=True)

    best_mae, best_state, wait = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        sched.step()

        if X_va is not None:
            val_mae = mean_absolute_error(y_va, np.clip(predict(model, X_va), 0, RUL_CLIP))
            if verbose:
                print(f"    epoch {epoch+1:3d} | val MAE {val_mae:.4f}")
            if val_mae < best_mae - 1e-4:
                best_mae, best_state, wait = val_mae, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        else:
            # no validation: keep last state
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mae

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
best_config = {
    "channels":     (32, 64, 64),
    "kernel_size":  3,
    "dropout":      0.3,
    "input_drop":   0.1,
    "lr":           1e-3,
    "batch_size":   BATCH_SIZE,
    "weight_decay": 1e-4,
}

# ------------------------------------------------------------------
# Stage 1: GroupKFold for an HONEST OOF MAE (reporting only)
# ------------------------------------------------------------------
print("\n===== Stage 1: GroupKFold for OOF estimate =====")
N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)
oof_pred = np.zeros(len(y_train))
oof_maes = []

for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups_train)):
    m, mae = fit(best_config, X_train[tr], y_train[tr],
                 X_train[va], y_train[va], seed=SEED + fold)
    oof_pred[va] = np.clip(predict(m, X_train[va]), 0, RUL_CLIP)
    oof_maes.append(mae)
    print(f"  fold {fold+1}/{N_SPLITS}: val MAE = {mae:.4f}")

mae_oof = mean_absolute_error(y_train, oof_pred)
print(f"OOF MAE (honest estimate): {mae_oof:.4f}")

# ------------------------------------------------------------------
# Stage 2: Deep ensemble on ALL training engines
# Each model sees 100% of engines -> better test predictions
# ------------------------------------------------------------------
print(f"\n===== Stage 2: Deep ensemble ({N_ENSEMBLE} models on ALL data) =====")
train_begin_time = time.time()
ensemble = []
for k in range(N_ENSEMBLE):
    m, _ = fit(best_config, X_train, y_train, None, None, seed=SEED + 100 + k)
    ensemble.append(m)
    print(f"  ensemble member {k+1}/{N_ENSEMBLE} trained")
train_end_time = time.time()
training_time = train_end_time - train_begin_time

# ------------------------------------------------------------------
# Test prediction: ensemble + TTA
# ------------------------------------------------------------------
predict_begin_time = time.time()
member_preds = [predict_tta(m, X_test, n_tta=N_TTA) for m in ensemble]
y_pred = np.mean(member_preds, axis=0)
predict_end_time = time.time()
prediction_time = predict_end_time - predict_begin_time
y_pred_capped = np.clip(y_pred, 0, RUL_CLIP)

# Also measure training MAE (in-sample, averaged over ensemble + TTA is overkill)
train_preds = np.mean([predict(m, X_train) for m in ensemble], axis=0)
mae_train = mean_absolute_error(y_train, np.clip(train_preds, 0, RUL_CLIP))

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred_capped)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_capped))
r2   = r2_score(y_test, y_pred_capped)

print("\n===== Model Performance =====")
print(f"Predict MAE: {mae:.4f}")
print(f"Training MAE (in-sample): {mae_train:.4f}")
print(f"OOF MAE (honest): {mae_oof:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

print("\n===== Time =====")
print(f"Training Time (ensemble): {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

print("\n===== Hyperparameters =====")
for k, v in best_config.items():
    print(f"{k}: {v}")
print(f"Ensemble size: {N_ENSEMBLE} | TTA copies: {N_TTA}")

# ------------------------------------------------------------------
# Safety check: over- / under-estimation
# ------------------------------------------------------------------
over_mask  = y_pred_capped > y_test
under_mask = y_pred_capped < y_test
over_count  = int(np.sum(over_mask))
under_count = int(np.sum(under_mask))
over_ratio  = over_count / len(y_test) * 100
under_ratio = under_count / len(y_test) * 100
mean_over  = np.mean(y_pred_capped[over_mask] - y_test[over_mask]) if over_count  else 0.0
mean_under = np.mean(y_test[under_mask] - y_pred_capped[under_mask]) if under_count else 0.0

print("\n===== Overestimation and Underestimation Analysis =====")
print(f"Overestimation Count: {over_count}/{len(y_test)} ({over_ratio:.2f}%)")
print(f"Underestimation Count: {under_count}/{len(y_test)} ({under_ratio:.2f}%)")
print(f"Mean Overestimation: {mean_over:.4f}")
print(f"Mean Underestimation: {mean_under:.4f}")