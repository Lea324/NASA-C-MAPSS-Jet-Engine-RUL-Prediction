# 1D-CNN for RUL prediction on NASA C-MAPSS FD001
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
DATA_DIR   = r"C:\Engine URL Prediction\Aircraft Engine Dataset\FD001"
SEQ_LEN    = 30          # sliding-window length fed to the CNN
RUL_CLIP   = 125.0       # piecewise-linear RUL cap (standard for FD001)
OVER_PEN   = 3.0         # penalty when predicting RUL too high (unsafe)
UNDER_PEN  = 1.0         # penalty when predicting RUL too low  (conservative)
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE   = 15          # early-stopping patience
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y_test   = (pd.read_csv(os.path.join(DATA_DIR, "RUL_targets.csv"))["true_RUL"]
            .astype(float).to_numpy())

# Same idea as  X = df.drop(['max_cycle','RUL','health_stage'], axis=1)
# but we also drop unit_number / time_in_cycles: the CNN gets the time
# dimension from the window itself, so feeding cycle index would leak.
drop_cols = [c for c in ["unit_number", "time_in_cycles", "max_cycle", "RUL", "health_stage", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
             if c in train_df.columns]
FEATURES = [c for c in train_df.columns if c not in drop_cols]
print(f"Using {len(FEATURES)} features: {FEATURES}")

# ---- Feature scaling (fit on train only, applied to both) ----------
scaler = StandardScaler().fit(train_df[FEATURES].to_numpy(np.float32))
train_df[FEATURES] = scaler.transform(train_df[FEATURES].to_numpy(np.float32))
test_df[FEATURES]  = scaler.transform(test_df[FEATURES].to_numpy(np.float32))

# ------------------------------------------------------------------
# Windowing: turn per-cycle rows into per-engine sequences
# ------------------------------------------------------------------
def build_windows(df, feature_cols, seq_len, rul_clip=125.0, last_only=False):
    """Sliding windows of shape (n_windows, seq_len, n_features).

    last_only=True  -> one window per unit (the last `seq_len` cycles),
                       used for the test set.
    """
    Xs, ys, gs = [], [], []
    has_rul = "RUL" in df.columns

    for unit, grp in df.groupby("unit_number", sort=True):
        grp = grp.sort_values("time_in_cycles")
        feats = grp[feature_cols].to_numpy(np.float32)
        rul = np.minimum(grp["RUL"].to_numpy(np.float32), rul_clip) if has_rul else None

        # edge-pad units with a history shorter than seq_len
        if len(feats) < seq_len:
            pad = seq_len - len(feats)
            feats = np.pad(feats, ((pad, 0), (0, 0)), mode="edge")
            if rul is not None:
                rul = np.pad(rul, (pad, 0), mode="edge")

        if last_only:
            Xs.append(feats[-seq_len:])
            gs.append(unit)
        else:
            for end in range(seq_len, len(feats) + 1):
                Xs.append(feats[end - seq_len:end])
                ys.append(rul[end - 1])
                gs.append(unit)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, np.float32) if ys else None
    return X, y, np.asarray(gs)

X_train, y_train, groups_train = build_windows(train_df, FEATURES, SEQ_LEN, RUL_CLIP, last_only=False)
X_test,  _,       _            = build_windows(test_df,  FEATURES, SEQ_LEN, RUL_CLIP, last_only=True)

print(f"Train windows: {X_train.shape} | Test engines: {X_test.shape}")

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i] if self.y is None else (self.X[i], self.y[i])

# ------------------------------------------------------------------
# Conservative loss  (the differentiable twin of `conservative_predict`)
#
# XGBoost got grad = 2*w*(pred-true), hess = 2*w  ->  equivalent to a
# weighted squared error.  So we simply weight the squared error.
# ------------------------------------------------------------------
class ConservativeLoss(nn.Module):
    def __init__(self, over_penalty=5.0, under_penalty=1.0):
        super().__init__()
        self.over, self.under = over_penalty, under_penalty

    def forward(self, pred, target):
        err = pred - target
        loss = torch.where(err > 0, self.over * err ** 2, self.under * err ** 2)
        return loss.mean()

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class RULCNN(nn.Module):
    def __init__(self, n_features, channels=(64, 128, 64),
                 kernel_size=3, dropout=0.3):
        super().__init__()
        layers, in_ch = [], n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: (B, seq_len, n_features)
        x = x.transpose(1, 2)             # -> (B, n_features, seq_len)
        return self.head(self.conv(x)).squeeze(-1)

# ------------------------------------------------------------------
# Train / predict helpers
# ------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch_size=512):
    model.eval()
    loader = DataLoader(WindowDataset(X), batch_size=batch_size, shuffle=False)
    out = [model(xb.to(DEVICE)).cpu().numpy() for xb in loader]
    return np.concatenate(out)

def fit(config, X_tr, y_tr, X_va, y_va, verbose=False):
    """Train one CNN with early stopping on validation MAE."""
    model = RULCNN(
        n_features=X_tr.shape[2],
        channels=config["channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
    ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=config["lr"],
                             weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=5)
    loss_fn = ConservativeLoss(OVER_PEN, UNDER_PEN)

    train_loader = DataLoader(WindowDataset(X_tr, y_tr),
                              batch_size=config["batch_size"],
                              shuffle=True, drop_last=True)

    best_mae, best_state, wait = np.inf, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

        val_pred = predict(model, X_va)
        val_mae = mean_absolute_error(y_va, np.clip(val_pred, 0, RUL_CLIP))
        sched.step(val_mae)

        if verbose:
            print(f"    epoch {epoch+1:3d} | val MAE {val_mae:.4f}")

        if val_mae < best_mae - 1e-4:
            best_mae, best_state, wait = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_mae

# ------------------------------------------------------------------
# (Optional) small random hyper-parameter search with GroupKFold
# ------------------------------------------------------------------
DO_SEARCH    = True    # set True to tune (slow on CPU)
N_SEARCH_IT  = 5
SEARCH_SPACE = {
    "channels":     [(32, 64, 32), (64, 128, 64), (64, 64, 64), (128, 64, 32)],
    "kernel_size":  [3, 5],
    "dropout":      [0.1, 0.3, 0.5],
    "lr":           [1e-3, 5e-4, 1e-4],
    "batch_size":   [128, 256, 512],
    "weight_decay": [0.0, 1e-5, 1e-4],
}

best_config = {                      # default config
    "channels": (64, 128, 64),
    "kernel_size": 3,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": BATCH_SIZE,
    "weight_decay": 0.0,
}

if DO_SEARCH:
    rng = random.Random(SEED)
    cv3 = GroupKFold(n_splits=3)
    best_score = np.inf
    print("\n===== Random search =====")
    for it in range(N_SEARCH_IT):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        fold_scores = []
        for tr, va in cv3.split(X_train, y_train, groups_train):
            _, mae = fit(cfg, X_train[tr], y_train[tr], X_train[va], y_train[va])
            fold_scores.append(mae)
        score = float(np.mean(fold_scores))
        print(f"  [{it+1}/{N_SEARCH_IT}] CV MAE {score:.4f} | {cfg}")
        if score < best_score:
            best_score, best_config = score, cfg
    print(f"Best config (CV MAE {best_score:.4f}): {best_config}")

# ------------------------------------------------------------------
# Final training: GroupKFold ensemble (uses every engine for training)
# ------------------------------------------------------------------
print("\n===== Final training (GroupKFold ensemble) =====")
N_SPLITS = 10
gkf = GroupKFold(n_splits=N_SPLITS)

oof_pred   = np.zeros(len(y_train))
fold_models, fold_maes = [], []

train_begin_time = time.time()
for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups_train)):
    model, val_mae = fit(best_config, X_train[tr], y_train[tr],
                         X_train[va], y_train[va])
    fold_models.append(model)
    fold_maes.append(val_mae)
    oof_pred[va] = np.clip(predict(model, X_train[va]), 0, RUL_CLIP)
    print(f"  fold {fold+1}/{N_SPLITS}: val MAE = {val_mae:.4f}")

train_end_time = time.time()
training_time = train_end_time - train_begin_time

# ------------------------------------------------------------------
# Test prediction (average of the fold models)
# ------------------------------------------------------------------
predict_begin_time = time.time()
y_pred = np.mean([predict(m, X_test) for m in fold_models], axis=0)
predict_end_time = time.time()
prediction_time = predict_end_time - predict_begin_time

y_pred_capped = np.clip(y_pred, 0, RUL_CLIP)   # never predict negative RUL

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
mae      = mean_absolute_error(y_test, y_pred_capped)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred_capped))
r2       = r2_score(y_test, y_pred_capped)
mae_oof  = mean_absolute_error(y_train, oof_pred)
mae_train= mean_absolute_error(y_train, np.clip(
              np.mean([predict(m, X_train) for m in fold_models], axis=0), 0, RUL_CLIP))

print("\n===== Model Performance =====")
print(f"Predict MAE: {mae:.4f}")
print(f"Training MAE (in-sample): {mae_train:.4f}")
print(f"OOF MAE (out-of-fold on train): {mae_oof:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

print("\n===== Time =====")
print(f"Training Time ({N_SPLITS} folds): {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

print("\n===== Hyperparameters =====")
for k, v in best_config.items():
    print(f"{k}: {v}")
print(f"Fold val MAEs: {[round(m, 4) for m in fold_maes]}")
print(f"Mean fold MAE: {np.mean(fold_maes):.4f}")

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