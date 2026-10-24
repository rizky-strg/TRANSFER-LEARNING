"""Langkah 3: latih ResNet-18 dengan 3 mode.
python train.py feature | partial | scratch"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from common import build_model, IMG_SIZE, MEAN, STD

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["feature", "partial", "scratch"])
p.add_argument("--data", default="dataset")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize(MEAN.tolist(), STD.tolist())
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(), norm])
tf_val = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])
ds_tr = datasets.ImageFolder(f"{a.data}/train", tf_train)
ds_va = datasets.ImageFolder(f"{a.data}/val", tf_val)
dl_tr = DataLoader(ds_tr, a.bs, shuffle=True)
dl_va = DataLoader(ds_va, a.bs)
classes = ds_tr.classes

m = build_model("resnet18", len(classes), pretrained=(a.mode != "scratch"))
trainable = {"feature": {"fc"}, "partial": {"layer4", "fc"}, "scratch": None}[a.mode]
if trainable is not None:
    for n, prm in m.named_parameters():
        prm.requires_grad = n.split(".")[0] in trainable
if a.mode == "feature":
    groups = [{"params": m.fc.parameters(), "lr": 1e-3}]
elif a.mode == "partial":
    groups = [{"params": m.layer4.parameters(), "lr": 1e-4}, {"params": m.fc.parameters(), "lr": 1e-3}]
else:
    groups = [{"params": m.parameters(), "lr": 1e-3}]
opt = torch.optim.Adam(groups)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
loss_fn = nn.CrossEntropyLoss()
m.to(dev)

def run(dl, train):
    m.train(train)
    if train and trainable is not None:      # lapisan beku tetap eval() (BatchNorm)
        for n, child in m.named_children():
            if n not in trainable: child.eval()
    tot = correct = 0; loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            out = m(x); loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
    return loss_sum / tot, correct / tot

hist = {"mode": a.mode, "classes": classes, "train_acc": [], "val_acc": [], "val_loss": [], "epoch_time": []}
best, t0, first90 = 0.0, time.time(), None
Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
for ep in range(1, a.epochs + 1):
    t1 = time.time()
    _, tra = run(dl_tr, True); vl, vaa = run(dl_va, False); sched.step()
    hist["train_acc"].append(tra); hist["val_acc"].append(vaa); hist["val_loss"].append(vl)
    hist["epoch_time"].append(time.time() - t1)
    if vaa >= 0.9 and first90 is None: first90 = ep
    if vaa >= best:
        best = vaa
        torch.save({"state": m.state_dict(), "classes": classes, "arch": "resnet18", "mode": a.mode},
                   f"models/resnet18_{a.mode}.pt")
    print(f"[{a.mode}] ep {ep:2d}/{a.epochs} train_acc={tra:.3f} val_acc={vaa:.3f} val_loss={vl:.3f}")
hist.update(best_val_acc=best, train_time_s=time.time() - t0, epoch_ge_90=first90)
json.dump(hist, open(f"results/{a.mode}.json", "w"), indent=2)
print(f"Selesai. Akurasi val terbaik={best:.3f}, waktu={hist['train_time_s']:.0f}s, epoch>=90%: {first90}")
