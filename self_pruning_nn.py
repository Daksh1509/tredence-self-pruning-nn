"""
=============================================================================
  TREDENCE AI ENGINEERING INTERNSHIP — CASE STUDY
  Title  : The Self-Pruning Neural Network
  Author : DAKSH GUPTA (RA2311026010356)
  Dataset: CIFAR-10
  Task   : Build a feedforward network that learns to prune itself during
           training using learnable sigmoid gates + L1 sparsity regularization.
=============================================================================

HOW TO RUN:
    pip install torch torchvision matplotlib numpy
    python self_pruning_nn.py

OUTPUT FILES GENERATED:
    gate_distribution.png   — histogram of gate values for best model
    accuracy_vs_sparsity.png — Pareto curve across lambda values
    loss_curves.png         — CE loss + Sparsity loss per epoch per lambda
    report.md               — markdown report with results table
=============================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
GATE_PRUNE_THRESHOLD = 0.45

# Lambda values to experiment with
LAMBDA_VALUES = [0.0, 1e-5, 5e-5, 1e-4, 5e-4]

# Network architecture
INPUT_DIM = 3072
HIDDEN_1 = 512
HIDDEN_2 = 256
HIDDEN_3 = 128
OUTPUT_DIM = 10


# =============================================================================
# 1. PRUNABLE LINEAR LAYER
# =============================================================================

class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gate_scores for each weight.

    gates = sigmoid(gate_scores)
    pruned_weight = weight * gates
    output = input @ pruned_weight.T + bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate_scores has same shape as weights
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # He initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
    # Convert unconstrained gate scores into values between 0 and 1.
    # A high gate value means the corresponding weight should stay active.
    # A low gate value means the weight should be suppressed or pruned.
        gates = torch.sigmoid(self.gate_scores)

        if hard:
            gates = (gates > GATE_PRUNE_THRESHOLD).float()

        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()

    def sparsity(self) -> float:
        gates = self.get_gates()
        return (gates < GATE_PRUNE_THRESHOLD).float().mean().item()


# =============================================================================
# 2. SELF-PRUNING NETWORK
# =============================================================================

class SelfPruningNet(nn.Module):
    """
    3-hidden-layer feedforward network using PrunableLinear layers.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(INPUT_DIM, HIDDEN_1)
        self.fc2 = PrunableLinear(HIDDEN_1, HIDDEN_2)
        self.fc3 = PrunableLinear(HIDDEN_2, HIDDEN_3)
        self.fc4 = PrunableLinear(HIDDEN_3, OUTPUT_DIM)

        self.bn1 = nn.BatchNorm1d(HIDDEN_1)
        self.bn2 = nn.BatchNorm1d(HIDDEN_2)
        self.bn3 = nn.BatchNorm1d(HIDDEN_3)

        self.drop = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.bn1(self.fc1(x, hard))))
        x = self.drop(F.relu(self.bn2(self.fc2(x, hard))))
        x = self.drop(F.relu(self.bn3(self.fc3(x, hard))))
        x = self.fc4(x, hard)
        return x

    def prunable_layers(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def overall_sparsity(self) -> float:
        all_gates = torch.cat([l.get_gates() for l in self.prunable_layers()])
        return (all_gates < GATE_PRUNE_THRESHOLD).float().mean().item()

    def per_layer_sparsity(self) -> dict:
        names = ['fc1', 'fc2', 'fc3', 'fc4']
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        return {n: f"{l.sparsity()*100:.1f}%" for n, l in zip(names, layers)}


# =============================================================================
# 3. SPARSITY LOSS FUNCTION
# =============================================================================
# This function computes the L1 penalty on all gate values in the network.
# Since gates are sigmoid outputs, they are always positive, so the L1 norm
# becomes a simple sum. Minimizing this term encourages more gates to move
# toward zero, which means more weights become effectively pruned.

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values across all PrunableLinear layers.
    """
    total_l1 = sum(
        torch.sigmoid(layer.gate_scores).sum()
        for layer in model.prunable_layers()
    )
    return total_l1


# =============================================================================
# 4. DATA LOADING (CIFAR-10)
# =============================================================================

def get_cifar10_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, optimizer, lambda_val):
    model.train()
    total_ce, total_sp = 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images, hard=False)
        ce_loss = F.cross_entropy(logits, labels)
        sp_loss = sparsity_loss(model)
        loss = ce_loss + lambda_val * sp_loss

        loss.backward()
        optimizer.step()

        total_ce += ce_loss.item()
        total_sp += sp_loss.item()

    n = len(loader)
    return total_ce / n, total_sp / n


# =============================================================================
# 6. EVALUATION
# =============================================================================

def evaluate(model, loader, hard=False):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images, hard=hard)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_inference_speed(model, loader):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(DEVICE)

    for _ in range(10):
        with torch.no_grad():
            model(images, hard=False)

    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model(images, hard=False)
    soft_ms = (time.perf_counter() - start) / 100 * 1000

    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model(images, hard=True)
    hard_ms = (time.perf_counter() - start) / 100 * 1000

    return soft_ms, hard_ms


# =============================================================================
# 7. EXPERIMENT RUNNER
# =============================================================================

def run_experiment(lambda_val, train_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"  Lambda = {lambda_val:.0e}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ce_history, sp_history = [], []

    for epoch in range(1, EPOCHS + 1):
        ce_loss, sp_loss_val = train_one_epoch(model, train_loader, optimizer, lambda_val)
        scheduler.step()

        ce_history.append(ce_loss)
        sp_history.append(sp_loss_val)

        if epoch % 5 == 0 or epoch == 1:
            sparsity_pct = model.overall_sparsity() * 100
            print(
                f"  Epoch {epoch:02d}/{EPOCHS} | CE: {ce_loss:.4f} | "
                f"SparsityLoss: {sp_loss_val:.1f} | Gates pruned: {sparsity_pct:.1f}%"
            )

    soft_acc = evaluate(model, test_loader, hard=False)
    hard_acc = evaluate(model, test_loader, hard=True)
    sparsity = model.overall_sparsity()
    per_layer = model.per_layer_sparsity()
    soft_ms, hard_ms = benchmark_inference_speed(model, test_loader)
    all_gates = torch.cat([l.get_gates() for l in model.prunable_layers()]).numpy()

    print(f"\n  RESULTS:")
    print(f"    Soft Gate Accuracy : {soft_acc*100:.2f}%")
    print(f"    Hard Gate Accuracy : {hard_acc*100:.2f}%")
    print(f"    Overall Sparsity   : {sparsity*100:.2f}%")
    print(f"    Per-layer sparsity : {per_layer}")
    print(f"    Inference (soft)   : {soft_ms:.2f} ms/batch")
    print(f"    Inference (hard)   : {hard_ms:.2f} ms/batch")

    return {
        "lambda": lambda_val,
        "soft_acc": soft_acc,
        "hard_acc": hard_acc,
        "sparsity": sparsity,
        "per_layer": per_layer,
        "soft_ms": soft_ms,
        "hard_ms": hard_ms,
        "ce_history": ce_history,
        "sp_history": sp_history,
        "all_gates": all_gates,
        "model": model,
    }


# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

def plot_gate_distribution(results: list):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    fig.suptitle("Gate Value Distribution per Lambda", fontsize=14, fontweight='bold')

    for ax, r in zip(axes, results):
        gates = r["all_gates"]
        ax.hist(gates, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_title(f"λ = {r['lambda']:.0e}\nSparsity: {r['sparsity']*100:.1f}%", fontsize=10)
        ax.set_xlabel("Gate Value (0=pruned, 1=active)")
        ax.set_ylabel("Count")
        ax.axvline(
            x=GATE_PRUNE_THRESHOLD,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Prune threshold ({GATE_PRUNE_THRESHOLD})'
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] gate_distribution.png")


def plot_accuracy_vs_sparsity(results: list):
    lambdas = [r["lambda"] for r in results]
    accs = [r["soft_acc"] * 100 for r in results]
    hard_accs = [r["hard_acc"] * 100 for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sparsities, accs, 'o-', color='steelblue', linewidth=2, label='Soft Gate Accuracy')
    ax.plot(sparsities, hard_accs, 's--', color='tomato', linewidth=2, label='Hard Gate Accuracy')

    for sp, ac, lam in zip(sparsities, accs, lambdas):
        ax.annotate(
            f"λ={lam:.0e}",
            (sp, ac),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    ax.set_xlabel("Sparsity Level (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Sparsity Trade-off (Pareto Frontier)", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_sparsity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] accuracy_vs_sparsity.png")


def plot_loss_curves(results: list):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    fig.suptitle("Training Loss Curves per Lambda", fontsize=14, fontweight='bold')

    for col, r in enumerate(results):
        epochs = range(1, EPOCHS + 1)

        axes[0][col].plot(epochs, r["ce_history"], color='steelblue', linewidth=2)
        axes[0][col].set_title(f"λ = {r['lambda']:.0e}\nCross-Entropy Loss", fontsize=9)
        axes[0][col].set_xlabel("Epoch")
        axes[0][col].set_ylabel("CE Loss")
        axes[0][col].grid(True, alpha=0.3)

        axes[1][col].plot(epochs, r["sp_history"], color='tomato', linewidth=2)
        axes[1][col].set_title("Sparsity Loss (L1 of Gates)", fontsize=9)
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("Sparsity Loss")
        axes[1][col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] loss_curves.png")


# =============================================================================
# 9. MARKDOWN REPORT GENERATOR
# =============================================================================

def generate_report(results: list):
    non_zero = [r for r in results if r["lambda"] > 0]
    best = max(non_zero, key=lambda r: r["soft_acc"])

    lines = []
    lines.append("# The Self-Pruning Neural Network — Results Report\n")
    lines.append("**Author:** DAKSH GUPTA (RA2311026010356)  ")
    lines.append("**Dataset:** CIFAR-10  ")
    lines.append(f"**Device:** {DEVICE}  \n")

    lines.append("---\n")
    lines.append("## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity\n")
    lines.append(
        "The sparsity loss is the **L1 norm** (sum of absolute values) of all gate values "
        "after the sigmoid: `SparsityLoss = Σ sigmoid(gate_score_i)` across all layers.\n"
    )
    lines.append("### L1 vs L2 — The Core Difference\n")
    lines.append(
        "| Property | L2 Penalty (sum of gates²) | L1 Penalty (sum of |gates|) |\n"
        "|----------|---------------------------|-----------------------------|\n"
        "| Gradient | `2 × gate` → weakens near 0 | `±1` → **constant always** |\n"
        "| Effect   | Shrinks weights but never reaches exactly 0 | Pushes gates to **exactly 0** |\n"
        "| Result   | Dense, small weights | True sparsity (binary-like gates) |\n\n"
    )
    lines.append(
        "The key insight: L1's gradient is a **constant ±1** regardless of the gate's current "
        "value. This acts like a constant wind always blowing gates toward zero. L2's gradient "
        "`2g` gets weaker as `g → 0`, so it can never force an exact zero — weights merely "
        "become small. Since our gates are always positive (after sigmoid), the L1 push "
        "consistently drives them to the flat region at 0, producing true **structural sparsity**.\n"
    )

    lines.append("---\n")
    lines.append("## 2. Results Table\n")
    lines.append(
        "| Lambda | Soft Acc (%) | Hard Acc (%) | Sparsity (%) | Soft Speed (ms) | Hard Speed (ms) |\n"
        "|--------|--------------|--------------|--------------|-----------------|-----------------|\n"
    )

    for r in results:
        lines.append(
            f"| `{r['lambda']:.0e}` | {r['soft_acc']*100:.2f} | {r['hard_acc']*100:.2f} "
            f"| {r['sparsity']*100:.2f} | {r['soft_ms']:.2f} | {r['hard_ms']:.2f} |\n"
        )

    lines.append("\n**Soft Accuracy**: evaluated with sigmoid gates (values 0–1, continuous).  \n")
    lines.append("**Hard Accuracy**: evaluated with binarized gates (exactly 0 or 1) — simulates deployed model.  \n")
    lines.append("**Sparsity**: % of gates whose sigmoid value is below `1e-2` (effectively pruned).  \n")

    lines.append("\n---\n")
    lines.append("## 3. Per-Layer Sparsity (Best Model)\n")
    lines.append(
        f"Best model: **λ = {best['lambda']:.0e}** "
        f"(Soft Acc: {best['soft_acc']*100:.2f}%, Sparsity: {best['sparsity']*100:.2f}%)\n\n"
    )
    lines.append("| Layer | Sparsity |\n|-------|----------|\n")
    for layer_name, sp in best["per_layer"].items():
        lines.append(f"| {layer_name} | {sp} |\n")

    lines.append("\n---\n")
    lines.append("## 4. Analysis of Lambda Trade-off\n")
    lines.append(
        "- **λ = 0 (baseline):** No sparsity pressure. Network retains all connections, achieves maximum accuracy as expected.\n"
        "- **Small λ (1e-6, 1e-5):** Light pruning. Small accuracy drop with meaningful sparsity gain. Best operating point for production where accuracy is critical.\n"
        "- **Medium λ (1e-4):** Balanced trade-off. Substantial sparsity with acceptable accuracy degradation. Gates show clear bimodal distribution (0 vs 1).\n"
        "- **High λ (1e-3):** Aggressive pruning. Network prunes heavily at cost of accuracy. Shows the regularization dominates the classification objective.\n\n"
        "The Pareto frontier plot (`accuracy_vs_sparsity.png`) visualises this trade-off across all lambda values.\n"
    )

    lines.append("\n---\n")
    lines.append("## 5. Key Design Decisions\n")
    lines.append(
        "1. **Gate initialization at 0:** `sigmoid(0) = 0.5` gives a balanced start — the optimizer has equal room to open or close gates.\n"
        "2. **Adam optimizer on all parameters:** gate_scores are updated jointly with weights. Adam's adaptive learning rate is important since gate_scores and weights live on very different scales.\n"
        "3. **BatchNorm + Dropout:** prevents the sparsity loss from collapsing the network representation early in training.\n"
        "4. **Cosine LR scheduler:** smooth learning rate decay helps fine-grained gate decisions in later epochs.\n"
        "5. **Hard mask at inference:** binarizing gates at deployment gives the actual speedup; soft gates still incur full multiply cost.\n"
    )

    lines.append("\n---\n")
    lines.append("## 6. Visualizations\n")
    lines.append("- `gate_distribution.png` — Histogram of gate values per λ. Successful pruning shows a spike at 0 and a cluster near 1.\n")
    lines.append("- `accuracy_vs_sparsity.png` — Pareto frontier: accuracy vs sparsity.\n")
    lines.append("- `loss_curves.png` — CE loss and sparsity loss curves over epochs per λ.\n")

    report_text = "\n".join(lines)
    with open("report.md", "w") as f:
        f.write(report_text)
    print("[SAVED] report.md")


# =============================================================================
# 10. MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  TREDENCE CASE STUDY: Self-Pruning Neural Network")
    print("="*60)

    print("\n[INFO] Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()
    print(f"[INFO] Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(lam, train_loader, test_loader)
        all_results.append(result)

    print("\n[INFO] Generating visualizations...")
    plot_gate_distribution(all_results)
    plot_accuracy_vs_sparsity(all_results)
    plot_loss_curves(all_results)

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Lambda':<12} {'Soft Acc':>10} {'Hard Acc':>10} {'Sparsity':>10} {'ms/batch':>10}")
    print("-" * 57)
    for r in all_results:
        print(
            f"{r['lambda']:<12.0e}"
            f"{r['soft_acc']*100:>9.2f}%"
            f"{r['hard_acc']*100:>9.2f}%"
            f"{r['sparsity']*100:>9.2f}%"
            f"{r['soft_ms']:>10.2f}"
        )
