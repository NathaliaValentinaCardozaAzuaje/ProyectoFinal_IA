"""
upgrade_dnn_architecture.py
===========================
Modifica el notebook persona_C_regression_app_bot.ipynb para:
  1. Deepening de la DNN: 512->256->128->64->32->16->1
  2. WeightedMSE fraud_weight=15 (up from 10)
  3. EarlyStopping patience=20 (up from 15), delta=1e-7
  4. N_EPOCHS=200, lr=5e-4, ReduceLROnPlateau patience=8
  5. Inicializacion He (kaiming_normal_) para mejor convergencia

Ejecutar desde la raiz del proyecto:
    python scripts/upgrade_dnn_architecture.py
"""

import json, re
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "persona_C_regression_app_bot.ipynb"

# ──────────────────────────────────────────────────────────────────────────────
# Celdas de reemplazo
# ──────────────────────────────────────────────────────────────────────────────

NEW_ARCH_SOURCE = [
    "class FraudDNN(nn.Module):\n",
    "    \"\"\"\n",
    "    Red Neuronal Profunda v3 para prediccion de risk_score.\n",
    "\n",
    "    Arquitectura profundizada para superar al Gradient Boosting:\n",
    "    - 512->256->128->64->32->16->1 (mayor capacidad expresiva)\n",
    "    - Dropout DECRECIENTE (0.35->0.05): mas regularizacion en capas tempranas\n",
    "    - Sigmoid en salida: garantiza risk_score en [0, 1]\n",
    "    - BatchNorm en todos los bloques: entrenamiento mas estable\n",
    "    - Inicializacion He (kaiming_normal_) para mejor convergencia con ReLU\n",
    "    \"\"\"\n",
    "    def __init__(self, input_dim: int):\n",
    "        super(FraudDNN, self).__init__()\n",
    "        self.network = nn.Sequential(\n",
    "            # Bloque 1: maxima capacidad, mayor regularizacion\n",
    "            nn.Linear(input_dim, 512),\n",
    "            nn.BatchNorm1d(512),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.35),\n",
    "\n",
    "            # Bloque 2\n",
    "            nn.Linear(512, 256),\n",
    "            nn.BatchNorm1d(256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.30),\n",
    "\n",
    "            # Bloque 3\n",
    "            nn.Linear(256, 128),\n",
    "            nn.BatchNorm1d(128),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.25),\n",
    "\n",
    "            # Bloque 4\n",
    "            nn.Linear(128, 64),\n",
    "            nn.BatchNorm1d(64),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.15),\n",
    "\n",
    "            # Bloque 5\n",
    "            nn.Linear(64, 32),\n",
    "            nn.BatchNorm1d(32),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.10),\n",
    "\n",
    "            # Bloque 6: refinamiento fino, minima regularizacion\n",
    "            nn.Linear(32, 16),\n",
    "            nn.BatchNorm1d(16),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.05),\n",
    "\n",
    "            # Salida: Sigmoid garantiza [0, 1]\n",
    "            nn.Linear(16, 1),\n",
    "            nn.Sigmoid(),\n",
    "        )\n",
    "        # Inicializacion He para capas lineales (mejor para ReLU)\n",
    "        self._init_weights()\n",
    "\n",
    "    def _init_weights(self):\n",
    "        for m in self.modules():\n",
    "            if isinstance(m, nn.Linear):\n",
    "                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')\n",
    "                if m.bias is not None:\n",
    "                    nn.init.zeros_(m.bias)\n",
    "\n",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
    "        return self.network(x).squeeze(1)\n",
    "\n",
    "\n",
    "dnn = FraudDNN(input_dim=len(FEATURES)).to(DEVICE)\n",
    "print('Arquitectura FraudDNN v3 (profundizada para superar GB):')\n",
    "print(dnn)\n",
    "total_params = sum(p.numel() for p in dnn.parameters() if p.requires_grad)\n",
    "print(f'\\nParametros entrenables: {total_params:,}')",
]

NEW_LOSS_SOURCE = [
    "class WeightedMSELoss(nn.Module):\n",
    "    \"\"\"\n",
    "    MSE ponderada por clase para manejar el desbalance severo del dataset.\n",
    "\n",
    "    Las transacciones con risk_score alto (fraudes) reciben peso fraud_weight\n",
    "    veces mayor que las legitimas.\n",
    "\n",
    "    Formalmente:\n",
    "        loss = mean( weights * (pred - target)^2 )\n",
    "        donde weights = 1.0 + (fraud_weight - 1.0) * target\n",
    "\n",
    "    fraud_weight=15.0 (aumentado desde 10.0) para mejor captura de fraudes\n",
    "    con la arquitectura mas profunda.\n",
    "    \"\"\"\n",
    "    def __init__(self, fraud_weight: float = 15.0):\n",
    "        super().__init__()\n",
    "        self.fraud_weight = fraud_weight\n",
    "\n",
    "    def forward(self, predictions: torch.Tensor,\n",
    "                targets: torch.Tensor) -> torch.Tensor:\n",
    "        weights = 1.0 + (self.fraud_weight - 1.0) * targets\n",
    "        return (weights * (predictions - targets) ** 2).mean()\n",
    "\n",
    "\n",
    "criterion = WeightedMSELoss(fraud_weight=15.0)\n",
    "print(f'WeightedMSELoss configurada con fraud_weight=15.0 (aumentado para DNN profunda)')\n",
    "print('Transacciones fraudulentas tienen 15x mas peso en la funcion de perdida.')",
]

NEW_OPTIM_SOURCE = [
    "# Optimizer con weight decay (L2 regularization adicional)\n",
    "# lr=5e-4 reducido para red mas profunda (512 neuronas primera capa)\n",
    "optimizer = optim.AdamW(dnn.parameters(), lr=5e-4, weight_decay=5e-5)\n",
    "\n",
    "# Scheduler: reduce LR x0.5 cuando val_loss no mejora por 8 epocas\n",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n",
    "    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7, verbose=False\n",
    ")\n",
    "\n",
    "print('Optimizer : AdamW (lr=5e-4, weight_decay=5e-5)')\n",
    "print('Scheduler : ReduceLROnPlateau (factor=0.5, patience=8)')\n",
    "print('Criterio  : WeightedMSELoss (fraud_weight=15)')",
]

NEW_TRAIN_SOURCE = [
    "# DNN profunda: mas epocas y paciencia para explorar mejor el espacio\n",
    "N_EPOCHS    = 200\n",
    "early_stop  = EarlyStopping(patience=20, delta=1e-7)\n",
    "train_losses, val_losses = [], []\n",
    "\n",
    "print('Iniciando entrenamiento DNN v3 (arquitectura profunda)...')\n",
    "print(f'{\"Epoca\":<6} {\"Train Loss\":<12} {\"Val Loss\":<12} {\"LR\":<10}')\n",
    "print('-' * 48)\n",
    "\n",
    "t0_dnn = time.time()\n",
    "\n",
    "for epoch in range(1, N_EPOCHS + 1):\n",
    "    tr_loss  = run_epoch(dnn, train_loader, criterion, optimizer, DEVICE, train=True)\n",
    "    val_loss = run_epoch(dnn, val_loader,   criterion, optimizer, DEVICE, train=False)\n",
    "    \n",
    "    scheduler.step(val_loss)\n",
    "    train_losses.append(tr_loss)\n",
    "    val_losses.append(val_loss)\n",
    "    early_stop(val_loss, dnn, epoch)\n",
    "\n",
    "    if epoch % 10 == 0 or epoch == 1:\n",
    "        lr_now = optimizer.param_groups[0]['lr']\n",
    "        print(f'{epoch:<6}/{N_EPOCHS} {tr_loss:<12.6f} {val_loss:<12.6f} {lr_now:<10.2e}')\n",
    "\n",
    "    if early_stop.stop:\n",
    "        print(f'\\nEarly Stopping en epoca {epoch}. Mejor val_loss={early_stop.best_loss:.6f} (epoca {early_stop.best_epoch})')\n",
    "        break\n",
    "\n",
    "t1_dnn = time.time()\n",
    "tiempo_dnn = t1_dnn - t0_dnn\n",
    "\n",
    "# Restaurar los mejores pesos\n",
    "dnn.load_state_dict(early_stop.best_weights)\n",
    "print(f'\\nEntrenamiento completado en {tiempo_dnn:.1f}s')\n",
    "print(f'Mejor epoca       : {early_stop.best_epoch}')\n",
    "print(f'Mejor val_loss    : {early_stop.best_loss:.6f}')\n",
    "print(f'Epocas efectivas  : {len(train_losses)}')",
]

# ──────────────────────────────────────────────────────────────────────────────
# Funciones de deteccion de celdas
# ──────────────────────────────────────────────────────────────────────────────

def source_contains(cell, *snippets):
    """True si TODOS los snippets aparecen en el source de la celda."""
    src = "".join(cell.get("source", []))
    return all(s in src for s in snippets)


def replace_cell_source(cells, detect_snippets, new_source, label):
    """Reemplaza el source de la primera celda que contenga detect_snippets."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and source_contains(cell, *detect_snippets):
            cells[i]["source"] = new_source
            cells[i]["outputs"] = []          # limpiar outputs anteriores
            cells[i]["execution_count"] = None
            print(f"  [OK] Celda '{label}' actualizada (indice {i})")
            return True
    print(f"  [!!] Celda '{label}' NO encontrada -- revisa los snippets de deteccion")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Notebook: {NOTEBOOK_PATH}")
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"No se encontró el notebook en {NOTEBOOK_PATH}")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"Total celdas: {len(cells)}\n")

    # 1. Arquitectura DNN
    replace_cell_source(
        cells,
        detect_snippets=["class FraudDNN", "nn.Linear(input_dim, 256)"],
        new_source=NEW_ARCH_SOURCE,
        label="Arquitectura DNN",
    )

    # 2. WeightedMSELoss
    replace_cell_source(
        cells,
        detect_snippets=["class WeightedMSELoss", "fraud_weight=10.0"],
        new_source=NEW_LOSS_SOURCE,
        label="WeightedMSELoss",
    )

    # 3. Optimizer / Scheduler
    replace_cell_source(
        cells,
        detect_snippets=["optim.AdamW", "ReduceLROnPlateau"],
        new_source=NEW_OPTIM_SOURCE,
        label="Optimizer + Scheduler",
    )

    # 4. Loop de entrenamiento / EarlyStopping
    replace_cell_source(
        cells,
        detect_snippets=["N_EPOCHS    = 150", "early_stop  = EarlyStopping"],
        new_source=NEW_TRAIN_SOURCE,
        label="Loop entrenamiento",
    )

    # Guardar
    NOTEBOOK_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\n[OK] Notebook guardado en {NOTEBOOK_PATH}")
    print("\nPróximos pasos:")
    print("  1. Abre el notebook en Jupyter")
    print("  2. Kernel > Restart & Run All")
    print("  3. Verifica que R²_DNN > R²_GradientBoosting (0.9656) en la tabla comparativa")


if __name__ == "__main__":
    main()
