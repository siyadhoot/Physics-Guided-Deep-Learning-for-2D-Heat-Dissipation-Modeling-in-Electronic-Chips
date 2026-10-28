# Boundary weight pilot configuration

All arms use seed 42, training rows 0 through 255, the complete 800-row validation split, batch size 16, exactly 10 epochs, AdamW learning rate `1e-3`, weight decay `1e-4`, and the existing architecture and preprocessing. Early stopping is disabled so every arm receives the same number of epochs. The selected checkpoint is the epoch with the lowest full-validation `val_data_loss`, the trainer's existing default criterion.

| Arm | `w_data` | `w_pde` | `w_bc` |
|---|---:|---:|---:|
| A | 1.0 | 0 | 0 |
| B | 1.0 | 6e-11 | 6e-11 |
| C | 1.0 | 6e-11 | 1.5e-4 |

The evaluator reads only the supplied checkpoint and `dataset/validation.h5`; it does not compute integrity hashes. Outputs must be new arm-specific directories. Its temperature metrics use the union of the first/last rows and first/last columns (124 boundary pixels) and the complement (900 interior pixels), pooled across all validation samples. Boundary physical residual means pool all 800 x 124 perimeter residuals; signed and absolute aggregates are reported in W/m^3. Training boundary loss remains `mean((R_boundary / 1e6)**2)`.

Later GPU commands, from the repository root:

```powershell
python experiments/physics_guided/training/train_pinn.py --n-train-samples 256 --batch-size 16 --epochs 10 --disable-early-stopping --seed 42 --lr 1e-3 --weight-decay 1e-4 --w-data 1 --w-pde 0 --w-bc 0 --ckpt-dir experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_a --results-dir outputs/boundary_weight_pilot/arm_a --exp-name arm_a
python experiments/physics_guided/training/train_pinn.py --n-train-samples 256 --batch-size 16 --epochs 10 --disable-early-stopping --seed 42 --lr 1e-3 --weight-decay 1e-4 --w-data 1 --w-pde 6e-11 --w-bc 6e-11 --ckpt-dir experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_b --results-dir outputs/boundary_weight_pilot/arm_b --exp-name arm_b
python experiments/physics_guided/training/train_pinn.py --n-train-samples 256 --batch-size 16 --epochs 10 --disable-early-stopping --seed 42 --lr 1e-3 --weight-decay 1e-4 --w-data 1 --w-pde 6e-11 --w-bc 1.5e-4 --ckpt-dir experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_c --results-dir outputs/boundary_weight_pilot/arm_c --exp-name arm_c
python experiments/physics_guided/evaluate_boundary_pilot.py --checkpoint experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_a/arm_a_best.pt --output-dir outputs/boundary_weight_pilot/arm_a/evaluation --device cuda
python experiments/physics_guided/evaluate_boundary_pilot.py --checkpoint experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_b/arm_b_best.pt --output-dir outputs/boundary_weight_pilot/arm_b/evaluation --device cuda
python experiments/physics_guided/evaluate_boundary_pilot.py --checkpoint experiments/physics_guided/checkpoints/boundary_weight_pilot/arm_c/arm_c_best.pt --output-dir outputs/boundary_weight_pilot/arm_c/evaluation --device cuda
```
