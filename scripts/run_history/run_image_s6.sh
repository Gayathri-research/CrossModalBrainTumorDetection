# Slice-wise: seed 1 did not converge (val macro-F1 0.22), so seed 6 replaces it (same rule as patient-wise:
# a seed passes if its best validation macro-F1 >= 0.60; test is evaluated only for a passing seed).
python -m crossmodalnet.cli.train --dataset full --split image --name crossmodalnet_s6 --seed 6 > logs/train_image_s6.log 2>&1 || true
best=$(python -c "import pandas as pd;print(pd.read_csv('runs/image/crossmodalnet_s6/history.csv').val_macro_f1.max())")
echo "image seed 6 best val macro-F1 $best" >> logs/seed_selection.log
if python -c "import sys;sys.exit(0 if float('$best')>=0.60 else 1)"; then
  python -m crossmodalnet.cli.seeds --dataset full --split image --seeds 0,2,6 --run > logs/seeds_image_s6.log 2>&1
fi
echo FINISHED > logs/IMAGE_S6_DONE
