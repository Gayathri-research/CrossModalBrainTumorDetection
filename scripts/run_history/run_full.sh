set -e
for s in patient image; do
  python -m crossmodalnet.cli.train --dataset full --split $s --name crossmodalnet > logs/train_$s.log 2>&1
  python -m crossmodalnet.cli.evaluate --dataset full --split $s --name crossmodalnet > logs/eval_$s.log 2>&1
done
python -m crossmodalnet.cli.explain --dataset full --split patient --n_per_class 20 > logs/explain_patient.log 2>&1
echo FINISHED > logs/FULL_DONE
