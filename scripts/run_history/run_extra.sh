set -e
for i in $(seq 1 1500); do [ -f logs/SEEDS_IMAGE_DONE ] && break; sleep 30; done
[ -f logs/SEEDS_IMAGE_DONE ] || { echo "seed runs did not finish"; exit 1; }
python -m crossmodalnet.cli.explain --dataset full --split image --n_per_class 20 > logs/explain_image.log 2>&1
for s in patient image; do
  python -m crossmodalnet.cli.explain --dataset full --split $s --seg_root Data/subset --slices_per_patient 10 > logs/annot_$s.log 2>&1
done
echo FINISHED > logs/EXTRA_DONE
