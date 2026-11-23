for s in patient image; do
  python -m crossmodalnet.cli.explain --dataset full --split $s --seg_root Data/seg_all --slices_per_patient 10 --no_figures > logs/annot74_$s.log 2>&1
done
echo FINISHED > logs/ANNOT74_DONE
