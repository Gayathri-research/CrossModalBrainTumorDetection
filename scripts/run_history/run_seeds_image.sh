set -e
for i in $(seq 1 1200); do [ -f logs/SEEDS_DONE ] && break; sleep 30; done
[ -f logs/SEEDS_DONE ] || { echo "patient seeds did not finish"; exit 1; }
cp -r runs/image/crossmodalnet runs/image/crossmodalnet_s0
cp -r results/image/crossmodalnet results/image/crossmodalnet_s0
python -m crossmodalnet.cli.seeds --dataset full --split image --seeds 0,1,2 --run > logs/seeds_image.log 2>&1
echo FINISHED > logs/SEEDS_IMAGE_DONE
