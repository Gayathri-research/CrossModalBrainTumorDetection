set -e
for i in $(seq 1 960); do [ -f logs/FULL_DONE ] && break; sleep 30; done
[ -f logs/FULL_DONE ] || { echo "main run did not finish"; exit 1; }
# the main patient-wise run used the default seed 0, so it is seed 0 of the repeat
cp -r runs/patient/crossmodalnet runs/patient/crossmodalnet_s0
cp -r results/patient/crossmodalnet results/patient/crossmodalnet_s0
python -m crossmodalnet.cli.seeds --dataset full --split patient --seeds 0,1,2 --run > logs/seeds_patient.log 2>&1
echo FINISHED > logs/SEEDS_DONE
