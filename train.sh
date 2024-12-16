export PYTHONPATH=$(pwd)

#bash run.sh train bemapnet_av2_res50 2 0 1
python3 configs/bemapnet_av2_res50_geo_splits_interval_4.py -d 0 -b 1 -e 24 --sync_bn 1 --no-clearml