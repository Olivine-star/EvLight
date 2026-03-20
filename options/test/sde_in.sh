export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/test/sde_in.yaml" \
  --log_dir="./log/test/sde_in/" \
  --alsologtostderr=True
