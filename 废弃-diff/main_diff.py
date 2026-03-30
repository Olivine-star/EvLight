"""
Entry point for EvWDiff training and evaluation.
Separate from the original main.py to avoid modifying existing code.

Usage:
    # Training
    python main_diff.py --yaml_file=options/train/sde_in_diff.yaml \
                        --log_dir=./log/evwdiff/sde_in

    # Testing
    python main_diff.py --yaml_file=options/train/sde_in_diff.yaml \
                        --log_dir=./log/evwdiff/sde_in_test \
                        --TEST_ONLY \
                        --RESUME_PATH=./log/evwdiff/sde_in/model_best.pth.tar \
                        --VISUALIZE
"""
import json
import os
import sys

import yaml
from absl import app, flags, logging
from absl.logging import info
from easydict import EasyDict

from egllie.core.launch_diff import DiffusionLaunch

FLAGS = flags.FLAGS

# Only register flags that haven't been registered by the original main.py
# This allows this script to be imported alongside the original
_existing_flags = set(FLAGS)

if 'yaml_file' not in _existing_flags:
    flags.DEFINE_string("yaml_file", None, "The config file.")
if 'RESUME_PATH' not in _existing_flags:
    flags.DEFINE_string("RESUME_PATH", None, "The RESUME.PATH.")
if 'RESUME_TYPE' not in _existing_flags:
    flags.DEFINE_string("RESUME_TYPE", None, "The RESUME.TYPE.")
if 'RESUME_SET_EPOCH' not in _existing_flags:
    flags.DEFINE_boolean("RESUME_SET_EPOCH", False, "The RESUME.SET_EPOCH.")
if 'TEST_ONLY' not in _existing_flags:
    flags.DEFINE_boolean("TEST_ONLY", False, "Test only mode.")
if 'VISUALIZE' not in _existing_flags:
    flags.DEFINE_boolean("VISUALIZE", False, "Visualization switch.")
if 'TRAIN_BATCH_SIZE' not in _existing_flags:
    flags.DEFINE_integer("TRAIN_BATCH_SIZE", None, "Override train batch size.")
if 'VAL_BATCH_SIZE' not in _existing_flags:
    flags.DEFINE_integer("VAL_BATCH_SIZE", None, "Override val batch size.")
if 'PUDB' not in _existing_flags:
    flags.DEFINE_boolean("PUDB", False, "Debug switch.")


def init_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()

    config["SAVE_DIR"] = FLAGS.log_dir

    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
        config["RESUME"]["SET_EPOCH"] = FLAGS.RESUME_SET_EPOCH
    if FLAGS.VISUALIZE:
        config["VISUALIZE"] = FLAGS.VISUALIZE
    if FLAGS.TRAIN_BATCH_SIZE:
        config["TRAIN_BATCH_SIZE"] = FLAGS.TRAIN_BATCH_SIZE
    if FLAGS.VAL_BATCH_SIZE:
        config["VAL_BATCH_SIZE"] = FLAGS.VAL_BATCH_SIZE
    if FLAGS.TEST_ONLY:
        config["TEST_ONLY"] = FLAGS.TEST_ONLY

    info(f"Launch Config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)


def main(args):
    if FLAGS.PUDB:
        from pudb import set_trace
        set_trace()

    config = init_config(FLAGS.yaml_file)
    launcher = DiffusionLaunch(config)
    launcher.run()


if __name__ == "__main__":
    app.run(main)
