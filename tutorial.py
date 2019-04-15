# coding: utf-8

"""
Helpful utilities for the "Physics-inspired feature engineering" turorial,
held at the IML workshop 2019.
"""

import os
import sys
import functools

import numpy as np
import six


# print function with auto-flush
print_ = functools.partial(six.print_, flush=True)

# define directories and urls
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
eos_dir = "/eos/user/m/mrieger/public/iml2019/lbn"
eos_url = "https://cernbox.cern.ch/index.php/s/xDYiSmbleT3rip4/download?path=%2Flbn%2Fdata&files={}"

# check if we have access to /eos or not
has_eos = os.access(eos_dir, os.R_OK)
print_("eos access: {}".format("✔︎" if has_eos else "✗"), flush=True)

# if eos is accessible, amend sys.path to find shared software
# otherwise, software must be installed manually (or via requirements.txt on binder)
if has_eos:
    sys.path.insert(0, os.path.join(eos_dir, "software/lib/python2.7/site-packages"))


# data loading helper
def load_data(level="low", sorting="gen", kind="train"):
    """
    Loads an LBN dataset defined by *level* ("low", "high" or "mixed"), *sorting* ("gen" or "pt"),
    and *kind* ("train" or "test"). The return value is dictionary-like object with two keys,
    "labels" and "features", which point to plain numpy arrays.
    """
    levels = ("low", "high", "mixed")
    if level not in levels:
        raise ValueError("unknown dataset level '{}', must be one of {}".format(
            level, ",".join(levels)))

    sortings = ("gen", "pt")
    if sorting not in sortings:
        raise ValueError("unknown dataset sorting '{}', must be one of {}".format(
            sorting, ",".join(sortings)))

    kinds = ("train", "test")
    if kind not in kinds:
        raise ValueError("unknown dataset kind '{}', must be one of {}".format(
            kind, ",".join(kinds)))

    # download the file from CERNBox when not eos is not accessible
    file_name = "{}_{}_{}.npz".format(level, sorting, kind)
    if not has_eos:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print_("downloading {} from CERNBox".format(file_name), flush=True)
            download(eos_url.format(file_name), file_path)
    else:
        file_path = os.path.join(eos_dir, "data", file_name)

    # open and return the numpy file object
    return np.load(file_path)


# file download helper
def download(src, dst, bar=None):
    import wget
    return wget.download(src, out=dst, bar=bar)
