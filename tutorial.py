# coding: utf-8

"""
Helpful utilities for the "Physics-inspired feature engineering" turorial,
held at the IML workshop 2019.
"""

import os
import sys
import functools
import tarfile
import tempfile
import shutil

import numpy as np
import six


# print function with auto-flush
print_ = functools.partial(six.print_, flush=True)

# define directories and urls
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
eos_dir = "/eos/user/m/mrieger/public/iml2019"
eos_url_pattern = "https://cernbox.cern.ch/index.php/s/xDYiSmbleT3rip4/download?path={}&files={}"

# create the data dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# check if we have access to /eos or not
has_eos = os.access(eos_dir, os.R_OK)
print_("eos access: {}".format("✔︎" if has_eos else "✗"))

# if eos is accessible, amend sys.path to find shared software
# otherwise, software must be installed manually (or via requirements.txt on binder)
if has_eos:
    sys.path.insert(0, os.path.join(eos_dir, "software/lib/python2.7/site-packages"))


# eos url with arguments
def eos_url(*args):
    path = os.path.normpath("/" + "/".join(str(s) for s in args))
    path, files = os.path.split(path)
    quote = six.moves.urllib.parse.quote
    return eos_url_pattern.format(quote(path, safe=""), quote(files, safe=""))


# file download helper
def download(src, dst, bar=None):
    import wget
    return wget.download(src, out=dst, bar=bar)


# gets a file from eos, passed relative to eos_dir (see above)
# returns the full eos path when eos is available, otherwise downloads it via cernbox and returns
# the location of the downloaded file
def get_file(eos_file, is_dir=False, silent=False):
    eos_file = eos_file.lstrip("/")
    if has_eos:
        return os.path.join(eos_dir, eos_file)
    else:
        local_path = os.path.join(data_dir, eos_file)
        if not os.path.exists(local_path):
            if not silent:
                print_("downloading {} from CERNBox".format(eos_file))

            if is_dir:
                tmp_dir = tempfile.mkdtemp(dir=data_dir)
                arc_path = download(eos_url(eos_file), tmp_dir)
                with tarfile.open(arc_path, "r") as arc:
                    arc.extractall(data_dir)
                shutil.rmtree(tmp_dir)

            else:
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                download(eos_url(eos_file), local_path)

        return local_path


# data loading helper
def load_lbn_data(kind="train", sorting="gen", level="low"):
    """
    Loads an LBN dataset defined by *level* ("low", "high" or "mixed"), *sorting* ("gen" or "pt"),
    and *kind* ("train" or "valid"). The return value is dictionary-like object with two keys,
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

    kinds = ("train", "valid")
    if kind not in kinds:
        raise ValueError("unknown dataset kind '{}', must be one of {}".format(
            kind, ",".join(kinds)))

    # download the file from CERNBox when not eos is not accessible
    local_path = get_file("lbn/data/{}_{}_{}.npz".format(level, sorting, kind))

    # open and return the numpy file object
    return np.load(local_path)
