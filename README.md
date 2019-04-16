# IML 2019 Tutorials

The tutorials run in environments with **either** TensorFlow *v1* or *v2*:

- **Introduction tutorial**: TensorFlow *v1*
- **Feature engineering tutorial**: TensorFlow *v2*
- **GAN tutorial**: TensorFlow *v1*

Be aware of slight differences in their setup as explained in the following.

In general, there are four ways to start the notebooks.


### 1. SWAN

Click on the following link:

[![SWAN](http://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://cern.ch/swanserver/cgi-bin/go?projurl=https://github.com/3pia/iml2019.git)

You will be asked to configure the environment in a small dialog.

- For the introduction tutorial, enter the following environment script path and then press the "Start my Session" button at the bottom:

```
/eos/user/m/mrieger/public/iml2019/intro/setup.sh
```

- For the feature engineering tutorial, do the same with the following environment script path:

```
/eos/user/m/mrieger/public/iml2019/lbn/setup.sh
```

- For the GAN tutorial, just press the "Start my Session" button.


### 2. Binder

For TensorFlow *v1*:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/3pia/iml2019/master)

For TensorFlow *v2*:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/3pia/iml2019/tf2)


### 3. Standalone docker image from the docker hub

Make sure **not** to execute the following commands with `sudo` as a port will be opened on your machine to run and host the notebook server. Otherwise, you potentially allow people within you local network to access your system with root permissions!

If you don't have the permission to execute docker with your user account, add yourself to the "docker" group (e.g. via `sudo usermod -a -G docker $(whoami)`).

```shell
# for TensorFlow v1 (introduction and GAN tutorials)
docker run -ti -u $(id -u):$(id -g) -p 8888:8888 3pia/iml2019:tf1

# for TensorFlow v2 (feature engineering tutorial)
docker run -ti -u $(id -u):$(id -g) -p 8888:8888 3pia/iml2019:tf2
```


### 4. Docker image with a local checkout

You can also check out the repository and use the script located in `docker/tf{1,2}/run.sh` to start the docker container. The script will run the same command as above **and** mounts the repository into the container. This way, changes you make in the notebooks are persistently stored within you local checkout.

As above, make sure not to run the container as root!

```shell
git clone https://github.com/3pia/iml2019
cd iml2019

# for TensorFlow v1 (introduction and GAN tutorials)
./docker/tf1/run.sh

# for TensorFlow v2 (feature engineering tutorial)
./docker/tf2/run.sh
```
