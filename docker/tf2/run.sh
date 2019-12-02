#!/usr/bin/env bash

action() {
    # determine the directy of this file (/docker) and the repo dir
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local repo_dir="$( cd "$( dirname "$( dirname "$this_dir" )" )" && pwd )"

    # some configs
    [ -z "$DOCKER_ROOT" ] && local DOCKER_ROOT="0"
    [ -z "$DOCKER_PORT" ] && local DOCKER_PORT="8888"

    # user option for docker run, depends on whether to run as root or not
    local user_opt="-u $(id -u):$(id -g)"
    [ "$DOCKER_ROOT" = "1" ] && user_opt=""

    # run the container
    docker run \
        --rm \
        -ti \
        -w /tutorial \
        -v "$repo_dir":/tutorial \
        -e "NB_PORT=$DOCKER_PORT" \
        -p $DOCKER_PORT:$DOCKER_PORT \
        $user_opt \
        3pia/iml2019:tf2 $@
}
action "$@"
