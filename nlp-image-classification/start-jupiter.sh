
#!/bin/zsh
# pwd=${PWD}
export mount=${PWD}

docker run \
    -v "${mount}":/home/jovyan \
    -e TMPDIR="/home/jovyan/tmp" \
    -p 8888:8888 \
    -p 6006:6006 \
    --user root \
    jupyter/tensorflow-notebook \
    start-notebook.py --IdentityProvider.token=''