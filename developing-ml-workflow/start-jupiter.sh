
#!/bin/zsh
# pwd=${PWD}
# TODO find a way to have mount path relative
export mount="/Users/stefanocampanini/Library/CloudStorage/OneDrive-TUI/Learn-tech/Ai/Udacity/AWS Machine Learning Engineer Nanodegree/projects/kaggle/udacity-aws-ml-eng-nanodegree/developing-ml-workflow"

docker run \
    -v "${mount}":/home/jovyan \
    -p 8888:8888 \
    --user root \
    -e GRANT_SUDO=yes \
    quay.io/jupyter/base-notebook \
    start-notebook.py --IdentityProvider.token=''
    