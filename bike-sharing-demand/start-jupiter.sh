
#!/bin/zsh
# pwd=${PWD}
# TODO find a way to have mount path relative
export mount="/Users/stefanocampanini/Library/CloudStorage/OneDrive-TUI/Learn-tech/Ai/Udacity/AWS Machine Learning Engineer Nanodegree/projects/kaggle/udacity-aws-ml-eng-nanodegree/bike-sharing-demand"

docker run \
    -v "${mount}":/home/jovyan \
    -p 8888:8888 \
    --user root \
    quay.io/jupyter/base-notebook