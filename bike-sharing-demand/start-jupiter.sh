
#!/bin/zsh
# pwd=${PWD}
export mount="/Users/stefanocampanini/Library/CloudStorage/OneDrive-TUI/Learn-tech/Ai/Udacity/AWS Machine Learning Engineer Nanodegree/projects/kaggle/bike-sharing-demand"
# docker run -it --rm -p "8888:8888" -v "${mount}":/home/root jupyter/base-notebook --user root
#    -e CHOWN_HOME=yes \
#    -e CHOWN_HOME_OPTS="-R" \
docker run \
    -v "${mount}":/home/jovyan \
    -p 8888:8888 \
    --user root \
    quay.io/jupyter/base-notebook