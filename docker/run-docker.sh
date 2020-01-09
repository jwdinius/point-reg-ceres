# NOTE: run this from dir above 'docker'!
docker run -it --rm \
    -v $(pwd):/home/ceresptr/point-reg-ceres \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --name pr-ceres-c \
    --net host \
    --runtime=nvidia \
    point-reg-ceres
