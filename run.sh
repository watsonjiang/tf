sudo docker run --mount type=bind,source="$PWD"/data,target=/root/data  tflearn:latest-gpu python -m tflearn.adult_wd
