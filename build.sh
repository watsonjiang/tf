if ! command -v python3 >/dev/null 2>&1; then
  echo python3 is not found && exit 1
fi

for f in dist tflearn.egg-info build
do
   if [ -d $f ];then
      echo "clean directory $f"
      rm -rf $f
   fi
done

python3 setup.py bdist_wheel

cp Dockerfile dist/
cd dist && sudo docker build -t tflearn:latest-gpu .
