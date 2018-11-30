# Only install most recent pcmdpy version
cd pcmdpy && git pull origin master
make install && cd ..

mkdir logs/
mkdir results/
