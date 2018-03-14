# Only install most recent pcmdpy version
cd pcmdpy && git pull origin master
echo "hello"
make pcmdpy_only && cd ..
