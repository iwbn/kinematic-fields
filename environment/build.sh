

#curl -O https://download.pytorch.org/whl/cu116/torch-1.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl

cd ..
docker build -t sgvrcluster.kaist.ac.kr/woobin/hexplane-base:latest -f environment/Dockerfile .
cd environment