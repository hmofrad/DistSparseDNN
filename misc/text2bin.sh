#!/bin/bash
# (c) Mohammad Hasanzadeh Mofrad, 2019
# (e) m.hasanzadeh.mofrad@gmail.com
# Run: chmod +x text2bin.sh && ./text2bin.sh

echo "Sparse Deep Neural Network Challange Dataset (http://graphchallenge.mit.edu/)"
echo "Script for converting dataset text files into binary files"
echo "Approximate required space is 100 GB"

DATA_DIR=../data1
TXT_DIR=text
BIN_DIR=bin

mkdir -p ${DATA_DIR}

#NEURONS=(1024 4096 16384 65536)
NEURONS=(1024)
TXT_DIR_MNIST=${DATA_DIR}/${TXT_DIR}/MNIST
mkdir -p ${TXT_DIR_MNIST}
MNIST_FILE_PREFIX="sparse-images"
MNIST_URLS_PREFIX="https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images"

echo "Downloading MNIST files to ${TXT_DIR_MNIST}"
for N in "${NEURONS[@]}"; do
	FILE_URL=${MNIST_URLS_PREFIX}-${N}.tsv.gz
	FILE_GZ=${TXT_DIR_MNIST}/${MNIST_FILE_PREFIX}-${N}.tsv.gz
	FILE_TSV=${TXT_DIR_MNIST}/${MNIST_FILE_PREFIX}-${N}.tsv
	if [ ! -f "${FILE_TSV}" ]; then
		wget ${FILE_URL} -P ${TXT_DIR_MNIST}
		gunzip ${FILE_GZ}
	fi
done




TXT_DIR_DNN=${DATA_DIR}/${TXT_DIR}/DNN
mkdir -p ${TXT_DIR_DNN}
DNN_FILE_PREFIX="neuron"
DNN_URLS_PREFIX="https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron"

echo "Downloading   DNN files to ${TXT_DIR_DNN}"
for N in "${NEURONS[@]}"; do
	FILE_URL=${DNN_URLS_PREFIX}${N}.tar.gz
	FILE_TAR_GZ=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}.tar.gz
	FILE_TAR=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}.tar
	FILE_TSV_PATH=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}
	if [ ! -d "${FILE_TSV_PATH}" ]; then
		wget ${FILE_URL} -P ${TXT_DIR_DNN}
		tar -xf  ${FILE_TAR_GZ} -C ${TXT_DIR_DNN}
		rm ${FILE_TAR_GZ}
	fi
done
LAYERS=(120 480 1920)
for N in "${NEURONS[@]}"; do
	for L in "${LAYERS[@]}"; do
		FILE_URL=${DNN_URLS_PREFIX}${N}-l${L}-categories.tsv
		FILE_TSV=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}-l${L}-categories.tsv
		if [ ! -f "${FILE_TSV}" ]; then
			wget ${FILE_URL} -P ${TXT_DIR_DNN}
		fi
	done
done

CONVERTER=text2bin
if [ ! -f "${CONVERTER}" ]; then
	g++ -o ${CONVERTER} ${CONVERTER}.cpp -std=c++14 -DNDEBUG -O3 -flto -fwhole-program -march=native
fi

BIN_DIR_MNIST=${DATA_DIR}/${BIN_DIR}/MNIST
mkdir -p ${BIN_DIR_MNIST}
echo "Converting MINST files from text (${TXT_DIR_MNIST}) to binary (${BIN_DIR_MNIST})"

for N in "${NEURONS[@]}"; do
	FILE_TSV=${TXT_DIR_MNIST}/${MNIST_FILE_PREFIX}-${N}.tsv
	FILE_BIN=${BIN_DIR_MNIST}/${MNIST_FILE_PREFIX}-${N}.bin
	if [ ! -f "${FILE_BIN}" ]; then
		./${CONVERTER} ${FILE_TSV} ${FILE_BIN} 3
	fi
done

BIN_DIR_DNN=${DATA_DIR}/${BIN_DIR}/DNN
mkdir -p ${BIN_DIR_DNN}
echo "Converting   DNN files from text (${TXT_DIR_DNN})   to binary (${BIN_DIR_DNN})"

for N in "${NEURONS[@]}"; do
	for L in "${LAYERS[@]}"; do
		FILE_TSV=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}-l${L}-categories.tsv
		FILE_BIN=${BIN_DIR_DNN}/${DNN_FILE_PREFIX}${N}-l${L}-categories.bin
		if [ ! -f "${FILE_BIN}" ]; then
			./${CONVERTER} ${FILE_TSV} ${FILE_BIN} 1
		fi
	done
done

for N in "${NEURONS[@]}"; do
	FILE_TSV_PATH=${TXT_DIR_DNN}/${DNN_FILE_PREFIX}${N}
	FILE_BIN_PATH=${BIN_DIR_DNN}/${DNN_FILE_PREFIX}${N}
	if [ ! -d "${FILE_BIN_PATH}" ]; then
		mkdir ${FILE_BIN_PATH}
		for i in {1..1920..1}; do
			FILE_TSV=${FILE_TSV_PATH}/n${N}-l${i}.tsv
			FILE_BIN=${FILE_BIN_PATH}/n${N}-l${i}.bin
			./${CONVERTER} ${FILE_TSV} ${FILE_BIN} 3
		done
	fi
done

exit;
