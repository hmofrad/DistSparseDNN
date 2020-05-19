#!/bin/bash
# sparse_mnist.sh: Script for converting sparse mnist network text files into binary
# (c) Mohammad Hasanzadeh Mofrad, 2020
# (e) m.hasanzadeh.mofrad@gmail.com
# Run: chmod +x sparse_mnist.sh && ./sparse_mnist.sh

echo "Run sparse_mnist.py to generate the text files representing the sparse network first"


if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
	echo "Usage: $0 DIRECTORY"
	exit 1
fi

DATA_DIR=$1
TXT_DIR=text
BIN_DIR=bin


TXT_DIR=${DATA_DIR}/${TXT_DIR}
BIN_DIR=${DATA_DIR}/${BIN_DIR}
mkdir -p ${BIN_DIR}

CONVERTER=text2bin
if [ ! -f "${CONVERTER}" ]; then
	g++ -o ${CONVERTER} ${CONVERTER}.cpp -std=c++14 -DNDEBUG -O3 -flto -fwhole-program -march=native
fi

echo "Converting MINST input from text (${TXT_DIR}) to binary (${BIN_DIR})"
NEURONS=1024
FILE_TXT=${TXT_DIR}/input.txt
FILE_BIN=${BIN_DIR}/input.bin
if [ ! -f "${FILE_BIN}" ]; then
	./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 3
fi

echo "Converting Sparse DNN files from text (${TXT_DIR}) to binary (${BIN_DIR})"

FILE_TXT=${TXT_DIR}/predictions.txt
FILE_BIN=${BIN_DIR}/predictions.bin
if [ ! -f "${FILE_BIN}" ]; then
	./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 2
fi

LAYERS=30
for (( i=0; i<$LAYERS; i++ )); do
	FILE_TXT=${TXT_DIR}/weights${i}.txt
	FILE_BIN=${BIN_DIR}/weights${i}.bin
	if [ ! -f "${FILE_BIN}" ]; then
		./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 3
	fi
done

for (( i=0; i<$LAYERS; i++ )); do
	FILE_TXT=${TXT_DIR}/bias${i}.txt
	FILE_BIN=${BIN_DIR}/bias${i}.bin
	if [ ! -f "${FILE_BIN}" ]; then
		./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 4
	fi
done

exit;