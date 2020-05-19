#!/bin/bash
# text2bin.sh: Script for converting sparse network text files into binary files
# (c) Mohammad Hasanzadeh Mofrad, 2020
# (e) m.hasanzadeh.mofrad@gmail.com
# Run: chmod +x text2bin.sh && ./text2bin.sh

echo "Run sparse_dnn_generator.py to generate the text files representing the sparse network first"


if [ "$#" -ne 2 ] || ! [ -d "$1" ]; then
	echo "USAGE: $0 DIRECTORY NLAYERS"
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
FILE_TXT=${TXT_DIR}/input.txt
FILE_BIN=${BIN_DIR}/input.bin
./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 3

echo "Converting Sparse DNN files from text (${TXT_DIR}) to binary (${BIN_DIR})"
FILE_TXT=${TXT_DIR}/predictions.txt
FILE_BIN=${BIN_DIR}/predictions.bin
./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 2

NLAYERS=$2
for (( i=0; i<${NLAYERS}; i++ )); do
	FILE_TXT=${TXT_DIR}/weights${i}.txt
	FILE_BIN=${BIN_DIR}/weights${i}.bin
	./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 3
done

for (( i=0; i<${NLAYERS}; i++ )); do
	FILE_TXT=${TXT_DIR}/bias${i}.txt
	FILE_BIN=${BIN_DIR}/bias${i}.bin
	./${CONVERTER} ${FILE_TXT} ${FILE_BIN} 4
done

exit