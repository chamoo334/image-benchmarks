#!/bin/bash
# TODO: verify requirements installed,
# Ptyhon 3.8, CUDA 11.2

export CUDA_HOME=/usr/lib/cuda

# set variables based on flag arguments
while getopts "i:r:p:" flag
do
     case $flag in
         i) imageName=$OPTARG;;
         r) resultsFile=$OPTARG;;
         p) pythonCom=$OPTARG;;
     esac
done

# set default variables
if [ -z "$imageName" ]
then
    imageName="images/houses.bmp"
fi

if [ -z "$pythonCom" ]
then
    pythonCom="python3"
fi

if [ -z "$resultsFile" ]
then
    resultsFile="results.txt"
fi

echo "image file: $imageName, python: $pythonCom, results file: $resultsFile"

threadCount=(16 32)
blockCount=(4 8 16 32 64)
treadArrLength=${#threadCount[@]}
blockArrLength=${#blockCount[@]}

declare -A results
resPos=1

echo "Running C Sequential"
nvcc cuda_c/line_detection.cu
results[$resPos,1]=0
results[$resPos,2]=0
results[$resPos,3]=$(./a.out $imageName seq >&1)
results[$resPos,4]=0
results[$resPos,5]=NA

echo "Running C CUDA"
for t in ${threadCount[@]}; do
  nvcc cuda_c/line_detection.cu -DNUM_THREADS=$t
  ((resPos+=1))
  tempPos=1
  
  IFS=';' read -ra TEMP <<< "$(./a.out $imageName par >&1)"
  for i in "${TEMP[@]}"; do
    results[$resPos,$tempPos]=$i
    ((tempPos+=1))
  done
done

echo "Running Python Sequential"
results[$resPos,1]=0
results[$resPos,2]=0
results[$resPos,3]=$($pythonCom py_cuda/line_detection.py $imageName seq >&1)
results[$resPos,4]=0
results[$resPos,5]=NA

echo "Running Python CUDA (Numba)"
for t in ${threadCount[@]}; do
  ((resPos+=1))
  tempPos=1
  
  IFS=';' read -ra TEMP <<< "$($pythonCom py_cuda/line_detection.py $imageName par $t >&1)"
  for i in "${TEMP[@]}"; do
    results[$resPos,$tempPos]=$i
    ((tempPos+=1))
  done
done

# blocks, threads, PixelsPerMSTotalTime, PixelsPerMSTotalTimeGPUTime, PixelsPerMSTimeDif
for ((j=1;j<=resPos;j++)) do
    printf "${results[$j,1]}, ${results[$j,2]}, ${results[$j,3]}, ${results[$j,4]}, ${results[$j,5]}"
    echo
done
