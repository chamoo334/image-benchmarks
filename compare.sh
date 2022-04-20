#!/bin/bash
# ./compare.sh -i images/houses.bmp -p python3
# source py_img/bin/activate
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

# verify appropriate arguments
# echo file: $imageName
# echo results: $resultsFile
# echo pythonCom: $pythonCom

results=()

nvcc cuda_c/line_detection.cu
start=$SECONDS
pixelCountCS=$(./a.out $imageName seq >&1)
durationCS=$(printf %.10f $(( (10**10 * SECONDS - start) * 1000 ))e-10 >&1)
results+=([$durationCS, $pixelCountCS])

# start=$SECONDS
# pixelCountCP=$(./a.out $imageName par >&1)
# durationCP=$(printf %.10f $(( (10**10 * SECONDS - start) * 1000 ))e-10 >&1)
# results+=([$durationCP, $pixelCountCP])

start=$SECONDS
pixelCountPS=$($pythonCom py_cuda/line_detection.py $imageName seq >&1)
# durationPS=$(printf %.10f $(( 10**10 * SECONDS - start ))e-10 >&1)
durationPS=$(printf %.10f $(( (10**10 * SECONDS - start) * 1000 ))e-10 >&1)
results+=([$durationPS, $pixelCountPS])

# start=$SECONDS
# pixelCountPP=$($pythonCom line_detection.py $imageName par >&1)
# durationPP=$(printf %.10f $(( (10**10 * SECONDS - start) * 1000 ))e-10 >&1)
# results+=([$durationPP, $pixelCountPP])

echo ${results[@]}
