#!/bin/bash
# TODO: verify requirements installed, establish cuda capability limits
# Ptyhon 3.8, CUDA 11.2

export CUDA_HOME=/usr/lib/cuda

threadCount=(1, 2, 4, 8, 16, 32, 64, 128)
blockCount=(1, 2, 4, 8, 16, 32, 64, 128)

declare -A results
resPos=0

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
    imageName="images/houses_512.bmp"
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

declare -A results

echo "Running C Sequential"
((resPos+=1))
nvcc cuda_c/line_detection.cu
results[$resPos,1]=NA
results[$resPos,2]=NA
results[$resPos,3]=NA
results[$resPos,4]=NA
results[$resPos,5]=$(./a.out $imageName seq >&1)
results[$resPos,6]=NA



for t in ${threadCount[@]}; do
    for u in ${blockCount[@]}; do
        echo "Running C CUDA threads=$t blocks=$u"
        nvcc cuda_c/line_detection.cu -DNUM_THREADS=$t -DNUM_BLOCKS=$u

        IFS='::' read -ra TEMPMAIN <<< "$(./a.out $imageName par >&1)"
        for i in "${TEMPMAIN[@]}"; do
            IFS=';' read -ra TEMPSUB <<< "$i"
            if [ ${#TEMPSUB[@]} -gt 1 ]; then
                ((resPos+=1))
                tempPos=1

                for i in "${TEMPSUB[@]}"; do
                    results[$resPos,$tempPos]=$i
                    ((tempPos+=1))
                done
            fi
        done
    done
done

# echo "Running Python Sequential"
# ((resPos+=1))
# results[$resPos,1]=NA
# results[$resPos,2]=NA
# results[$resPos,3]=NA
# results[$resPos,4]=$($pythonCom py_cuda/line_detection.py $imageName seq 0 >&1)
# results[$resPos,5]=NA
# results[$resPos,6]=NA

# echo "Running Python CUDA (Numba)"
# for t in ${threadCount[@]}; do
#   ((resPos+=1))
#   tempPos=1
  
#   IFS=';' read -ra TEMP <<< "$($pythonCom py_cuda/line_detection.py $imageName par $t >&1)"
#   for i in "${TEMP[@]}"; do
#     results[$resPos,$tempPos]=$i
#     ((tempPos+=1))
#   done
# done

# blocks, threads, pixelsPerThread, pixelsPerBlock, totalTime, gpuTime, time difference
for ((j=1;j<=resPos;j++)) do
    printf "${results[$j,1]}, ${results[$j,2]}, ${results[$j,3]}, ${results[$j,4]}, ${results[$j,5]}, ${results[$j,6]}"
    echo
done
