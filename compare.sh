#!/bin/bash

export CUDA_HOME=/usr/lib/cuda

# threadCount=(1, 2, 4, 8, 16, 32, 64, 128)
# blockCount=(1, 2, 4, 8, 16, 32, 64, 128)
declare -A results
resPos=0
threadCount=(32)
blockCount=(256)

run_c_cuda()
{
     >&2 echo "Running C Sequential"
    ((resPos+=1))
    # nvcc cuda_c/line_detection.cu
    results[$resPos,1]=0
    results[$resPos,2]=0
    results[$resPos,3]=0
    results[$resPos,4]=0
    results[$resPos,5]=0
    results[$resPos,6]=50.555
    # results[$resPos,6]=$(./a.out $imageName seq >&1)
    results[$resPos,7]=0

    >&2 echo "Running C Sequential"
    ((resPos+=1))
    # nvcc cuda_c/line_detection.cu
    results[$resPos,1]='{0,0,0}'
    results[$resPos,2]='{0,0,0}'
    results[$resPos,3]=0
    results[$resPos,4]=0
    results[$resPos,5]=0
    results[$resPos,6]=100
    # results[$resPos,6]=$(./a.out $imageName seq >&1)
    results[$resPos,7]=0

    # for t in ${threadCount[@]}; do
    #     for u in ${blockCount[@]}; do
    #         >&2 echo "Running C CUDA threads=$t blocks=$u"
    #         nvcc cuda_c/line_detection.cu -DNUM_THREADS=$t -DNUM_BLOCKS=$u

    #         IFS='::' read -ra TEMPMAIN <<< "$(./a.out $imageName par 1>&1 2>>errors.txt)"
    #         for i in "${TEMPMAIN[@]}"; do
    #             IFS=';' read -ra TEMPSUB <<< "$i"
    #             if [ ${#TEMPSUB[@]} -gt 1 ]; then
    #                 ((resPos+=1))
    #                 tempPos=1

    #                 for i in "${TEMPSUB[@]}"; do
    #                     results[$resPos,$tempPos]=$i
    #                     ((tempPos+=1))
    #                 done
    #             fi
    #         done
    #     done
    # done
}

run_py_cuda()
{
    >&2 echo "Running Python Sequential"
    ((resPos+=1))
    results[$resPos,1]=0
    results[$resPos,2]=0
    results[$resPos,3]=0
    results[$resPos,4]=0
    results[$resPos,5]=0
    results[$resPos,6]=400.2342
    # results[$resPos,6]=$($pythonCom py_cuda/line_detection.py $imageName seq 0 0 >&1)
    results[$resPos,7]=0

    # for t in ${threadCount[@]}; do
    #     for u in ${blockCount[@]}; do
    #         >&2 echo "Running Python CUDA threads=$t blocks=$u"

    #         IFS='::' read -ra TEMPMAIN <<< "$($pythonCom py_cuda/line_detection.py $imageName par $t $u 1>&1 2>>errors.txt)"
    #         for i in "${TEMPMAIN[@]}"; do
    #             IFS=';' read -ra TEMPSUB <<< "$i"
    #             if [ ${#TEMPSUB[@]} -gt 1 ]; then
    #                 ((resPos+=1))
    #                 tempPos=1

    #                 for i in "${TEMPSUB[@]}"; do
    #                     results[$resPos,$tempPos]=$i
    #                     ((tempPos+=1))
    #                 done
    #             fi
    #         done
    #     done
    # done
}

# set variables based on flag arguments
while getopts "i:r:p:t:" flag
do
     case $flag in
         i) imageName=$OPTARG;;
         r) resultsFile=$OPTARG;;
         p) pythonCom=$OPTARG;;
         t) typeRun=$OPTARG;;
     esac
done

# set default variables
if [ -z "$typeRun" ]
then
    echo "Please specify whether to run CUDA C (1), Python Numba CUDA (2), or both (3)"
    exit 0
fi

if [ -z "$resultsFile" ]
then
    echo "Please specify a file for numerical results"
    exit 0
fi

if [ -z "$imageName" ]
then
    imageName="images/houses_512.bmp"
fi

if [ -z "$pythonCom" ]
then
    echo "Please specify command to run Python script"
    exit 0
fi

if [[ "$typeRun" == "1" ]]; then
  run_c_cuda
elif [[ "$typeRun" == "2" ]]; then
  run_py_cuda
else
  run_c_cuda
  run_py_cuda
fi

# blocksDim, threadsDim, pixelsPerThread, pixelsPerBlock, time difference (totalTime - gputime), pixelsPerMSTotal, pixelsPerMSGPU
for ((j=1;j<=resPos;j++)) do
    echo "${results[$j,1]}, ${results[$j,2]}, ${results[$j,3]}, ${results[$j,4]}, ${results[$j,5]}, ${results[$j,6]}, ${results[$j,7]}" >> $resultsFile
    echo "${results[$j,1]};${results[$j,2]};${results[$j,3]};${results[$j,4]};${results[$j,5]};${results[$j,6]};${results[$j,7]}"
done

