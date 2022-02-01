#!/bin/bash
# ./compare.sh -i images/houses.bmp -p python3
# source py_img/bin/activate
# pipreqs ./ --ignore py_img
# TODO: verify python3-opencv is installed
# opencv 4.5.5

while getopts "i:r:p:" flag
do
     case $flag in
         i) imageName=$OPTARG;;
         r) resultsFile=$OPTARG;;
         p) pythonCom=$OPTARG;;
     esac
done

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

# echo file: $imageName
# echo results: $resultsFile
# echo pythonCom: $pythonCom

# g++ -Wall line_detection.cpp ImageProcessing.cpp
# ./a.out $imageName $resultsFile

$pythonCom line_detection.py $imageName