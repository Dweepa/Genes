#/bin/bash

array=($(find . | grep report* --line-buffered))
for u in "${array[@]}"
do
      echo awk $u \'NR==1{print $2}\'
done
