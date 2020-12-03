#!/bin/bash
if [[ $# -lt 3 ]] ; then
  echo 'Please provide: '
  echo 'the number of mini-batches n > 0 (n = 1 for batch mode)'
  echo 'the number of Trees (ntrees > 0)'
  echo 'an output file name (without extension) for reporting results'
  exit 0
fi
n=$1
ntrees=$2
filename=$3.txt
echo ''>> $filename
echo "#########  Without labels  ##########" >> $filename
echo ''>> $filename
echo "Without labels:"
./mondrianforest_demo.py --dataset usps --n_mondrians $ntrees --n_minibatches $n &>> $filename
echo '    usps: ok'
./mondrianforest_demo.py --dataset satimage --n_mondrians $ntrees --n_minibatches $n &>> $filename
echo '    satimage: ok'
./mondrianforest_demo.py --dataset letter --n_mondrians $ntrees --n_minibatches $n &>> $filename
echo '    letter: ok'
./mondrianforest_demo.py --dataset dna --n_mondrians $ntrees --n_minibatches $n &>> $filename
echo '    dna: ok'
echo ''>> $filename
echo "\n\n#########  With labels  ##########" >> $filename
echo ''>> $filename
echo "With labels:"
./mondrianforest_demo.py --dataset usps --n_mondrians $ntrees --n_minibatches $n --split_policy 16 6 &>> $filename
echo '    usps: ok'
./mondrianforest_demo.py --dataset satimage --n_mondrians $ntrees --n_minibatches $n --split_policy 6 6 &>> $filename
echo '    satimage: ok'
./mondrianforest_demo.py --dataset letter --n_mondrians $ntrees --n_minibatches $n --split_policy 4 6 &>> $filename
echo '    letter: ok'
./mondrianforest_demo.py --dataset dna --n_mondrians $ntrees --n_minibatches $n --split_policy 13 6 &>> $filename
echo '    dna: ok'
echo 'Finished successfully !'
