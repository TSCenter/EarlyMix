for DATASET in `cat datasets.txt`
#for DATASET in ECG200
do
  for CLS_TYPE in variable
  do
    for LIKELIHOOD_TYPE in variable
    do
      for SL in 50
      do
        for CNNS in 2 3
        do
#          echo $DATASET $CLS_TYPE $LIKELIHOOD_TYPE $SL $CNNS
          python -u main_pretrain_cls_fcn.py $DATASET --num_epochs 2001 --cls_type ${CLS_TYPE} --num_cnns $CNNS > raw_outputs/${DATASET}_${CLS_TYPE}_${LIKELIHOOD_TYPE}_cnns_${CNNS}_strided_len_${SL}.log
          python -u main_early_second_order.py $DATASET --num_epochs 2001 --cls_type ${CLS_TYPE} --num_cnns $CNNS > raw_outputs/${DATASET}_${CLS_TYPE}_${LIKELIHOOD_TYPE}_cnns_${CNNS}_strided_len_${SL}.log
        done
      done
    done
  done
done
