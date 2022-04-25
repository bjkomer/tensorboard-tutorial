for units in 128 256 512
do
  for activation in relu sigmoid
  do
    for optimizer in adam rmsprop
    do
       for lr in 0.01 0.001 0.00001 0.0000001
       do
         python hyperparameter_example.py --units ${units} --activation ${activation} --optimizer ${optimizer} --lr ${lr}
       done
    done
  done
done
