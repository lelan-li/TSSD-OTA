import numpy as np

original_file = './train.txt'
write_file = './train_reduce.txt'

reduce_to_size = 20000
with open(write_file, 'w') as wf:
    with open(original_file, 'r') as f:
        lines = f.readlines()
        ind = np.linspace(0, len(lines)-1, reduce_to_size).astype(np.int)
        for i in ind:
            wf.write(lines[i])
