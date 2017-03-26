# Recurrent Neural Networks
***This repository contains the code for Recurrent Neural Network from scratch using Python 3 and numpy.***

# Requirements
**Numpy**

# Sample Output
```
Error:[ 3.94375112]
Pred:[1 1 1 1 1 1 1 1]
True:[0 0 1 1 0 1 1 1]
28 + 27 = 255
------------
Error:[ 3.89378112]
Pred:[0 0 1 0 1 0 1 0]
True:[1 0 0 1 0 0 1 0]
21 + 125 = 42
------------
Error:[ 3.80079469]
Pred:[1 1 1 0 1 0 0 1]
True:[1 1 0 1 1 0 0 1]
100 + 117 = 233
------------
Error:[ 3.75256184]
Pred:[0 0 0 1 0 0 0 0]
True:[0 0 1 0 0 1 1 1]
12 + 27 = 16
------------
Error:[ 3.47163732]
Pred:[0 0 0 1 0 1 0 1]
True:[0 1 0 1 1 1 0 1]
67 + 26 = 21
------------
Error:[ 3.46614289]
Pred:[1 1 1 0 0 1 1 0]
True:[1 0 0 0 0 1 1 0]
92 + 42 = 230
------------
Error:[ 0.57723326]
Pred:[0 1 1 1 0 1 1 0]
True:[0 1 1 1 0 1 1 0]
86 + 32 = 118
------------
Error:[ 0.83430643]
Pred:[1 1 1 0 1 0 1 0]
True:[1 1 1 0 1 0 1 0]
107 + 127 = 234
------------
Error:[ 0.50010502]
Pred:[0 0 1 0 1 0 0 0]
True:[0 0 1 0 1 0 0 0]
15 + 25 = 40
------------
Error:[ 0.42438922]
Pred:[0 1 1 1 0 1 1 1]
True:[0 1 1 1 0 1 1 1]
28 + 91 = 119
------------
```

**Note that the RNN keeps on training, predicting output values and collecting dJdW2 and dJdW1 values at each output stage. Once it reaches the last stage of an addition, it starts backpropagating all the errors till the first stage. Hence, after initial 3-4 steps it starts predicting the accurate output.**

# Resources

| S.No.  |                       Papers / Blogs / Authors            |                        Paper Links                   |
| ------ | --------------------------------------------------------- | ---------------------------------------------------- |
|1.      |"A Critical Review of RNN for Sequence Learning" by Zachary C. Lipton|    https://arxiv.org/pdf/1506.00019.pdf    |
|2.      |                    "i am trask" Blog                      |https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/|


