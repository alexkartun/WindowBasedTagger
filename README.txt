Instructions:
Executing tagger1.py:
All arguments are in 'config' file. Each argument in new row.
1st argument: number of task {1, 3, 4}.
2nd argument: type of tagger {ner, pos}.
3rd argument: learning rate.
4th argument: number of epochs.
5th argument: hidden layer dim.
6th argument: include embedding vectors {1:yes, 0:not}.
7th arguments: include sub word units {1:yes, 0:not}.
note: I added config example file. You can play with each one of the arguments explained above for each of the tasks.
***********************************************************************************
Example of config file:
1
pos
0.005
3
100
1
0
**********************************************************************************
Additional files:
After executing of the script 'test{}.pos' or 'test{}.ner' file will be created as an output of the model on test data. In the brackets
will be number of the task.
In addition will be created 2 images: 'acc_dev{}.png' and 'loss_dev{}.png'. Same explanation of the brackets like above.
This are the graphs showing the accuracy and loss on the dev data as a function of the number of epochs.

The script is laying on relative directories 'pos' and 'ner' which every one of them including train, dev and test data.
So, for the script to run smoothly you should put both directories near the 'tagger1.py' file.
The 'config' should be near the 'tagger1.py' file.
And for addition 'vocab' and 'wordVectors' files should be near the 'tagger1.py' file.
The output files I described before will be saved in one of this directories depends on the user's input, 'pos' or 'ner'.