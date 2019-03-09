Assignment 1 : Supervised Learning 
Link for Code and Data :
https://drive.google.com/drive/folders/1xKUvn9jEdFMS9g1wHOeImfWP3g5GV_UZ?usp=sharing


Software used : Development environment Spyder 3.3.2 is used. It is python IDE , which comes along with Anaconda 3 installation.It uses python version 3.7 .Spyder offers built-in integration with many popular scientific package including NumPy, SciPy, Pandas, IPython, QtConsole, Matplotlib, SymPy and more.
Spyder display results and graphs in IPython console.The IPython Console allows you to execute commands and enter, interact with and visualize data inside any number of fully featured IPython interpreters.
Spyder is included by default in the Anaconda Python distribution, which comes with         everything you need to get started in an all-in-one package.


Anaconda can be installed from this link :
https://www.anaconda.com/distribution/




Steps to run the code:
1. Launch spyder in Anaconda Navigator.
2. From the top menu bar select File->Open , to open the file from the files in the folder which you want to run for ex - spambase_DT.py
3. Click “Run” from the top menu bar  and you will see the results in Python console along with the charts and graphs.




Link for data in UCI Machine Learning Repository :
Spambase Dataset:        https://archive.ics.uci.edu/ml/datasets/spambase
EEG Eye State Dataset:        https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State


Note : EEG eye state dataset is available in UCI repository in .arff format , which is compatible to WEKA only. I have converted the dataset from arff to csv and stored in the “Data” folder in the google drive link shared.


Here goes the more details about data attributes:
Spambase Dataset:In total has 57 attributes .The attribute description and distribution is given below:
* 8 continuous real [0,100] attributes of type
word_freq_WORD = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A “word” in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
* 6 continuous real [0,100] attributes of type char_freq_CHAR = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail
* 1 continuous real [1,…] attribute of type capital_run_length_average = average length of uninterrupted sequences of capital letters
* 1 continuous integer [1,…] attribute of type capital_run_length_longest = length of longest uninterrupted sequence of capital letters
* 1 continuous integer [1,…] attribute of type capital_run_length_total = sum of length of uninterrupted sequences of capital letters = total number of capital letters in the e-mail


EEG Eye State Dataset: This dataset features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.