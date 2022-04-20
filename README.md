# QuoTS
Matching patterns in time series with natural language

### Content Description
This repository has the code available to reproduce results and experiment
on any data that you might want to try on.
Some available examples of signals can be used to try out the QuoTS app. 

Additionally, you have access to matlab livescripts with example of code
and search queries. 

We also provide a video of interacting with the interface.

### Scrolling and using content

The code can be used in several ways. You can use the individual functions available in "\functions\" as exemplified in "\examples\telemetry_examples.md", or calling the interface
from your own signal using the method **Quots(signal, win_size)**.

The code for each word feature vector is available as well and can be 
visualized at "examples\wordfv_match.md".

You can also use the available interface at "\interface\Quots_app.mlapp" and load the available example signals (example_1, example_2, example_3 and example_4). 

### QuoTS for UCR examples

You can also use QuoTS to sort classes of signals based on a query. For that you can use the available interface at "interfaces\NL_UCR_DATA_app.mlapp". Use the name of the UCR dataset
and it can be used to sort descritely each signal based on a specific query.

### Dataset

We share the acquired dataset, which can be used as "example_3" on the interface. More information on the dataset setup and acquisition is available at "datasets\example3\".

### Using specific shapes as queries
In order to use queries based on specific shapes that you might 
have available, you can add a txt file at "docs\special_wfv\". The 
file name should be the word to assign to the signal inside the txt file.
Only 1D signals are supported.