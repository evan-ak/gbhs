# ReadMe

This is the source code for the paper *Graph-based Heuristic Search for Module Selection Procedure in Neural Module Network*.

<br/>

# Environment

The code was written and certified on Python 3.7 and PyTorch 1.4.0. 

The other required libraries include :

+ cv2
+ json
+ numpy
+ pandas (for saving results to .csv only)

<br/>

# Getting start

## Data preparation

### FigureQA

Download the raw data of FigureQA from <https://www.microsoft.com/en-us/research/project/figureqa-dataset/>. Extract them to `./data/`.

To pre-process the images, run :

```
python ./data/image_compress.py
```

We provide the other necessary pre-processed data for the five subsets as `/data/*.zip`. Extract them to `./data/` before usage.

They can also be generated from the raw data through :

```
python ./data/qa_formalize.py
python ./data/object_formalize.py
```

## Training

To train the model, run :
```
python ./train.py
```

After training, the log of programs and the parameters of modules are saved to `./saves/save_*`.

## Testing 

To test the model, run :
```
python ./test.py
```

For testing, a save point `./saves/save_*` must be assigned.

<br/>

# Notification

This is the early version of our algorithm and it is only for the verification on FigureQA. 

This version may not be kept maintained in the future.

Check <https://github.com/evan-ak/wsfl> for the latest version of our algorithm.
