## Requirements
 - set dataset in `./`
```
dataset
 - image_dir
 - annotation train txt file
 - annotation test txt file
```
annotation file format is like WFLW annotation txt file  

 - install pipenv

## Setup
```
git submodule update --init --recursive
pipenv install
```

## Run
```
python create_PCN_label.py [train or test]
```