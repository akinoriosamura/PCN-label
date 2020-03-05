## Setup
```
git submodule update --init --recursive
docker-compose up -d
docker attach pcn-label
sh build_PCN.sh
pipenv install
pipenv shell
```

## for creating pcn label txt file
### Requirements
 - set dataset in `./`
```
dataset
 - image_dir
 - annotation train txt file
 - annotation test txt file
```
annotation file format is like WFLW annotation txt file  

 - install pipenv


### Run
```
python create_PCN_label.py [train or test]
```

## for changing pcn bb label in xml
### abstruct
 - change txt bb to pcn bb

### Requirements
 - prepare bb and landmark xml label  
 - change xml to txt line  
 - install pipenv  

### Run
```
python change_bb_to_PCN.py
```
