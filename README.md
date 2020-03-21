# anomaly-detection

- Anomaly detection  
  1. vanilla CNN (supervised)
  2. resnet18 (supervised)

- Dependencies
  - Python 3.6+
  - PyTorch==1.3
  - Dataset: [MVTec Anomaly Detection Dataset]
  
  
### Dataset structure

```
./data   
│
├── bottle
│   ├── ground_truth
│   ├── train
│   └── test
│
├── carpet
├── leather
├── grid

```

### Train model
* Run the following command.
```
python train.py --target 'bottle' -c 0 --lr 0.001
```
  
### Reference
1. [to do]


[MVTec Anomaly Detection Dataset]: https://www.mvtec.com/company/research/datasets/mvtec-ad/

   
