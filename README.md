# Demo

A minimal example to reproduce `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` when using IBM ART.

## Experiment Setting
* Dataset: CIFAR-10   
* Model: PyTorch pretrained ResNet-18
* Attack: PGD
* To see the error: Run `python eval.py --config-file ./config.json`
