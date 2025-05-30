# Experiment: MNIST Classifier with Advanced Techniques

## Objective
Build a neural network to classify MNIST handwritten digits with the following goals:
- Achieve >99.5% test accuracy
- Model size under 100k parameters
- Training time under 5 minutes on CPU

## Approach
Start with a simple CNN and iteratively improve using:
1. Data augmentation (rotation, shifting)
2. Regularization techniques (dropout, batch norm)
3. Architecture optimization
4. Learning rate scheduling

## Success Criteria
- Test accuracy consistently above 99.5%
- Fast inference time (<10ms per image)
- Robust to slightly rotated/shifted inputs

## Constraints
- Use PyTorch
- No pre-trained models
- Standard MNIST dataset only