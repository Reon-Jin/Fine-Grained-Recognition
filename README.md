# Fine Grained Recognition

This project provides a minimal setup for training a fine-grained image classifier using PyTorch.

## Structure

- `models/` - model definitions
- `config.py` - configuration parameters
- `data_prepare.py` - dataset loading and split
- `main.py` - training script
- `trained_model/` - saved weights

## Usage

1. Place your dataset under the `data/` folder in the standard ImageFolder format:
   ```
   data/
       class1/xxx.jpg
       class2/yyy.jpg
       ...
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python main.py
   ```
   The best model based on validation accuracy will be saved to `trained_model/best_model.pth`.

During training, progress bars display epoch statistics and matplotlib shows live curves of loss and accuracy.
