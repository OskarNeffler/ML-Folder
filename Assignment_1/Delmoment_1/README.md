# OCR-Perceptron for Handwritten Digits (MNIST)

Detta projekt implementerar olika versioner av perceptron (neurala nätverk) för OCR-igenkänning av handskrivna siffror med hjälp av MNIST-datasetet. Implementationerna varierar från en enkel neuron till en PyTorch-modell som körs på en CUDA GPU, samt en optimerad CNN-modell.

## Projektstruktur

Projektet är organiserat i olika delar (A, B, C, D, och optimerad CNN), där varje del har sin egen implementationsfil:

- `A_single_neuron.py`: Implementation av en enskild neuron med NumPy-vektorisering
- `B_numpy_layer.py`: Implementation av neurala nätverkslager med NumPy
- `C_pytorch_model.py`: PyTorch neural network implementation (CPU)
- `D_cuda_model.py`: PyTorch neural network implementation (CUDA GPU)
- `optimized_cnn.py`: Optimerad CNN-implementation med data augmentation

## Krav

- Python 3.10 eller högre
- NumPy
- PyTorch 2.1 eller högre
- Matplotlib
- tqdm
- En CUDA-kompatibel GPU (för Del D och optimized_cnn)

Du kan installera de nödvändiga paketen med:

```bash
pip install numpy torch torchvision matplotlib tqdm
```

## Köra projektet

Varje del kan köras individuellt direkt från respektive fil:

```bash
# Kör Del A: Enskild neuron
python A_single_neuron.py

# Kör Del B: NumPy neural network
python B_numpy_layer.py

# Kör Del C: PyTorch på CPU
python C_pytorch_model.py

# Kör Del D: PyTorch på CUDA GPU
python D_cuda_model.py

# Kör optimerad CNN-modell med data augmentation
python optimized_cnn.py
```

## Delbeskrivningar

### Del A: Enskild neuron med NumPy

Denna del implementerar en enskild neuron med hjälp av NumPy för vektoriserade beräkningar. Implementationen inkluderar olika aktiveringsfunktioner (Sigmoid, ReLU, Leaky ReLU och Tanh).

### Del B: Neuralt nätverkslager (NumPy)

Denna del implementerar ett komplett neuralt nätverkslager med NumPy för matrisoperationer. Implementationen inkluderar:
- Forward pass genom nätverket
- Olika aktiveringsfunktioner
- Enkel prediktionsfunktionalitet

### Del C: PyTorch-implementation (CPU)

Denna del implementerar ett neuralt nätverk med PyTorch's högninvå-API:er, inklusive:
- Nätverksarkitekturdefinition med `nn.Module`
- Träningsloop med förlustberäkning och optimering
- Utvärdering och prestationsmått
- Modell-sparande

### Del D: PyTorch-implementation (CUDA GPU)

Denna del utökar PyTorch-implementationen för att köras på en CUDA-kompatibel GPU för acceleration, inklusive:
- GPU-enhetsval
- Data- och modellöverföring till GPU
- Körningstidsjämförelse

### Optimerad CNN-implementation

Denna del implementerar en förbättrad CNN-modell med avancerade tekniker:
- Convolutional Neural Network (CNN)-arkitektur
- Data augmentation för förbättrad generalisering
- Learning rate scheduling
- Dropout för regularisering

## Resultat

- Del A (Single Neuron): Demonstrerar en enskild neurons beräkningshastighet
- Del B (NumPy Network): Visar forward pass-prestanda för en hel batchbearbetning
- Del C (PyTorch CPU): ~91% accuracy efter 1 epoch
- Del D (PyTorch GPU): ~91% accuracy efter 1 epoch, ~97% efter 10 epoker
- Optimerad CNN: ~99.5% accuracy efter 20 epoker med data augmentation 