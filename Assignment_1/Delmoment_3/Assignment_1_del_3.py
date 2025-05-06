# Del 1: Installationer och importera bibliotek
# Kör detta först för att sätta upp miljön

# Installera nödvändiga bibliotek (om de inte redan finns)
!pip install -q tqdm seaborn scikit-learn

# Importera bibliotek
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm.notebook import tqdm

# Kontrollera om GPU är tillgänglig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Använder enhet: {device}")

# Del 2: Ladda ner dataset (om du inte redan gjort det) och ange sökvägar

# OBS: Kör dessa steg OM du inte redan har laddat upp kaggle.json och laddat ner datasettet
from google.colab import files
uploaded = files.upload()  # Välj din kaggle.json när dialogen visas

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip

# Ange sökvägar
data_dir = "/content/chest_xray"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

# Verifiera att mapparna finns
print(f"Träningsdata finns: {os.path.exists(train_dir)}")
print(f"Testdata finns: {os.path.exists(test_dir)}")
print(f"Valideringsdata finns: {os.path.exists(val_dir)}")

# Visa antalet bilder i varje mapp
if os.path.exists(train_dir):
    print("\nAntal träningsbilder:")
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"  {class_name}: {num_images}")

if os.path.exists(val_dir):
    print("\nAntal valideringsbilder:")
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"  {class_name}: {num_images}")

if os.path.exists(test_dir):
    print("\nAntal testbilder:")
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"  {class_name}: {num_images}")

# Skapa mapp för att spara modeller
os.makedirs("./models", exist_ok=True)

# Del 3: Visualisering av exempelbilder
# Denna del kan köras oberoende för att se exempel på röntgenbilder

import random
from PIL import Image
import matplotlib.pyplot as plt

def show_sample_images(directory, n_samples=3):
    """Visar exempelbilder från varje klass i angiven mapp"""
    classes = os.listdir(directory)
    plt.figure(figsize=(15, 5*len(classes)))
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        images = os.listdir(class_dir)
        sample_images = random.sample(images, min(n_samples, len(images)))
        
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            plt.subplot(len(classes), n_samples, i*n_samples + j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{class_name}: {img_name}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visa några exempel på bilder från träningsuppsättningen
print("Exempel på röntgenbilder från träningsuppsättningen:")
show_sample_images(train_dir)

# Visa storleksfördelning för bilderna
def analyze_image_sizes(directory):
    """Analyserar och visar storleksfördelning för bilderna"""
    classes = os.listdir(directory)
    widths = []
    heights = []
    
    # Samla storleksinformation
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        image_files = os.listdir(class_dir)
        # Begränsa till max 100 bilder per klass för snabbare analys
        sample_images = random.sample(image_files, min(100, len(image_files)))
        
        for img_name in sample_images:
            try:
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path)
                width, height = img.size
                widths.append(width)
                heights.append(height)
            except Exception as e:
                print(f"Kunde inte öppna {img_name}: {e}")
    
    # Skapa diagram över storleksfördelning
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20)
    plt.title('Fördelning av bildbredder')
    plt.xlabel('Bredd (pixlar)')
    plt.ylabel('Antal bilder')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20)
    plt.title('Fördelning av bildhöjder')
    plt.xlabel('Höjd (pixlar)')
    plt.ylabel('Antal bilder')
    
    plt.tight_layout()
    plt.show()
    
    # Visa sammanfattande statistik
    print(f"Genomsnittlig bredd: {sum(widths)/len(widths):.1f} pixlar")
    print(f"Genomsnittlig höjd: {sum(heights)/len(heights):.1f} pixlar")
    print(f"Min bredd: {min(widths)} pixlar")
    print(f"Max bredd: {max(widths)} pixlar")
    print(f"Min höjd: {min(heights)} pixlar")
    print(f"Max höjd: {max(heights)} pixlar")

# Analysera bildstorlekar (valfritt, kan ta lite tid)
print("\nAnalys av bildstorlekar:")
analyze_image_sizes(train_dir)

# Del 4: Kontrollera och skapa valideringsuppsättning
# Detta steg är viktigt eftersom originaldata ofta har en liten valideringsuppsättning

from sklearn.model_selection import train_test_split
import shutil

# Kontrollera om valideringsuppsättningen är för liten (vanligt problem med detta dataset)
if os.path.exists(val_dir) and (
   len(os.listdir(os.path.join(val_dir, 'NORMAL'))) < 20 or 
   not os.path.exists(os.path.join(val_dir, 'NORMAL'))):
    
    print("Varning: Valideringsuppsättningen är för liten eller saknas. Skapar en ny från träningsdata.")
    
    # Skapa ny valideringsmapp
    new_val_dir = os.path.join(data_dir, "new_val")
    os.makedirs(os.path.join(new_val_dir, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, 'PNEUMONIA'), exist_ok=True)
    
    # Funktion för att dela upp en mapp i träning och validering
    def split_folder(src_folder, val_folder, val_split=0.2):
        """Skapar en valideringsuppsättning från träningsdata"""
        
        # Kontrollera om källmappen finns
        if not os.path.exists(src_folder):
            print(f"Varning: Mappen {src_folder} finns inte.")
            return 0, 0
        
        # Hämta alla filer
        all_files = os.listdir(src_folder)
        
        # Dela upp filerna
        train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=42)
        
        # Flytta filer till validering
        for file in val_files:
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(val_folder, file)
            shutil.copy(src_path, dst_path)
        
        return len(train_files), len(val_files)
    
    # Dela upp NORMAL och PNEUMONIA-mapparna
    normal_train, normal_val = split_folder(
        os.path.join(train_dir, 'NORMAL'), 
        os.path.join(new_val_dir, 'NORMAL')
    )
    
    pneumonia_train, pneumonia_val = split_folder(
        os.path.join(train_dir, 'PNEUMONIA'), 
        os.path.join(new_val_dir, 'PNEUMONIA')
    )
    
    print(f"Skapade valideringsuppsättning: {normal_val} NORMAL och {pneumonia_val} PNEUMONIA-bilder")
    
    # Uppdatera sökvägen till valideringsuppsättningen
    val_dir = new_val_dir
    print(f"Ny valideringssökväg: {val_dir}")
else:
    print("Valideringsuppsättningen ser bra ut. Fortsätter med befintliga data.")

# Del 5: Skapa datauppsättningar och dataloaders
# Detta skapar de transformationer och datauppsättningar som krävs för träning

# Sätt hyperparametrar - ändra dessa om du vill experimentera
BATCH_SIZE = 16  # Mindre batch-storlek för Colab
NUM_EPOCHS = 10  # Färre epoker för Colab-sessioner
LEARNING_RATE = 0.001
IMAGE_SIZE = 224  # ResNet50 förväntar 224x224 pixlar

# Datatransformeringar
# För träning - lägg till dataförstärkning
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),  # Slumpmässig horisontell spegling
    transforms.RandomRotation(10),      # Slumpmässig rotation med max 10 grader
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Justera ljusstyrka/kontrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# För validering och test - bara ändra storlek och normalisera
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ladda datauppsättningarna
print("Laddar datauppsättningar...")
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# Skapa dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Hämta datasetinformation
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"Klasser: {class_names}")
print(f"Träningsexempel: {len(train_dataset)}")
print(f"Valideringsexempel: {len(val_dataset)}")
print(f"Testexempel: {len(test_dataset)}")

# Visa klassfördelning
def show_class_distribution():
    """Visar klassfördelningen i dataset med diagram"""
    
    # Beräkna antal per klass
    train_counts = [0] * num_classes
    val_counts = [0] * num_classes
    test_counts = [0] * num_classes
    
    for _, label in train_dataset.samples:
        train_counts[label] += 1
        
    for _, label in val_dataset.samples:
        val_counts[label] += 1
        
    for _, label in test_dataset.samples:
        test_counts[label] += 1
    
    # Skapa diagram
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, train_counts, width, label='Träning')
    plt.bar(x, val_counts, width, label='Validering')
    plt.bar(x + width, test_counts, width, label='Test')
    
    plt.title('Antal bilder per klass')
    plt.xlabel('Klass')
    plt.ylabel('Antal bilder')
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Beräkna procent obalans
    train_total = sum(train_counts)
    val_total = sum(val_counts)
    test_total = sum(test_counts)
    
    print("\nFördelning av klasser:")
    for i, class_name in enumerate(class_names):
        train_pct = train_counts[i] / train_total * 100
        val_pct = val_counts[i] / val_total * 100
        test_pct = test_counts[i] / test_total * 100
        
        print(f"{class_name}:")
        print(f"  - Träning: {train_counts[i]} ({train_pct:.1f}%)")
        print(f"  - Validering: {val_counts[i]} ({val_pct:.1f}%)")
        print(f"  - Test: {test_counts[i]} ({test_pct:.1f}%)")

# Visa klassfördelningen
show_class_distribution()

# Visa några exempel på transformerade träningsbilder
def show_transformed_samples():
    """Visar några exempel på transformerade träningsbilder"""
    # Hämta några slumpmässiga bilder från träningsuppsättningen
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Visa bilderna
    plt.figure(figsize=(15, 8))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i+1)
        # Konvertera tensor till bild
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Avnormalisera
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'Klass: {class_names[labels[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visa transformerade träningsbilder
print("\nExempel på transformerade träningsbilder:")
show_transformed_samples()

# Del 6: Definiera modeller och träningsfunktioner
# Detta definierar modellarkitektur och funktioner men tränar inte än

# Definiera en egen CNN-modell från grunden
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        # Första konvolutionsblocket
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Andra konvolutionsblocket
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Tredje konvolutionsblocket
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fjärde konvolutionsblocket
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Beräkna storleken efter konvolutioner och poolning
        # Utgående från 224x224, efter 4 max pooling-lager (224/2^4)
        final_size = IMAGE_SIZE // 16
        
        # Fullständigt anslutna lager
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Konvolutionsblock
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Platta ut
        x = x.view(x.size(0), -1)
        
        # Fullständigt ansluten
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Beräkna klassvikter för att hantera obalanserad data
total_samples = len(train_dataset)
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
weight_normal = total_samples / (2 * num_normal)
weight_pneumonia = total_samples / (2 * num_pneumonia)
class_weights = torch.tensor([weight_normal, weight_pneumonia], device=device)
print(f"Klassvikter: NORMAL={weight_normal:.4f}, PNEUMONIA={weight_pneumonia:.4f}")

# Funktion för att träna och validera en modell
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30, model_name="model", early_stopping_patience=5):
    # Initialisera variabler för att hålla koll på bästa modell
    best_model_wts = None
    best_acc = 0.0
    patience_counter = 0  # Early stopping-räknare
    
    # Träningshistorik
    history = {
        'train_losses': [],
        'val_accuracies': [],
        'best_epoch': 0,
        'best_accuracy': 0.0
    }
    
    training_log = []
    
    # Skapa modellmapp
    model_dir = os.path.join("./models", f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Starttid
    start_time = time.time()
    
    # Loop över epoker
    for epoch in range(num_epochs):
        print(f'Epok {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Träningsfas
        model.train()
        running_loss = 0.0
        
        # Framstegsindikator för träning
        train_pbar = tqdm(train_loader, desc=f"Tränar Epok {epoch+1}/{num_epochs}")
        
        # Iterera över data
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Nollställ parametergradienterna
            optimizer.zero_grad()
            
            # Framåtpass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Bakåtpass och optimera
            loss.backward()
            optimizer.step()
            
            # Statistik
            running_loss += loss.item() * inputs.size(0)
            
            # Uppdatera framstegsindikator
            train_pbar.set_postfix({"Loss": loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Träningsförlust: {epoch_loss:.4f}')
        
        # Valideringsfas
        model.eval()
        correct = 0
        total = 0
        
        # Framstegsindikator för validering
        val_pbar = tqdm(val_loader, desc=f"Validerar Epok {epoch+1}/{num_epochs}")
        
        # Ingen gradientberäkning behövs
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Framåtpass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Statistik
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Uppdatera framstegsindikator
                current_acc = correct / total
                val_pbar.set_postfix({"Noggrannhet": f"{current_acc:.4f}"})
        
        epoch_acc = correct / total
        print(f'Valideringsnoggrannhet: {epoch_acc:.4f}')
        
        # Spara historik
        history['train_losses'].append(epoch_loss)
        history['val_accuracies'].append(epoch_acc)
        
        # Logga epokresultat
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_accuracy': epoch_acc
        }
        training_log.append(epoch_log)
        
        # Spara träningslogg efter varje epok
        with open(os.path.join(model_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=4)
        
        # Deep copy av modellen om den är bäst hittills
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_accuracy'] = best_acc
            patience_counter = 0  # Återställ räknaren för early stopping
            
            # Spara bästa modell
            checkpoint = {
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': best_acc,
                'model_name': model_name,
                'params': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'num_epochs': NUM_EPOCHS
                },
                'training_history': history
            }
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pt'))
            print(f'Sparade ny bästa modell med noggrannhet: {best_acc:.4f}')
        else:
            patience_counter += 1  # Öka early stopping-räknaren
            print(f"Ingen förbättring. Early stopping-räknare: {patience_counter}/{early_stopping_patience}")
        
        # Kontrollera om vi ska avbryta träningen tidigt
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping efter {epoch+1} epoker')
            break
        
        print()
    
    # Träningstid
    time_elapsed = time.time() - start_time
    print(f'Träning slutförd på {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Bästa valideringsnoggrannhet: {best_acc:.4f}')
    
    # Ladda bästa modellvikter
    model.load_state_dict(best_model_wts)
    
    # Spara slutlig modell
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': history['train_losses'],
        'val_accuracies': history['val_accuracies'],
        'best_epoch': history['best_epoch'],
        'best_accuracy': history['best_accuracy'],
        'model_name': model_name,
        'params': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS
        },
        'training_history': history
    }
    torch.save(final_checkpoint, os.path.join(model_dir, 'final_model.pt'))
    
    return model, history, model_dir

# Funktion för att utvärdera modellen på testdata
def evaluate_model(model, test_loader, model_dir):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    # Framstegsindikator för testning
    test_pbar = tqdm(test_loader, desc="Utvärderar på testset")
    
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Uppdatera framstegsindikator
            current_acc = correct / total
            test_pbar.set_postfix({"Noggrannhet": f"{current_acc:.4f}"})
    
    # Beräkna noggrannhet
    test_accuracy = correct / total
    print(f'Testnoggrannhet: {test_accuracy:.4f}')
    
    # Beräkna förvirringsmatris
    cm = confusion_matrix(all_labels, all_preds)
    
    # Skapa en klassificeringsrapport
    cr = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Spara utvärderingsresultat
    evaluation = {
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr
    }
    
    with open(os.path.join(model_dir, 'test_evaluation.json'), 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    # Plotta förvirringsmatris
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Förutsagd')
    plt.ylabel('Sann')
    plt.title('Förvirringsmatris')
    
    # Spara förvirringsmatrisplott
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.show()
    
    return test_accuracy, cm, cr

# Skriv ut att modell och funktioner har laddats
print("Modell och träningsfunktioner har definierats och är klara att användas.")


# Del 7: Träna Custom CNN (Classic Learning)
# Denna del kan köras separat för att träna den egna CNN-modellen

def train_custom_model():
    print("\n=== Tränar egen CNN-modell från grunden (Classic Learning) ===\n")
    
    # Initialisera modell
    model = CustomCNN(num_classes=num_classes)
    model = model.to(device)
    
    # Ställ in förlustfunktion och optimerare
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Använd viktad förlustfunktion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Träna modellen
    model, history, model_dir = train_model(
        model, criterion, optimizer, train_loader, val_loader, 
        num_epochs=30,  # Ökad till 30 epoker
        model_name="custom_cnn",
        early_stopping_patience=5  # Avbryt efter 5 epoker utan förbättring
    )
    
    # Utvärdera modellen
    test_accuracy, cm, cr = evaluate_model(model, test_loader, model_dir)
    
    # Plotta träningshistorik
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'])
    plt.title('Träningsförlust')
    plt.xlabel('Epok')
    plt.ylabel('Förlust')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'])
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--')
    plt.plot(history['best_epoch'], history['best_accuracy'], 'ro')
    plt.title('Valideringsnoggrannhet')
    plt.xlabel('Epok')
    plt.ylabel('Noggrannhet')
    plt.grid(True)
    
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.show()
    
    # Visa klassificeringsrapport
    print("\nKlassificeringsrapport:")
    for cls in class_names:
        print(f"\nKlass: {cls}")
        print(f"  Precision: {cr[cls]['precision']:.4f}")
        print(f"  Recall: {cr[cls]['recall']:.4f}")
        print(f"  F1-score: {cr[cls]['f1-score']:.4f}")
    
    return model, test_accuracy, model_dir

# Kör träningen av den egna CNN-modellen
custom_model, custom_acc, custom_dir = train_custom_model()

# Spara resultaten i en global variabel för senare jämförelse
globals()['custom_model'] = custom_model
globals()['custom_acc'] = custom_acc
globals()['custom_dir'] = custom_dir

print(f"Träning av egen CNN klar. Modellen uppnådde {custom_acc:.4f} testnoggrannhet.")



# Del 8: Träna ResNet50 (Transfer Learning)
# Denna del kan köras separat för att träna transferinlärningsmodellen

def train_transfer_model():
    print("\n=== Tränar transferinlärningsmodell (ResNet50) ===\n")
    
    # Ladda förtränad ResNet50-modell
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Frysa alla lager först (för snabbare träning)
    for param in model.parameters():
        param.requires_grad = False
    
    # Modifiera det sista fullständigt anslutna lagret för vår binära klassificering
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Flytta modell till enhet
    model = model.to(device)
    
    # Ställ in förlustfunktion och optimerare
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Använd viktad förlustfunktion
    
    # Träna bara det sista lagret
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Träna modellen (fas 1 - endast sista lagret)
    print("Fas 1: Tränar endast det sista lagret...")
    model, history_phase1, _ = train_model(
        model, criterion, optimizer, train_loader, val_loader, 
        num_epochs=3, model_name="resnet50_phase1",
        early_stopping_patience=5  # Avbryt efter 5 epoker utan förbättring
    )
    
    # Tina upp de sista lagren för finjustering
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Ställ in olika inlärningshastigheter för olika delar
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE * 0.1}
    ])
    
    # Träna modellen (fas 2 - finjustering)
    print("Fas 2: Finjusterar de sista konvolutionslagren...")
    model, history_phase2, model_dir = train_model(
        model, criterion, optimizer, train_loader, val_loader, 
        num_epochs=27, model_name="resnet50_transfer",  # 30 totalt (3 + 27)
        early_stopping_patience=5  # Avbryt efter 5 epoker utan förbättring
    )
    
    # Kombinera historik från båda faserna
    combined_history = {
        'train_losses': history_phase1['train_losses'] + history_phase2['train_losses'],
        'val_accuracies': history_phase1['val_accuracies'] + history_phase2['val_accuracies'],
        'best_epoch': len(history_phase1['train_losses']) + history_phase2['best_epoch'] 
                     if history_phase2['best_accuracy'] > history_phase1['best_accuracy'] 
                     else history_phase1['best_epoch'],
        'best_accuracy': max(history_phase1['best_accuracy'], history_phase2['best_accuracy'])
    }
    
    # Utvärdera modellen
    test_accuracy, cm, cr = evaluate_model(model, test_loader, model_dir)
    
    # Plotta träningshistorik
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['train_losses'])
    plt.axvline(x=3, color='g', linestyle='--')
    plt.title('Träningsförlust')
    plt.xlabel('Epok')
    plt.ylabel('Förlust')
    plt.grid(True)
    plt.text(3, max(combined_history['train_losses'])*0.8, "Fas 2: Finjustering", rotation=90)
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['val_accuracies'])
    plt.axvline(x=3, color='g', linestyle='--')
    plt.axvline(x=combined_history['best_epoch'], color='r', linestyle='--')
    plt.plot(combined_history['best_epoch'], combined_history['best_accuracy'], 'ro')
    plt.title('Valideringsnoggrannhet')
    plt.xlabel('Epok')
    plt.ylabel('Noggrannhet')
    plt.grid(True)
    plt.text(3, max(combined_history['val_accuracies'])*0.8, "Fas 2: Finjustering", rotation=90)
    
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.show()
    
    # Visa klassificeringsrapport
    print("\nKlassificeringsrapport:")
    for cls in class_names:
        print(f"\nKlass: {cls}")
        print(f"  Precision: {cr[cls]['precision']:.4f}")
        print(f"  Recall: {cr[cls]['recall']:.4f}")
        print(f"  F1-score: {cr[cls]['f1-score']:.4f}")
    
    return model, test_accuracy, model_dir

# Kör träningen av ResNet50-modellen
transfer_model, transfer_acc, transfer_dir = train_transfer_model()

# Spara resultaten i en global variabel för senare jämförelse
globals()['transfer_model'] = transfer_model
globals()['transfer_acc'] = transfer_acc
globals()['transfer_dir'] = transfer_dir

print(f"Träning av ResNet50 transferinlärning klar. Modellen uppnådde {transfer_acc:.4f} testnoggrannhet.")


# Del 9: Jämför modellerna
# Denna del kan köras separat för att jämföra resultaten

def compare_models(custom_acc, transfer_acc, custom_dir, transfer_dir):
    print("\n=== Modelljämförelse ===\n")
    print(f"Egen CNN-noggrannhet: {custom_acc:.4f}")
    print(f"ResNet50 Transferinlärningsnoggrannhet: {transfer_acc:.4f}")
    
    # Ladda utvärderingsresultat
    with open(os.path.join(custom_dir, 'test_evaluation.json'), 'r') as f:
        custom_eval = json.load(f)
    
    with open(os.path.join(transfer_dir, 'test_evaluation.json'), 'r') as f:
        transfer_eval = json.load(f)
    
    # Jämför mätvärden
    custom_cr = custom_eval['classification_report']
    transfer_cr = transfer_eval['classification_report']
    
    # Skapa jämförelseplot
    plt.figure(figsize=(12, 10))
    
    # Noggrannhetsjämförelse
    plt.subplot(2, 2, 1)
    bars = plt.bar(['Egen CNN', 'ResNet50 Transfer'], [custom_acc, transfer_acc])
    plt.title('Testnoggrannhet')
    plt.ylim(0, 1)
    
    # Lägg till värdesetiketter på staplar
    for bar, acc in zip(bars, [custom_acc, transfer_acc]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha='center', fontsize=10)
    
    # Precisionsjämförelse
    plt.subplot(2, 2, 2)
    custom_precision = [custom_cr[c]['precision'] for c in class_names]
    transfer_precision = [transfer_cr[c]['precision'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, custom_precision, width, label='Egen CNN')
    plt.bar(x + width/2, transfer_precision, width, label='ResNet50 Transfer')
    plt.xlabel('Klass')
    plt.ylabel('Precision')
    plt.title('Precision per klass')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Recall-jämförelse
    plt.subplot(2, 2, 3)
    custom_recall = [custom_cr[c]['recall'] for c in class_names]
    transfer_recall = [transfer_cr[c]['recall'] for c in class_names]
    
    plt.bar(x - width/2, custom_recall, width, label='Egen CNN')
    plt.bar(x + width/2, transfer_recall, width, label='ResNet50 Transfer')
    plt.xlabel('Klass')
    plt.ylabel('Recall')
    plt.title('Recall per klass')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # F1-poängjämförelse
    plt.subplot(2, 2, 4)
    custom_f1 = [custom_cr[c]['f1-score'] for c in class_names]
    transfer_f1 = [transfer_cr[c]['f1-score'] for c in class_names]
    
    plt.bar(x - width/2, custom_f1, width, label='Egen CNN')
    plt.bar(x + width/2, transfer_f1, width, label='ResNet50 Transfer')
    plt.xlabel('Klass')
    plt.ylabel('F1-poäng')
    plt.title('F1-poäng per klass')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Spara jämförelse
    results_dir = os.path.join("./models", f"comparison_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    plt.show()
    
    # Skapa en sammanfattning av skillnaderna
    print("\n=== Prestandajämförelse ===")
    print(f"Egen CNN vs ResNet50 Transferinlärning:")
    print(f"  - Noggrannhet: {custom_acc:.4f} vs {transfer_acc:.4f} ({(transfer_acc-custom_acc)*100:.2f}% skillnad)")
    
    # Jämför klassspecifika mätvärden
    for cls in class_names:
        print(f"\nKlass: {cls}")
        print(f"  - Precision: {custom_cr[cls]['precision']:.4f} vs {transfer_cr[cls]['precision']:.4f}")
        print(f"  - Recall: {custom_cr[cls]['recall']:.4f} vs {transfer_cr[cls]['recall']:.4f}")
        print(f"  - F1-poäng: {custom_cr[cls]['f1-score']:.4f} vs {transfer_cr[cls]['f1-score']:.4f}")
    
    # Spara jämförelsedata
    comparison = {
        'custom_cnn': {
            'accuracy': custom_acc,
            'classification_report': custom_cr,
            'model_dir': custom_dir
        },
        'resnet50_transfer': {
            'accuracy': transfer_acc,
            'classification_report': transfer_cr,
            'model_dir': transfer_dir
        },
        'difference': {
            'accuracy': float(transfer_acc - custom_acc),
            'relative_improvement': float((transfer_acc - custom_acc) / custom_acc * 100) if custom_acc > 0 else 0
        }
    }
    
    with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Skriv ut en reflektion över transferinlärningens effektivitet
    transfer_improvement = (transfer_acc - custom_acc) / custom_acc * 100 if custom_acc > 0 else 0
    
    print("\n=== Transferinlärningsreflektion ===")
    print(f"ResNet50-transferinlärningsmodellen {'presterade bättre' if transfer_acc > custom_acc else 'presterade sämre'} än den egna CNN-modellen.")
    
    if transfer_acc > custom_acc:
        print(f"Den uppnådde {transfer_improvement:.2f}% högre noggrannhet på testsetet.")
        print("\nVarför transferinlärning är effektivt för denna uppgift:")
        print("1. Den förtränade ResNet50-modellen har redan lärt sig allmänna egenskaper som kanter, texturer och former från ImageNets olika dataset.")
        print("2. Dessa lågnivåegenskaper är överförbara och användbara för medicinska bilduppgifter, även om modellen inte tränats på röntgenbilder ursprungligen.")
        print("3. Vi behövde bara finjustera modellen på vårt specifika pneumonidataset, vilket kräver mindre träningsdata och tid.")
        print("4. ResNet50:s djupa arkitektur med dess skip-anslutningar möjliggör bättre gradientflöde och inlärning av komplexa egenskaper.")
    else:
        print(f"Den presterade {-transfer_improvement:.2f}% sämre än den egna CNN-modellen.")
        print("\nMöjliga orsaker till varför transferinlärning inte fungerade lika bra för denna uppgift:")
        print("1. Domänskillnaden mellan naturbilder (ImageNet) och medicinska röntgenbilder kan vara för betydande.")
        print("2. Vår egna CNN kan vara bättre optimerad för denna specifika binära klassificeringsuppgift.")
        print("3. Vi kan behöva fler finjusteringsepoker eller en annan inlärningsstrategi för transferinlärning.")
        print("4. Även om förtränade modeller ofta har fördel, kan en skräddarsydd lösning ibland vara överlägsen för mycket domänspecifika problem.")
    
    return comparison

# Försök att jämföra modellerna om båda har körts
try:
    # Kontrollera om båda modellerna har körts
    if 'custom_acc' in globals() and 'transfer_acc' in globals():
        comparison = compare_models(
            globals()['custom_acc'], 
            globals()['transfer_acc'], 
            globals()['custom_dir'], 
            globals()['transfer_dir']
        )
        print("Jämförelse slutförd!")
    else:
        print("OBS: Kan inte jämföra modellerna eftersom båda inte har körts än.")
        print("Kör både 'train_custom_model()' och 'train_transfer_model()' först.")
except Exception as e:
    print(f"Fel vid jämförelse: {e}")
    print("Kör både 'train_custom_model()' och 'train_transfer_model()' först.")


# Del 10: Extra analyser och felanalys
# Denna del kan köras när som helst efter att minst en modell har tränats

# Visualisera felaktigt klassificerade bilder
def visualize_misclassifications(model, data_loader, class_names, max_images=10):
    """Visualisera bilder som klassificeras felaktigt av modellen"""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Hitta felaktiga klassificeringar
            incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_idx:
                if len(misclassified_images) >= max_images:
                    break
                    
                misclassified_images.append(inputs[idx].cpu())
                misclassified_labels.append(labels[idx].item())
                misclassified_preds.append(preds[idx].item())
                
            if len(misclassified_images) >= max_images:
                break
    
    # Visualisera bilderna
    if misclassified_images:
        plt.figure(figsize=(15, min(len(misclassified_images), 10) * 2))
        
        for i, (img, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_labels, misclassified_preds)):
            # Konvertera tensor till bild
            img = img.permute(1, 2, 0).numpy()
            # Avnormalisera
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            plt.subplot(min(len(misclassified_images), 5), 2, i+1)
            plt.imshow(img)
            plt.title(f'Sann: {class_names[true_label]}\nFörutsagd: {class_names[pred_label]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Inga felaktigt klassificerade bilder hittades (eller datasettet är för litet).")

# Gradientbaserad visualisering för att se vilka delar av bilden modellen fokuserar på
def visualize_model_attention(model, img_path, class_names):
    """Visualisera vilka delar av bilden modellen fokuserar på med Grad-CAM"""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        # Om pytorch-grad-cam inte är installerat, installera det
        print("Installerar pytorch-grad-cam...")
        !pip install -q pytorch-grad-cam
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    
    # Ladda och förbered bilden
    img = Image.open(img_path)
    img = val_test_transform(img).unsqueeze(0).to(device)
    
    # Modell till utvärderingsläge
    model.eval()
    
    # Hitta rätt lager för CAM (sista konvolutionslagret)
    if isinstance(model, CustomCNN):
        target_layer = [model.conv4]
    else:  # ResNet50
        target_layer = [model.layer4[-1]]
    
    # Skapa GradCAM
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=device.type=='cuda')
    
    # Generera CAM
    grayscale_cam = cam(input_tensor=img)
    grayscale_cam = grayscale_cam[0, :]
    
    # Ladda originalbilden för visualisering
    orig_img = Image.open(img_path).convert('RGB')
    orig_img = orig_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    orig_img = np.array(orig_img) / 255.0
    
    # Visualisera
    visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
    
    # Förutsäg klass
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
    
    # Visa både originalbild och CAM-visualisering
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title(f'Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM: {predicted_class} ({confidence:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Funktion för att analysera träningskonvergens
def analyze_training_convergence(model_dir):
    """Analysera träningskonvergens från träningsloggen"""
    log_path = os.path.join(model_dir, 'training_log.json')
    
    if not os.path.exists(log_path):
        print(f"Ingen träningslogg hittades i {model_dir}")
        return
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Extrahera data
    epochs = [entry['epoch'] for entry in log_data]
    train_losses = [entry['train_loss'] for entry in log_data]
    val_accuracies = [entry['val_accuracy'] for entry in log_data]
    
    # Hitta bästa epok
    best_epoch = epochs[np.argmax(val_accuracies)]
    best_accuracy = max(val_accuracies)
    
    # Plotta konvergens
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Träningsförlust')
    
    # Lägg till trendlinje
    z = np.polyfit(epochs, train_losses, 1)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), "r--", label='Trend')
    
    plt.title('Träningsförlust över tid')
    plt.xlabel('Epok')
    plt.ylabel('Förlust')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Valideringsnoggrannhet')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.axhline(y=best_accuracy, color='r', linestyle='--')
    plt.plot(best_epoch, best_accuracy, 'ro')
    
    plt.annotate(f'Bäst: {best_accuracy:.4f} vid epok {best_epoch}', 
                 xy=(best_epoch, best_accuracy), 
                 xytext=(best_epoch - 2, best_accuracy - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Valideringsnoggrannhet över tid')
    plt.xlabel('Epok')
    plt.ylabel('Noggrannhet')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Beräkna konvergensmått
    if len(train_losses) >= 3:
        final_loss_avg = np.mean(train_losses[-3:])
        loss_improvement = train_losses[0] - final_loss_avg
        loss_improvement_pct = (loss_improvement / train_losses[0]) * 100
        
        acc_improvement = best_accuracy - val_accuracies[0]
        acc_improvement_pct = (acc_improvement / val_accuracies[0]) * 100 if val_accuracies[0] > 0 else 0
        
        print(f"Träningskonvergens analys:")
        print(f"  - Initial förlust: {train_losses[0]:.4f}")
        print(f"  - Slutlig genomsnittlig förlust: {final_loss_avg:.4f}")
        print(f"  - Förlustförbättring: {loss_improvement:.4f} ({loss_improvement_pct:.1f}%)")
        print(f"  - Initial noggrannhet: {val_accuracies[0]:.4f}")
        print(f"  - Bästa noggrannhet: {best_accuracy:.4f}")
        print(f"  - Noggrannhetsförbättring: {acc_improvement:.4f} ({acc_improvement_pct:.1f}%)")
        
        # Bedöm konvergens
        if train_losses[-1] > train_losses[-2] and train_losses[-2] > train_losses[-3]:
            print("  - Förlusttrend: Ökande (modellen kan vara övertränad)")
        elif np.std(train_losses[-3:]) < 0.01:
            print("  - Förlusttrend: Platå (träningen har konvergerat)")
        else:
            print("  - Förlusttrend: Fortfarande sjunkande (mer träning kan förbättra modellen)")
    else:
        print("För få epoker för att analysera konvergens")

# Körbart exempel: välj en modell att analysera (om minst en har tränats)
try:
    # Kontrollera om någon modell har tränats
    if 'custom_model' in globals():
        print("\n=== Extra analyser för egen CNN-modell ===")
        
        # Visa felaktigt klassificerade bilder
        print("\nFelaktigt klassificerade bilder (egen CNN):")
        visualize_misclassifications(globals()['custom_model'], test_loader, class_names)
        
        # Analysera träningskonvergens
        print("\nTräningskonvergens för egen CNN:")
        analyze_training_convergence(globals()['custom_dir'])
        
    elif 'transfer_model' in globals():
        print("\n=== Extra analyser för ResNet50-modell ===")
        
        # Visa felaktigt klassificerade bilder
        print("\nFelaktigt klassificerade bilder (ResNet50):")
        visualize_misclassifications(globals()['transfer_model'], test_loader, class_names)
        
        # Analysera träningskonvergens
        print("\nTräningskonvergens för ResNet50:")
        analyze_training_convergence(globals()['transfer_dir'])
    
    else:
        print("Ingen modell har tränats än. Kör modellträning först.")
        
except Exception as e:
    print(f"Ett fel uppstod vid analys: {e}")
    print("Se till att modeller har tränats och att nödvändiga bibliotek är installerade.")