# ML-Folder

Repository för maskininlärningsprojekt och övningar.

## Innehåll

### Assignment 1
Implementering av neurala nätverk med olika metoder för bildklassificering (MNIST och SVHN).

### Assignment 2: Multi-Agent AI Debattplattform

En implementering av ett multi-agent AI-system där olika AI-personligheter debatterar den optimala datorplattformen för AI/ML-kurser.

#### Funktioner
- **AIAgent-klass** för olika plattformsspecialister (Linux, macOS, Windows, Cloud)
- **DebateManager** för att orkestrera konversationer
- **Strukturerat debattflöde** med uttalanden, jämförelser och rekommendationer
- **Kostnadsbesparande funktioner** för att minska API-användning

#### Dokumentation
- **README.md**: Grundläggande projektöversikt
- **DOCUMENTATION.md**: Detaljerad teknisk dokumentation
- **COST_SAVING_GUIDE.md**: Guide för att optimera API-användning och minska kostnader

#### Installation

```bash
pip install -r Assignment_2/requirements.txt
```

#### Konfiguration
Kopiera `.env-example` till `.env` och lägg till din OpenAI API-nyckel:

```
OPENAI_API_KEY=your_api_key_here
```

#### Användning
Huvudskriptet kan köras med:

```bash
python assignment2.py
```

För fler detaljer, se dokumentationen i respektive Assignment-mapp.