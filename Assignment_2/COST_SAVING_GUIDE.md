# Kostnadsbesparing för Multi-Agent Debatt

Den här guiden förklarar hur man använder Multi-Agent systemet på ett kostnadseffektivt sätt så att du kan minimera dina utgifter för OpenAI API:n.

## Kostnader för OpenAI API

OpenAI:s API prissättning baseras på tokens (ungefär 4 tecken = 1 token):

| Modell | Kostnad per 1K tokens (input) | Kostnad per 1K tokens (output) |
|--------|-------------------------------|--------------------------------|
| GPT-3.5-Turbo | $0.0010 (~0,01 SEK) | $0.0020 (~0,02 SEK) |
| GPT-4 | $0.03 (~0,30 SEK) | $0.06 (~0,60 SEK) |

Som du kan se är GPT-4 ungefär 30 gånger dyrare än GPT-3.5-Turbo.

## Kostnadsbesparande funktioner

Vi har implementerat flera funktioner för att minska API-kostnaderna:

1. **Budget-läge**: Använder färre agenter och ämnen (aktiveras som standard)
2. **Modellval**: Använder billigare GPT-3.5-Turbo för de flesta agenter
3. **Begränsad kontextfönster**: Sparar bara de senaste utbytena i budget-läge
4. **Kortare svar**: Begränsar svarlängden till max 400 tokens
5. **Selektiv GPT-4**: Använder GPT-4 endast för moderatorn och ML-experten

## Uppskattade kostnader

| Konfiguration | Uppskattad kostnad |
|---------------|---------------------|
| Mini-debatt test | ~$0.05-0.10 (0,50-1 SEK) |
| Budget-läge full debatt | ~$0.50-1.50 (5-15 SEK) |
| Komplett full debatt | ~$5.00-15.00 (50-150 SEK) |

## Hur man kör systemet kostnadseffektivt

### 1. Använd testskriptet först

```bash
python test_agent_system.py
```

Detta skript testar endast API-anslutningen och en enkel agent-respons, vilket kostar mycket mindre än att köra en full debatt.

### 2. Kör debatt i budget-läge

När du kör huvudprogrammet, välj budget-läge när du blir tillfrågad:

```bash
python assignment2.py
```

När du får frågan "Do you want to run in budget-saving mode? (y/n, default=y):", tryck Enter eller svara "y".

### 3. Anpassa inställningarna (för avancerade användare)

Om du vill justera kostnadsbesparingsinställningarna, redigera konstanterna i början av `assignment2.py`:

```python
# Cost saving settings
DEFAULT_MODEL = "gpt-3.5-turbo"  # Använd billigare modell som standard
MAX_TOKENS = 400  # Begränsa token-användning
TEMPERATURE = 0.7
USE_CHEAPER_MODEL = True  # Sätt till False om du vill använda GPT-4 för alla svar
```

## Rekommendation för uppgiften

För uppgiftspresentationen rekommenderar vi:

1. Kör i budget-läge när du utvecklar och testar systemet
2. Gör ett sista körning med full kvalitet för din slutgiltiga inlämning

## Andra sätt att minska kostnaderna

- **Testläge för OpenAI**: Använd GPT-3.5-Turbo för all utveckling och testning
- **Lokal LLM**: Använd en lokal modell som Llama för testning utan API-kostnader
- **Cache-lagra svar**: Implementera cachelagring för att undvika att generera samma svar flera gånger

Vi hoppas att dessa kostnadsbesparande funktioner hjälper dig att genomföra uppgiften utan att överskrida din budget! 