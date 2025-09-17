# Portfolio Signals (Guru‑mix, free data)

Tento malý projekt každý den stáhne ceny z Yahoo Finance, spočítá indikátory a „guru‑mix“ skóre (Buffett/Graham/Greenblatt/Raschke/Lynch/Icahn/Ackman) a vygeneruje semafor **BUY / HOLD / TRIM** pro tvé tickery.

## Co upravit
- `config.yaml` – seznam tickerů (portfolio, watchlist) a limity.
- `templates/report.html` – vzhled reportu.

## Jak spustit
1) Nahraj repo na GitHub.  
2) Zkontroluj záložku **Actions** – workflow `daily-signals` vygeneruje `report/report.html` a `report/report.csv`.  
3) (Volitelné) zapni GitHub Pages pro složku `/report` a budeš mít veřejný odkaz na HTML report.

Všechna data jsou zdarma z Yahoo Finance přes knihovnu `yfinance`. Pokud některé fundamentální položky chybí, skóre se částečně opře o technická pravidla.
