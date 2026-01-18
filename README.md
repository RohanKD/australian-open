# australian-open

## Data setup

Use the helper script to download recent ATP match files from the Jeff Sackmann
repository and build `australian_open_history.csv` for the model scripts.

```bash
python download_atp_data.py --start-year 2005 --end-year 2025
```

This will create `data/raw/atp_matches_20xx.csv` files and a combined
`australian_open_history.csv` in the repo root.
