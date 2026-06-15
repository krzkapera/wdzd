# Wykrywanie nietypowych lotów (OpenSky)

## Uruchomienie

1. **Środowisko** (zalecane `uv`, Python 3.12):
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv -r requirements.txt
   ```
   Alternatywnie: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

2. **Dane** — pobierz godzinną próbkę OpenSky (plik jest w `.gitignore`):
   ```bash
   curl -L -o data/states.json.tar \
     https://opensky-network.org/datasets/states/2022-06-27/23/states_2022-06-27-23.json.tar
   tar -xf data/states.json.tar -C data states_2022-06-27-23.json.gz
   gunzip data/states_2022-06-27-23.json.gz
   ```

3. **Notebook** — otwórz `src/lol.ipynb` (kernel *Python (wdzd)*) lub uruchom z linii poleceń:
   ```bash
   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.kernel_name=wdzd-venv src/lol.ipynb
   ```
   Wykonanie eksportuje interaktywne wykresy do `presentation/figures/`.
   Notebook można odtworzyć z `src/_build_notebook.py`.

4. **Prezentacja** — otwórz `presentation/index.html` w przeglądarce (interaktywne slajdy reveal.js).

### FIt-SNE (opcjonalnie)
Notebook domyślnie używa `openTSNE`. Aby włączyć FIt-SNE:
```bash
sudo apt install libfftw3-dev libfftw3-double3 libfftw3-single3 python3-dev
pip install Cython && pip install --no-build-isolation fitsne
```


### Cel projektu
Celem projektu jest identyfikacja nietypowych lotów na podstawie parametrów trajektorii, takich jak wysokość, prędkość, zmiany kursu i długość trasy.

### Zakres projektu
Należy przygotować reprezentację lotów jako wektorów cech i przeprowadzić analizę podobieństwa oraz anomalii.

### Wymagania

- przygotowanie cech opisujących trajektorie,
- wizualizacja lotów w przestrzeni 2D,
- wykrycie obserwacji odstających,
- porównanie różnych metod redukcji wymiaru,
- przygotowanie dashboardu do eksploracji lotów,
- interpretacja wykrytych przypadków nietypowych.
- Element związany z redukcją wymiaru:
- Wymagane jest użycie co najmniej dwóch metod spośród: PCA, UMAP, PaCMAP, TriMAP, FIt-SNE/flt-SNE.

### Element rozszerzony
Można dodać klastrowanie trajektorii lub klasyfikację typów lotów.

### Przykładowe dane i narzędzia
OpenSky Network, Python, Dash, Altair.
