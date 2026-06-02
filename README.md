`sudo apt install libfftw3-dev libfftw3-double3 libfftw3-single3`
`sudo apt-get install python3-dev`
`pip install -r requirements.txt`
`pip install Cython`
`pip install --no-build-isolation fitsne`

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
