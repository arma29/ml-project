# Projeto - Aprendizagem de Máquina

O Objetivo do projeto é a implementação do artigo: Pasi Fränti, Sami Sieranoja. **How much can k-means be improved by using better initialization and repeats?** Pattern Recognition, Volume 93, September 2019, Pages 95-112.

### Requisitos

- pip
- virtualenv
- pacotes texlive
  - sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super


### Passos

Caso não possua o ambiente virtual:

```
python3 -m venv <virtual_env_name>
```

Ativação:

```
source <virtual_env>/bin/activate
```

Instalação dos pacotes:

```
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Execução:

```
python src/main.py
```