# Naive Bayes com k-Means  - Aprendizagem de Máquina

O Objetivo do projeto é a implementação do classificador Bayesiano em uma base de dados clusterizada com o k-Means, bem como sua comparação com o 1-NN e o Naive Bayes sem este procedimento inicial. As duas bases de dados foram retiradas do repositório [Promise](http://promise.site.uottawa.ca/SERepository/datasets-page.html).

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