# Projeto de Mineração de Dados: Clusterização de Textos com Python

Este projeto é uma atividade acadêmica para a disciplina de Mineração de Dados.

O objetivo é aplicar o algoritmo de clusterização **K-Means** para agrupar automaticamente uma coleção de 2500 artigos de notícias (a base de dados `Reuters C50`).

O script `clusterizar.py` utiliza a biblioteca **Scikit-learn** para:
1.  Carregar os arquivos de texto.
2.  Converter os textos em vetores numéricos usando **TF-IDF**.
3.  Aplicar o algoritmo K-Means para agrupar os textos em 10 clusters (tópicos).
4.  Exibir os 10 termos mais importantes de cada cluster para permitir a análise humana.

## 🚀 Resultados

O algoritmo foi capaz de identificar com sucesso 10 tópicos distintos nos dados, incluindo:
* Finanças (mercado britânico e canadense)
* Indústria da Aviação (Boeing vs. Airbus)
* Indústria de Tecnologia (Microsoft, Apple, IBM)
* Política (China e Hong Kong)
* Setor Automotivo (Greves da GM)
* E outros.