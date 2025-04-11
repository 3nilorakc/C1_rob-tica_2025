# 📚 IDENTIFICADOR DE GÊNERO LITERÁRIO v1.0 📚

## *** LEIA-ME PRIMEIRO!!! ***

Desenvolvido por: [Karoline Lemos Costa]
Data: Abril 2025
Versão: 1.0

## DESCRIÇÃO DO SISTEMA
======================

Este programa REVOLUCIONÁRIO utiliza a mais avançada tecnologia de INTELIGÊNCIA ARTIFICIAL para identificar o gênero literário de livros através de suas capas!!! Sim, você leu certo! Basta carregar a imagem da capa e o sistema fará o resto!

## RECURSOS INCRÍVEIS!!!
=====================

* Análise de imagens em TEMPO REAL
* Interface amigável e SUPER intuitiva
* Resultados com as 3 MELHORES previsões
* Tecnologia TensorFlow.js de ÚLTIMA GERAÇÃO

## REQUISITOS DO SISTEMA
=====================

* Navegador web MODERNO (Chrome, Firefox, Edge, etc.)
* Conexão com a Internet para carregar as bibliotecas
* Arquivos model.json e metadata.json (NÃO INCLUÍDOS - devem ser gerados separadamente)

## COMO INSTALAR
=============

1. COPIE todos os arquivos para um diretório em seu servidor web:
   - index.html
   - style.css
   - script.js
   - model.json (deve ser gerado separadamente)
   - metadata.json (deve ser gerado separadamente)

2. ACESSE o aplicativo através do seu navegador favorito!

## COMO USAR
=========

1. CLIQUE no botão para selecionar uma imagem de capa de livro
2. VISUALIZE a prévia da imagem selecionada
3. PRESSIONE o botão "Prever Gênero"
4. MARAVILHE-SE com os resultados!!!

## DETALHES TÉCNICOS
=================

Este sistema utiliza um modelo de rede neural convolucional (CNN) pré-treinado para classificar imagens de capas de livros em diferentes gêneros literários. O modelo é carregado via TensorFlow.js e executa inferências diretamente no navegador do usuário!

```
ATENÇÃO: O modelo deve ser treinado separadamente e exportado nos formatos model.json e metadata.json.
```

## ARQUIVOS DO SISTEMA
==================

* index.html - Interface principal do programa
* style.css - Estilos visuais INCRÍVEIS
* script.js - Código JavaScript que faz a MÁGICA acontecer
* model.json - Modelo de IA pré-treinado (não incluído)
* metadata.json - Metadados com os rótulos dos gêneros (não incluído)

## COMO FUNCIONA
============

1. O sistema carrega o modelo TensorFlow.js e os metadados
2. A imagem da capa é redimensionada para 224x224 pixels
3. A imagem é convertida em um tensor e processada pelo modelo
4. As previsões são ordenadas e as 3 melhores são exibidas

## PROBLEMAS CONHECIDOS
===================

* O modelo precisa ser carregado antes de fazer previsões
* Algumas capas muito abstratas podem gerar resultados imprecisos
* O sistema requer conexão com a internet para carregar o TensorFlow.js

## AGRADECIMENTOS
=============

OBRIGADO por escolher o IDENTIFICADOR DE GÊNERO LITERÁRIO v1.0!!!

© 2025 - TODOS OS DIREITOS RESERVADOS
