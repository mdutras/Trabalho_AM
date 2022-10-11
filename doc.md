Tema: Classificação de livros utilizando a sinopse dos livros

# Passo a passo
1. Inicialmente, todos os documentos de uma base de dados usada em um sistema de Classificação de Documentos são escritos em linguagem natural.
2. Essa base de dados em linguagem natural (chamada de Corpus) é dividida em conjunto de treinamento e conjunto de teste, o primeiro será utilizado para treinar o classificador e o segundo será utilizado para avaliar o classificado.
3. A primeira etapa é tratar esses documentos por diversas rotinas de pré-processamento
4. Após o pré-processamento, é necessário modificar a representação original (linguagem natural) para alguma representação aplicável em Aprendizagem de Máquina, sendo a abordagem mais comum a geração de um vetor de características.
5. Por fim, essa representação é armazenada numa memória principal ou secundária, para ser usada na construção do classificador.

**Resumidamente**
1. Pré-processamento
2. Vetor de características

# Pré-processamento
O pré-processamento contém as primeiras operações realizadas sobre o Corpus, sendo executado antes de qualquer etapa. Seu objetivo principal é organizar as informações presentes nos documentos, preparando-os para a geração do vetor de características

**Operações realizadas nessa etapa:**
- Deixar todos os caracteres minúsculos.
- Remoção de qualquer valor que não seja letra.
- Remoção de acentos.
- Remoção de elementos que não sejam o radical.
- Remoção de pontuação.
- Remoção de palavras que não possuam muito significado (pronomes, artigos, conectivos, conjuções, preposições)

# Vetor de características


--- REFERÊNCIAS ---
# Bag of words
https://machinelearningmastery.com/gentle-introduction-bag-words-model/
