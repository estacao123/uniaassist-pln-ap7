# Parecer técnico — AP7

## Resultados

| Método | Acurácia no teste | Observações |
|---|---:|---|
| Regras | 80,0% (0.8000) | Alta precisão (1.00) quando encontra palavras-chave, mas encaminhou 4/20 (20%) mensagens do teste para `Revisão humana` por falta de correspondência. |
| TF-IDF + LinearSVC | 70,0% (0.7000) | Classificou 100% das mensagens sem abstenção, mas sofreu perda de desempenho na classe `Financeiro` (F1-score 0.44) devido ao tamanho reduzido da base de treino (60 exemplos). |
## Matriz de confusão

```text
Ordem das classes: ['Estágio', 'Financeiro', 'Matrícula', 'Suporte']

[[5 0 0 0]
 [1 2 1 1]
 [0 2 3 0]
 [1 0 0 4]]
```

## Mensagens ambíguas (pelo menos 3)

| ID | Regras | Modelo | O que dificultou? | Política proposta |
|---|---|---|---|---|
| **C01** (*"Não posso acessar o boleto porque esqueci minha senha"*) | Revisão humana | Suporte (margem: -0.227) | Conflito de dois setores: "boleto" (Financeiro) vs. "acessar" e "senha" (Suporte), gerando empate nas regras e margem negativa (baixa confiança) no SVM. | Encaminhar automaticamente para `Revisão humana` ou priorizar triagem de **Suporte** (causa bloqueante) quando houver choque de regras ou score SVM abaixo do limiar. |
| **C02** (*"Preciso quitar a parcela do semestre"*) | Revisão humana | Financeiro (margem: -0.026) | Uso de sinônimos fora do dicionário fixo ("quitar", "parcela" em vez de "pagar", "boleto/mensalidade"), causando falha total de cobertura nas regras. | Utilizar o modelo de ML como apoio para capturar variações semânticas, mas exigir validação humana quando a margem de decisão for próxima de zero. |
| **C05** (*"O sistema caiu ao enviar o relatório de estágio"*) | Estágio | Estágio (margem: 0.423) | Ambiguidade entre problema técnico ("sistema caiu" -> Suporte) e contexto da tarefa ("relatório de estágio" -> Estágio); ambos os métodos ignoraram o incidente de TI por ausência da palavra-chave exata. | Implementar fila híbrida de dupla etiqueta ou rotear incidentes de indisponibilidade ("sistema caiu") primeiro para o **Suporte** antes da análise acadêmica. |
| **C06** (*"Paguei a taxa mas a rematrícula continua pendente"*) | Matrícula | Matrícula (margem: 0.200) | Dependência entre processos: o verbo "paguei" remete ao Financeiro, enquanto "rematrícula" aponta para Matrícula; a flexão verbal ("paguei") não ativou a regra ("pagamento"). | Acionar fluxo integrado entre **Financeiro** (compensação bancária) e **Matrícula** sempre que houver termos de pagamento associados à liberação de vínculo. |
| **C08** (*"Preciso saber onde registrar as horas da empresa"*) | Revisão humana | Matrícula (margem: -0.328) | Vocabulário implícito ("horas da empresa" refere-se a **Estágio**, sem citar o termo literal), levando as regras à abstenção e induzindo o ML a erro com margem altamente negativa (-0.328). | Definir limiar mínimo de confiança no `LinearSVC` (ex.: `score < 0.0`): toda predição com margem negativa deve ser bloqueada e enviada à `Revisão humana`. |

## Experimento com unigramas

Ao alterar `ngram_range=(1, 2)` para `(1, 1)`, a acurácia global no teste manteve-se em **70,0% (0.7000)**, porém houve redistribuição entre as classes: `Financeiro` subiu de F1-score `0.44` para `0.60`, enquanto `Estágio` (`0.83` para `0.80`) e `Matrícula` (`0.67` para `0.60`) recuaram. Metodologicamente, não se deve escolher um modelo vencedor ajustando hiperparâmetros diretamente sobre o conjunto de teste: como o teste possui apenas 20 mensagens, comparações repetidas provocam vazamento indireto de informação (*overfitting* ao teste), comprometendo a estimativa real de generalização — o correto seria usar validação cruzada (*cross-validation*) nos 60 exemplos de treino.

## Parecer final (até 15 linhas)

A comparação demonstra que o baseline por regras atinge maior acurácia aparente (80%) por operar com precisão máxima em termos exatos, mas depende de manutenção manual constante e falha diante de sinônimos (como em C02 e C08), devolvendo 20% do volume de teste para atendimento humano. Em contrapartida, o pipeline `TF-IDF + LinearSVC` generaliza variações vocabulares sem dicionários rígidos, porém sofre com o tamanho reduzido da base de treinamento (60 registros) e possui uma limitação crítica em produção: o classificador sempre força uma das quatro categorias, mesmo quando a margem de decisão é negativa (baixa confiança) ou quando a mensagem envolve dois setores simultaneamente (como C01 e C06). Portanto, nenhum dos métodos deve operar isoladamente de forma totalmente autônoma. Recomenda-se uma arquitetura híbrida em cascata: utilizar regras de conflito para detectar múltiplas intenções e aplicar o modelo de Machine Learning com um corte mínimo de confiança (`decision_function > 0`), encaminhando automaticamente os casos ambíguos ou de baixa margem para a fila de `Revisão humana`, cujos novos rótulos retroalimentarão o treinamento periódico do modelo.