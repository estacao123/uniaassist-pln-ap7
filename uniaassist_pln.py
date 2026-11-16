"""AP7 UniAssist: classificador de mensagens. Complete os TODOs sem consultar gabarito."""
import csv
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PASTA = Path(__file__).resolve().parent


def ler_dataset(arquivo):
    with (PASTA / arquivo).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classificar_regras(texto):
    """Baseline; devolva uma das quatro classes ou 'Revisão humana'."""
    regras = {
        "Financeiro": ["boleto", "mensalidade", "cobrança", "pagamento", "fatura"],
        "Matrícula": ["matrícula", "rematrícula", "disciplina", "turma", "trancamento"],
        "Estágio": ["estágio", "termo", "supervisor", "concedente", "convênio"],
        "Suporte": ["senha", "login", "portal", "acesso", "aplicativo", "erro"],
    }
    # TODO 1: normalize com .casefold(), conte os termos encontrados por categoria;
    # se houver empate ou nenhuma correspondência, devolva 'Revisão humana'.
    texto_norm = texto.casefold()
    pontuacao = {}

    for categoria, termos in regras.items():
        pontuacao[categoria] = sum(1 for termo in termos if termo.casefold() in texto_norm)

    max_pontos = max(pontuacao.values())
    if max_pontos == 0:
        return "Revisão humana"

    vencedores = [cat for cat, pts in pontuacao.items() if pts == max_pontos]
    if len(vencedores) > 1:
        return "Revisão humana"

    return vencedores[0]


def main():
    registros = ler_dataset("mensagens.csv")
    textos = [r["texto"] for r in registros]
    rotulos = [r["categoria"] for r in registros]

    print(f"Total de mensagens: {len(registros)}")
    print(f"Distribuição por categoria: {dict(Counter(rotulos))}\n")

    # TODO 2: crie divisão treino/teste com test_size=0.25,
    # random_state=42 e stratify=rotulos.
    X_train, X_test, y_train, y_test = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos
    )

    # TODO 3: crie Pipeline com TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    # e LinearSVC(random_state=42); ajuste com .fit no treino.
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(random_state=42)),
    ])
    modelo.fit(X_train, y_train)

    # TODO 4: preveja o teste; imprima acurácia, classification_report,
    # e matriz de confusão com labels=sorted(set(rotulos)).
    labels = sorted(set(rotulos))
    y_pred_ml = modelo.predict(X_test)

    print("=== AVALIAÇÃO: TF-IDF (1, 2) + LinearSVC ===")
    print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_ml):.4f}\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred_ml, zero_division=0))
    print(f"Ordem das classes na matriz: {labels}")
    print("Matriz de Confusão (ML):")
    print(confusion_matrix(y_test, y_pred_ml, labels=labels))
    print()

    # TODO 5: compare previsão por regras no MESMO teste.
    y_pred_regras = [classificar_regras(t) for t in X_test]
    acuracia_regras = accuracy_score(y_test, y_pred_regras)
    revisao_humana_qtd = sum(1 for p in y_pred_regras if p == "Revisão humana")

    print("=== AVALIAÇÃO: BASELINE POR REGRAS ===")
    print(f"Acurácia no teste: {acuracia_regras:.4f} (Revisão humana: {revisao_humana_qtd}/{len(y_test)})")
    print(classification_report(y_test, y_pred_regras, zero_division=0))
    print()

    # Experimento item 7 do LEIA_ME.md: ngram_range=(1, 1)
    modelo_unigrama = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
        ("clf", LinearSVC(random_state=42)),
    ])
    modelo_unigrama.fit(X_train, y_train)
    y_pred_uni = modelo_unigrama.predict(X_test)

    print("=== EXPERIMENTO: UNIGRAMAS ngram_range=(1, 1) ===")
    print(f"Acurácia no teste (1, 1): {accuracy_score(y_test, y_pred_uni):.4f}\n")
    print(classification_report(y_test, y_pred_uni, zero_division=0))
    print()

    # TODO 6: classifique desafios.csv com ambos, imprima resultados e
    # registre análise no parecer.md.
    desafios = ler_dataset("desafios.csv")
    textos_desafios = [d["texto"] for d in desafios]
    preds_desafios_ml = modelo.predict(textos_desafios)
    scores_desafios = modelo.decision_function(textos_desafios)

    print("=== CLASSIFICAÇÃO: DESAFIOS.CSV ===")
    for i, item in enumerate(desafios):
        id_msg = item.get("id", i + 1)
        texto = item["texto"]
        pred_r = classificar_regras(texto)
        pred_m = preds_desafios_ml[i]
        margem = max(scores_desafios[i])
        print(f"ID {id_msg} | Regras: {pred_r:<15} | ML: {pred_m:<11} (margem: {margem:.3f})")
        print(f"   Texto: {texto}")


if __name__ == "__main__":
    main()