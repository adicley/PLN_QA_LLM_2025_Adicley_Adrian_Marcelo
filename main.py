from docx import Document
from pypdf import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


def extract_docx_data(doc):
    docx_str = ""
    for index, table in enumerate(doc.tables):
        rows = table.rows
        cells = rows[0].cells
        size_cells = len(cells)

        if (size_cells <= 2):
           docx_str += f"\nTable: {cells[0].text}\n"
           continue

        for row in rows:
            docx_str += str([cell.text for cell in row.cells]) + "\n"
    return docx_str


def extract_pdf_data(doc):
    pdf_str = ""
    for page in doc.pages:
        pdf_str += page.extract_text()
    return pdf_str


def show_model_resp(pipe, question, context, array_resp):
    resp = pipe(question=question, context=context)
    res = resp['answer']
    score = resp['score']

    obj_resp = {'question': question, 'answer': res}
    array_resp.append(obj_resp)
    print(f'Question: {question}')
    print(f'- Answer: {repr(res)} ({score:.2f})')
    print()


def find_cossine_similarity(sentence_transformer_model, expected_resp, model_resp):
    embedding_expected_resp = sentence_transformer_model.encode(expected_resp, convert_to_tensor=True)
    embedding_model_resp = sentence_transformer_model.encode(model_resp, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_expected_resp, embedding_model_resp).item()


def show_resp(sentence_transformer_model, expected_resp, model_resp):
    cos_sim = find_cossine_similarity(sentence_transformer_model, expected_resp, model_resp)
    return (cos_sim, expected_resp)


def display_results(document_path, question_list, model_resp_list, expected_resp_list, sentence_transformer_model):
    print(f'Document: {document_path}\n')

    for j in range(len(question_list)):
        print(f'Question: {question_list[j]}')
        print(f'Model response: {repr(model_resp_list[j]["answer"])}\n')

        results = []
        for expected_resp in expected_resp_list[j]:
            results.append(show_resp(sentence_transformer_model, expected_resp, model_resp_list[j]['answer']))

        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        for cos_sim, expected_resp in sorted_results:
            print(f'- Expected: {expected_resp} ({cos_sim*100:.4f}%)')

        print()


def main():
    docx_path = 'samples/DICIONARIO_DE_DADOS.docx'
    pdf_path = 'samples/doencas_respiratorias_cronicas.pdf'
    docx_document = Document(docx_path)
    pdf_document = PdfReader(pdf_path)
    docx_data = extract_docx_data(docx_document)
    pdf_data = extract_pdf_data(pdf_document)

    questions_docx = ['Qual tabela representa o indicador LFCES058?',
                    'Quais campos das tabelas se repetem com maior frequência?',
                    'Quais tabelas possuem algum elemento no campo DOMINIO?']

    questions_pdf = ['Quais são os meios de tratar uma rinite alergica?',
                    'Como corticoide pode ser utilizada e quais são suas contraindicações?',
                    'O que é tabagismo?']

    expected_resp_docx = [
        ['TB_ESTAB_BANCO'],
        ['UNIDADE_ID', 'DATA_ATU', 'DT_ATUALIZACAO', 'USUARIO', 'CO_USUARIO', 'CHKSUM', 'STATUS', 'STATUSMOV', 'DT_ATUALIZACAO_ORIGEM', 'DT_CMTP_INICIO', 'DT_CMTP_FIM', 'NU_SEQ_PROCESSO'],
        ['FCESGEST', 'TB_ESTABELECIMENTO', 'TB_SERVICO_REFERENCIADO', 'TB_COLETA_SELETIVA_REJEITO', 'TB_SERVICO_APOIO', 'TB_ATIVIDADE_PROFISSIONAL', 'TB_CBO']
    ]

    expected_resp_pdf = [
        ['beta-agonista', 'alérgenos', 'anti-histamínico', 'corticoide', 'broncodilatadores'],
        ['imunossupressor', 'dexametasona', 'gotas', 'injeções', 'intranasais', 'intramuscular', 'intranasal', 'perfuração', 'via oral', 'sedação', 'irritação', 'sangramento'],
        ['nicotina', 'dependência quimica à droga nicotina', 'doença crônica']
    ]

    models = ['eraldoluis/faquad-bert-base-portuguese-cased',
            'pierreguillou/bert-base-cased-squad-v1.1-portuguese',
            'timpal0l/mdeberta-v3-base-squad2']

    relations_tables = docx_data.split("\n\n")

    tables = ""
    for item in relations_tables:
        if "Table" in item:
            tables += item + "\n\n"

    models_resp = []


    for i, model in enumerate(models):
        obj = {'model_name': model, 'docx_resp': [], 'pdf_resp': []}
        models_resp.append(obj)

        pipe = pipeline('question-answering', model=model, tokenizer=model)

        print(f'Model: {model}\n')

        print(f"Document: {docx_path}\n")
        context = relations_tables[0]
        array_resp = models_resp[i]['docx_resp']
        show_model_resp(pipe, questions_docx[0], context, array_resp)
        context = tables
        for question in questions_docx[1:]:
            show_model_resp(pipe, question, context, array_resp)

        print()

        print(f"Document: {pdf_path}\n")
        context = pdf_data
        array_resp = models_resp[i]['pdf_resp']
        for question in questions_pdf:
            show_model_resp(pipe, question, context, array_resp)

        print()



    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i in range(len(models)):
        print(f'Model Name: {models[i]}')

        display_results(docx_path, questions_docx, models_resp[i]['docx_resp'], expected_resp_docx, sentence_transformer_model)
        display_results(pdf_path, questions_pdf, models_resp[i]['pdf_resp'], expected_resp_pdf, sentence_transformer_model)

    print("--- Model Performance Summary ---")

    for i in range(len(models)):
        print(f'\nModel Name: {models[i]}')
        total_similarity = 0
        num_questions = 0

        print(f"\nDocument: {docx_path}")
        for j in range(len(questions_docx)):
            num_questions += 1
            results = []
            for expected_resp in expected_resp_docx[j]:
                results.append(find_cossine_similarity(sentence_transformer_model, expected_resp, models_resp[i]['docx_resp'][j]['answer']))
            max_similarity = max(results) if results else 0
            total_similarity += max_similarity
            print(f"  Question: {questions_docx[j]} - Max Cosine Similarity: {max_similarity:.4f}")

        print(f"\nDocument: {pdf_path}")
        for j in range(len(questions_pdf)):
            num_questions += 1
            results = []
            for expected_resp in expected_resp_pdf[j]:
                results.append(find_cossine_similarity(sentence_transformer_model, expected_resp, models_resp[i]['pdf_resp'][j]['answer']))
            max_similarity = max(results) if results else 0
            total_similarity += max_similarity
            print(f"  Question: {questions_pdf[j]} - Max Cosine Similarity: {max_similarity:.4f}")

        average_similarity = total_similarity / num_questions if num_questions > 0 else 0
        models_resp[i]['average_similarity'] = average_similarity
        print(f"\nAverage Cosine Similarity for {models[i]}: {average_similarity:.4f}")

    best_model = None
    highest_average_similarity = -1

    for model_info in models_resp:
        if model_info['average_similarity'] > highest_average_similarity:
            highest_average_similarity = model_info['average_similarity']
            best_model = model_info['model_name']

    print("\n--- Best Model ---")
    print(f"The model with the highest average cosine similarity is: {best_model} ({highest_average_similarity:.4f})")


if __name__ == "__main__":
    main()
