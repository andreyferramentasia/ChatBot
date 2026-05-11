import streamlit as st
from transformers import pipeline

if "pipe" not in st.session_state:
    st.session_state.pipe = pipeline(model="facebook/bart-large-mnli")

pipe = st.session_state.pipe

st.title("Assistente Virtual de calouros")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá, seja bem-vindo, Como posso ajudar-lo hoje?"}
    ]

prompt = st.chat_input("Pergunte alguma coisa")

with st.container(border=True):
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        resultado = pipe(prompt, candidate_labels=["Matricula", "secretaria", "coordenação", "notas"])
        categoria = resultado["labels"][0]
        score = resultado["scores"][0]

        respostas = {
            "notas": "Para ver suas notas acesse o portal do aluno.",
            "Matricula": "Para matrícula procure a secretaria no bloco X.",
            "secretaria": "A secretaria funciona de segunda a sexta das 8h às 18h.",
            "coordenação": "Para falar com a coordenação, envie um e-mail ou vá ao bloco X.",
        }

        resposta = respostas.get(categoria, "Não entendi, pode reformular?")
        st.session_state.messages.append({"role": "assistant", "content": f"{resposta} (Confiança: {score:.2f})"})
        st.chat_message("assistant").write(f"{resposta} (Confiança: {score:.2f})")

