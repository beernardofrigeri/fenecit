# PayAI 🤖💳

Sistema inteligente de acessibilidade para leitura de valores e QR Codes em maquininhas de cartão, desenvolvido em Python.

O PayAI utiliza visão computacional, OCR e síntese de voz para auxiliar pessoas com deficiência visual durante pagamentos digitais.

---

## ✨ Funcionalidades

- 💰 Reconhecimento automático de valores monetários
- 🔍 Detecção de QR Codes em tempo real
- 🔊 Feedback por voz instantâneo
- 🌎 Suporte multilíngue
  - Português (PT-BR)
  - Espanhol (ES-CO)
- 🎥 Processamento em tempo real via câmera
- 🧠 Sistema inteligente anti-repetição
- 📊 Estatísticas de detecção
- 🖼️ Interface moderna e acessível
- 📸 Captura de screenshots
- ♻️ Gerenciamento automático de memória

---

## 🛠️ Tecnologias Utilizadas

- Python
- OpenCV
- EasyOCR
- PyTTSx3
- Pillow (PIL)
- NumPy
- Threading

---

## 📦 Instalação

Clone o repositório:

```bash
git clone https://github.com/beernardofrigeri/fenecit.git
cd fenecit
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o sistema:

```bash
python PayAI.py
```

---

## ⌨️ Atalhos do Sistema

| Tecla | Função |
|------|------|
| `A` | Modo automático |
| `V` | Modo leitura de valores |
| `Q` | Modo QR Code |
| `I` | Alternar idioma |
| `R` | Repetir última leitura |
| `S` | Salvar screenshot |
| `ESC` | Encerrar sistema |

---

## 📁 Estrutura do Projeto

```text
fenecit/
│
├── PayAI.py
├── requirements.txt
├── README.md
├── .gitignore
├── config.json
│
├── logs/
├── screenshots/
└── .venv/
```

---

## 🎯 Objetivo

O projeto foi desenvolvido com foco em acessibilidade e inclusão digital, buscando facilitar o uso de maquininhas de pagamento para pessoas com deficiência visual.

---

## 👨‍💻 Autor

Desenvolvido por Bernardo Girardi Frigeri 🚀