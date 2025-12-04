# Simulador de Comunicação Digital (BPSK / QPSK + AWGN)

Este projeto implementa, em Python, um **sistema completo de comunicação digital** para fins didáticos na disciplina de Redes de Computadores.  

O simulador inclui:

- Geração de mensagem ASCII e conversão para bits;
- Codificação de linha (NRZ unipolar, Manchester, AMI) para demonstração;
- Codificação de canal simples por **código de repetição (3,1)**;
- Modulações digitais **BPSK** e **QPSK** (com mapeamento Gray);
- Canal ruidoso **AWGN** com controle de **Eb/N0 (dB)**;
- Demodulação, decodificação e cálculo da **taxa de erro de bits (BER)**;
- Geração de gráficos:
  - Formas de onda das codificações de linha;
  - Curvas **BER × Eb/N0** para cada esquema testado.

---

## 1. Requisitos

- Python 3.x (testado com Python 3.10+)
- Bibliotecas Python:
  - `numpy`
  - `matplotlib`

Instalação das dependências:

```bash
pip install numpy matplotlib
