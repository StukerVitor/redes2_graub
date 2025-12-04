import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Utilidades de conversão ASCII <-> bits
# =========================

def ascii_to_bits(text):
    """Converte uma string ASCII em um array de bits (0/1)."""
    bits = []
    for ch in text:
        b = format(ord(ch), '08b')  # 8 bits por caractere
        bits.extend(int(bit) for bit in b)
    return np.array(bits, dtype=int)


def bits_to_ascii(bits):
    """Converte um array de bits (0/1) em string ASCII (múltiplo de 8 bits)."""
    bits_str = ''.join(str(int(b)) for b in bits)
    chars = []
    for i in range(0, len(bits_str), 8):
        byte = bits_str[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)


# =========================
# 2. Codificação de linha (AMI / Manchester) – exemplos
# =========================

def manchester_encode(bits):
    """
    Codificação Manchester clássica:
    - bit 1 -> [1, -1]
    - bit 0 -> [-1, 1]
    Retorna sequência de +1/-1 (dois chips por bit).
    """
    encoded = np.empty(len(bits) * 2, dtype=float)
    idx = 0
    for b in bits:
        if b == 1:
            encoded[idx] = 1.0
            encoded[idx + 1] = -1.0
        else:
            encoded[idx] = -1.0
            encoded[idx + 1] = 1.0
        idx += 2
    return encoded


def manchester_decode(chips):
    """
    Decodificação simples de Manchester:
    - se padrão ~[+, -] -> 1
    - se padrão ~[-, +] -> 0
    """
    assert len(chips) % 2 == 0, "Sequência Manchester deve ter comprimento par"
    bits_hat = []
    for i in range(0, len(chips), 2):
        first, second = chips[i], chips[i + 1]
        if first > 0 and second < 0:
            bits_hat.append(1)
        else:
            bits_hat.append(0)
    return np.array(bits_hat, dtype=int)


def ami_encode(bits):
    """
    Codificação AMI (Alternate Mark Inversion):
    - bit 0 -> 0
    - bit 1 -> alterna entre +1 e -1
    """
    encoded = np.zeros(len(bits), dtype=float)
    last_level = -1.0
    for i, b in enumerate(bits):
        if b == 1:
            last_level *= -1.0  # alterna sinal
            encoded[i] = last_level
        else:
            encoded[i] = 0.0
    return encoded


def ami_decode(levels, threshold=0.5):
    """
    Decodificação simples de AMI:
    - |amplitude| > limiar -> bit 1
    - |amplitude| <= limiar -> bit 0
    """
    bits_hat = np.zeros(len(levels), dtype=int)
    for i, lev in enumerate(levels):
        if abs(lev) > threshold:
            bits_hat[i] = 1
        else:
            bits_hat[i] = 0
    return bits_hat


# =========================
# 3. Codificação de repetição (código de canal simples)
# =========================

def repetition3_encode(bits):
    """Código de repetição (3,1): cada bit vira 3 bits iguais."""
    return np.repeat(bits, 3)


def repetition3_decode(bits):
    """Decodifica código (3,1) por votação de maioria."""
    assert len(bits) % 3 == 0, "Comprimento deve ser múltiplo de 3"
    reshaped = bits.reshape(-1, 3)
    sums = reshaped.sum(axis=1)
    return (sums >= 2).astype(int)


# =========================
# 4. Modulação / demodulação
# =========================

def bpsk_mod(bits):
    """BPSK: 0 -> -1, 1 -> +1 (energia por símbolo = 1)."""
    return 2 * bits - 1  # 0->-1, 1->+1


def bpsk_demod(symbols):
    """Demodulador BPSK por detecção de sinal."""
    return (symbols >= 0).astype(int)


def qpsk_mod(bits):
    """
    QPSK Gray-coded com mapeamento consistente com o demodulador:
      00 ->  (1 + 1j)/sqrt(2)
      10 -> (-1 + 1j)/sqrt(2)
      11 -> (-1 - 1j)/sqrt(2)
      01 ->  (1 - 1j)/sqrt(2)
    """
    # garante número par de bits (se ímpar, adiciona um 0 no fim)
    if len(bits) % 2 == 1:
        bits = np.append(bits, 0)
    bits = bits.reshape(-1, 2)
    symbols = []
    for b0, b1 in bits:
        if   b0 == 0 and b1 == 0:
            s = (1 + 1j) / np.sqrt(2)
        elif b0 == 1 and b1 == 0:
            s = (-1 + 1j) / np.sqrt(2)
        elif b0 == 1 and b1 == 1:
            s = (-1 - 1j) / np.sqrt(2)
        else:  # b0 == 0 and b1 == 1
            s = (1 - 1j) / np.sqrt(2)
        symbols.append(s)
    return np.array(symbols)


def qpsk_demod(symbols):
    """
    Demodulador QPSK coerente, usando sinais dos eixos I/Q.
    Inverso do mapeamento acima (Gray).
    """
    bits_hat = []
    for s in symbols:
        b0 = 1 if s.real < 0 else 0
        b1 = 1 if s.imag < 0 else 0
        bits_hat.extend([b0, b1])
    return np.array(bits_hat, dtype=int)


# =========================
# 5. Canal AWGN com controle de Eb/N0 (em dB)
# =========================

def add_awgn(signal, EbN0_dB, bits_per_symbol=1):
    """
    Adiciona ruído AWGN a um sinal (real ou complexo), dado Eb/N0 em dB.
    Supõe energia média do símbolo Es = 1.
    bits_per_symbol: quantidade de bits que cada símbolo carrega (1 em BPSK, 2 em QPSK).
    """
    EbN0 = 10 ** (EbN0_dB / 10.0)
    Es = 1.0
    Eb = Es / bits_per_symbol
    N0 = Eb / EbN0
    sigma2 = N0 / 2.0
    sigma = np.sqrt(sigma2)

    if np.iscomplexobj(signal):
        noise = sigma * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    else:
        noise = sigma * np.random.randn(*signal.shape)
    return signal + noise


# =========================
# 6. Simulação de BER
# =========================

def simulate_ber(EbN0_dB_range, n_bits=100000):
    """
    Simula BER para:
      - BPSK sem codificação
      - BPSK com código de repetição (3,1)
      - QPSK sem codificação
      - QPSK com código de repetição (3,1)
    """
    # mensagem binária aleatória
    bits = np.random.randint(0, 2, n_bits)

    ber_bpsk = []
    ber_bpsk_rep = []
    ber_qpsk = []
    ber_qpsk_rep = []

    for EbN0_dB in EbN0_dB_range:
        # ---- BPSK sem codificação ----
        tx_bpsk = bpsk_mod(bits)
        rx_bpsk = add_awgn(tx_bpsk, EbN0_dB, bits_per_symbol=1)
        bits_hat_bpsk = bpsk_demod(rx_bpsk)
        errors_bpsk = np.sum(bits != bits_hat_bpsk[:n_bits])
        ber_bpsk.append(errors_bpsk / n_bits)

        # ---- BPSK com código de repetição (3,1) ----
        bits_enc = repetition3_encode(bits)
        tx_bpsk_c = bpsk_mod(bits_enc)
        rx_bpsk_c = add_awgn(tx_bpsk_c, EbN0_dB, bits_per_symbol=1)
        bits_hat_c = bpsk_demod(rx_bpsk_c)
        bits_hat_c = bits_hat_c[:len(bits_enc)]  # garante mesmo tamanho
        bits_dec = repetition3_decode(bits_hat_c)
        errors_bpsk_rep = np.sum(bits != bits_dec[:n_bits])
        ber_bpsk_rep.append(errors_bpsk_rep / n_bits)

        # ---- QPSK sem codificação ----
        tx_qpsk = qpsk_mod(bits)
        rx_qpsk = add_awgn(tx_qpsk, EbN0_dB, bits_per_symbol=2)
        bits_hat_qpsk = qpsk_demod(rx_qpsk)
        bits_hat_qpsk = bits_hat_qpsk[:n_bits]
        errors_qpsk = np.sum(bits != bits_hat_qpsk)
        ber_qpsk.append(errors_qpsk / n_bits)

        # ---- QPSK com código de repetição (3,1) ----
        bits_enc2 = repetition3_encode(bits)
        tx_qpsk_c = qpsk_mod(bits_enc2)
        rx_qpsk_c = add_awgn(tx_qpsk_c, EbN0_dB, bits_per_symbol=2)
        bits_hat_qpsk_c = qpsk_demod(rx_qpsk_c)
        bits_hat_qpsk_c = bits_hat_qpsk_c[:len(bits_enc2)]
        bits_dec2 = repetition3_decode(bits_hat_qpsk_c)
        errors_qpsk_rep = np.sum(bits != bits_dec2[:n_bits])
        ber_qpsk_rep.append(errors_qpsk_rep / n_bits)

        print(f"Eb/N0 = {EbN0_dB:>4.1f} dB | "
              f"BER BPSK = {ber_bpsk[-1]:.3e}, "
              f"BPSK rep(3,1) = {ber_bpsk_rep[-1]:.3e}, "
              f"QPSK = {ber_qpsk[-1]:.3e}, "
              f"QPSK rep(3,1) = {ber_qpsk_rep[-1]:.3e}")

    return (np.array(ber_bpsk),
            np.array(ber_bpsk_rep),
            np.array(ber_qpsk),
            np.array(ber_qpsk_rep))


def plot_ber(EbN0_dB_range, ber_bpsk, ber_bpsk_rep, ber_qpsk, ber_qpsk_rep):
    """Gera gráfico BER x Eb/N0 em escala semi-log."""
    plt.figure(figsize=(8, 6))
    plt.semilogy(EbN0_dB_range, ber_bpsk, 'o-', label='BPSK sem codificação')
    plt.semilogy(EbN0_dB_range, ber_bpsk_rep, 's-', label='BPSK repetição (3,1)')
    plt.semilogy(EbN0_dB_range, ber_qpsk, 'd-', label='QPSK sem codificação')
    plt.semilogy(EbN0_dB_range, ber_qpsk_rep, 'x-', label='QPSK repetição (3,1)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Taxa de erro de bits (BER)')
    plt.title('BER x Eb/N0 para diferentes modulações e codificação de canal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ber_vs_ebn0.png', dpi=300)
    plt.show()


def demo_line_coding():
    """
    Gera um exemplo de codificação de linha (NRZ, Manchester, AMI)
    para uma mensagem curta, e plota os sinais.
    """
    msg = "REDES"
    bits = ascii_to_bits(msg)

    # NRZ unipolar: 0 -> 0, 1 -> 1
    nrz = bits.astype(float)
    manch = manchester_encode(bits)
    ami = ami_encode(bits)

    # constrói eixos de tempo (apenas para visualização qualitativa)
    t_nrz = np.arange(len(nrz))
    t_manch = np.arange(len(manch)) / 2.0  # duas amostras por bit
    t_ami = np.arange(len(ami))

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.step(t_nrz, nrz, where='post')
    plt.ylim(-0.5, 1.5)
    plt.title('Codificação NRZ unipolar (exemplo)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.step(t_manch, manch, where='post')
    plt.ylim(-1.5, 1.5)
    plt.title('Codificação Manchester (exemplo)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.step(t_ami, ami, where='post')
    plt.ylim(-1.5, 1.5)
    plt.title('Codificação AMI (exemplo)')
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo (unidades de bit)')

    plt.tight_layout()
    plt.savefig('line_coding_examples.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # Exemplo de uso completo:
    np.random.seed(42)  # reprodutibilidade

    # 1) Geração de mensagem ASCII e conversão para bits
    mensagem = "Trabalho de Redes"
    bits_msg = ascii_to_bits(mensagem)
    print("Mensagem original:", mensagem)
    print("Primeiros 32 bits gerados:", bits_msg[:32])

    # 2) Demonstra codificação de linha (apenas para ilustração, opcional)
    demo_line_coding()

    # 3) Simulação de BER para BPSK e QPSK com/sem codificação de repetição
    EbN0_dB_range = np.arange(0, 11, 2)  # 0, 2, 4, ..., 10 dB
    ber_bpsk, ber_bpsk_rep, ber_qpsk, ber_qpsk_rep = simulate_ber(EbN0_dB_range, n_bits=50000)

    # 4) Gera gráfico BER x Eb/N0
    plot_ber(EbN0_dB_range, ber_bpsk, ber_bpsk_rep, ber_qpsk, ber_qpsk_rep)