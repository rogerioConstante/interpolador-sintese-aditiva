# Interpolador Granular de Áudios — Síntese Aditiva

**Interpolador por síntese aditiva** é uma ferramenta de pesquisa para interpolação tímbrica entre dois áudios usando síntese aditiva granular. Com interface gráfica (Tkinter), permite carregar dois arquivos WAV, sintetizá-los por parciais espectrais e gerar um terceiro áudio que mistura gradualmente as características de timbre dos dois originais.

Versão atual: **v2.17**. O histórico de versões anteriores está disponível nas [tags](../../tags) do repositório.

---

## Requisitos

- Python 3.10+
- Linux Mint 22.3 / Ubuntu 24.04 (ou compatível)

### Dependências do sistema

```bash
sudo apt install libportaudio2 python3-tk
```

### Dependências Python

```bash
pip3 install -r requirements.txt
```

(Em sistemas que bloqueiam o `pip` global, use um ambiente virtual — `python3 -m venv .venv && source .venv/bin/activate` — ou acrescente `--break-system-packages`.)

---

## Execução

```bash
python3 interpolador_aditivo.py
```

---

## Fluxo básico de uso

1. **Configure** os parâmetros globais (Grão, Overlap, Parciais)
2. **Carregue** os dois arquivos de áudio (📂 Abrir)
3. **Processe** cada áudio individualmente (⚡ Processar)
4. **Configure** os parâmetros de interpolação (fator, modo, aleatoriedade)
5. **Gere** a interpolação (🎲 Gerar Interpolação)
6. **Ouça** e **salve** o resultado

---

## Documentação completa

Consulte o arquivo [`docs/MANUAL.odt`](docs/MANUAL.odt) para uma descrição detalhada de todos os parâmetros, modos de interpolação e fundamentos musicais da síntese aditiva granular.

---

## Conceito resumido

O programa divide cada áudio em pequenos fragmentos temporais chamados **grãos**. Cada grão é analisado via FFT e ressintetizado somando senoides (parciais), preservando frequências, amplitudes e fases. A interpolação consiste em montar uma nova sequência de grãos mesclando os dois áudios na proporção definida pelo slider de interpolação.

---

## Autor / Contexto

Desenvolvido por Rogério Constante no âmbito de pesquisa acadêmica na UFRJ.

## Licença

Distribuído sob a licença [GNU GPL v3](LICENSE).
