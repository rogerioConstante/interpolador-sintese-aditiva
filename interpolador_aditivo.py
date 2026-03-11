import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ================= CONFIGURAÇÕES GLOBAIS =================
FFT_SIZE = 8192
NUM_PARTIALS = 512  # Valor padrão
ENVELOPE_DURATION = 25
SAMPLE_RATE = 44100
OVERLAP = 0.8
DYNAMIC_RANGE_PRESERVATION = True
NOISE_FLOOR = 1e-6
DEFAULT_GAIN = 1.0

# ================= CLASSES AUXILIARES =================
@dataclass
class AudioTrack:
    """Classe para armazenar dados de um áudio individual"""
    name: str
    original: Optional[np.ndarray] = None
    synthesized_raw: Optional[np.ndarray] = None
    synthesized: Optional[np.ndarray] = None
    sample_rate: int = SAMPLE_RATE
    grains: List[np.ndarray] = None
    synthesized_grains: List[np.ndarray] = None
    grain_envelopes: List[np.ndarray] = None
    grain_frequencies: List[np.ndarray] = None
    grain_magnitudes: List[np.ndarray] = None
    grain_phases: List[np.ndarray] = None
    dynamic_profile: Optional[np.ndarray] = None
    gain: float = DEFAULT_GAIN
    file_path: str = ""
    color: str = ""
    processing: bool = False
    grain_source_indices: List[int] = None  # Índices originais dos grãos
    grain_track_id: int = 0  # ID do track (0 para áudio 1, 1 para áudio 2)
    
    def __post_init__(self):
        if self.grains is None:
            self.grains = []
        if self.synthesized_grains is None:
            self.synthesized_grains = []
        if self.grain_envelopes is None:
            self.grain_envelopes = []
        if self.grain_frequencies is None:
            self.grain_frequencies = []
        if self.grain_magnitudes is None:
            self.grain_magnitudes = []
        if self.grain_phases is None:
            self.grain_phases = []
        if self.dynamic_profile is None:
            self.dynamic_profile = np.array([])
        if self.grain_source_indices is None:
            self.grain_source_indices = []
        self.grain_track_id = 0 if "1" in self.name else 1

class GrainProcessor:
    """Processador de grãos de áudio"""
    
    def __init__(self, num_partials=NUM_PARTIALS):
        self.num_partials = num_partials
    
    def set_num_partials(self, num_partials):
        """Define o número de parciais para síntese"""
        self.num_partials = max(1, min(num_partials, 2048))  # Limite razoável
    
    def create_tukey_envelope(self, grain_length: int, alpha: float = 0.5) -> np.ndarray:
        """Cria envelope Tukey para suavização"""
        if grain_length <= 0:
            return np.ones(1)
        
        t = np.linspace(0, 1, grain_length)
        envelope = np.ones_like(t)
        
        fade_length = int(alpha * grain_length / 2)
        if fade_length > 0:
            fade_in = 0.5 * (1 - np.cos(np.pi * t[:fade_length] / (alpha / 2)))
            fade_out = 0.5 * (1 + np.cos(np.pi * (t[-fade_length:] - (1 - alpha / 2)) / (alpha / 2)))
            envelope[:fade_length] = fade_in
            envelope[-fade_length:] = fade_out
        
        return envelope
    
    def extract_grains_with_dynamic_profile(self, audio: np.ndarray, grain_duration_ms: int, 
                                          sample_rate: int, overlap: float) -> Tuple[List[np.ndarray], np.ndarray, int]:
        """Extrai grãos e calcula perfil dinâmico"""
        grain_samples = int(grain_duration_ms * sample_rate / 1000)
        hop_size = int(grain_samples * (1 - overlap))
        
        grains = []
        envelope_means = []
        num_grains = max(1, (len(audio) - grain_samples) // hop_size + 1)
        
        for i in range(num_grains):
            start = i * hop_size
            end = start + grain_samples
            
            if end > len(audio):
                grain = np.zeros(grain_samples)
                available = len(audio) - start
                grain[:available] = audio[start:start + available]
            else:
                grain = audio[start:end].copy()
            
            grains.append(grain)
            rms = np.sqrt(np.mean(grain**2))
            envelope_means.append(rms)
        
        envelope_means = np.array(envelope_means)
        if len(envelope_means) > 10:
            envelope_means = gaussian_filter1d(envelope_means, sigma=2)
        
        return grains, envelope_means, hop_size
    
    def analyze_grain_with_adaptive_threshold(self, grain: np.ndarray, sample_rate: int, 
                                            rms_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analisa grão com limiar adaptativo"""
        window = np.hanning(len(grain))
        windowed_grain = grain * window
        
        fft_result = np.fft.rfft(windowed_grain, n=FFT_SIZE)
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        frequencies = np.fft.rfftfreq(FFT_SIZE, 1/sample_rate)
        
        max_magnitude = np.max(magnitudes)
        if max_magnitude > NOISE_FLOOR:
            threshold = max(NOISE_FLOOR, 0.01 * max_magnitude * (1.0 + 0.5 * (1.0 - rms_level)))
            valid_mask = magnitudes > threshold
        else:
            valid_mask = magnitudes > NOISE_FLOOR
        
        nyquist_mask = frequencies < (sample_rate / 2)
        combined_mask = valid_mask & nyquist_mask
        
        if not np.any(combined_mask):
            sorted_indices = np.argsort(magnitudes[nyquist_mask])[::-1][:self.num_partials]
            sorted_frequencies = frequencies[nyquist_mask][sorted_indices]
            sorted_magnitudes = magnitudes[nyquist_mask][sorted_indices]
            sorted_phases = phases[nyquist_mask][sorted_indices]
        else:
            sorted_indices = np.argsort(magnitudes[combined_mask])[::-1][:self.num_partials]
            sorted_frequencies = frequencies[combined_mask][sorted_indices]
            sorted_magnitudes = magnitudes[combined_mask][sorted_indices]
            sorted_phases = phases[combined_mask][sorted_indices]
        
        return sorted_frequencies, sorted_magnitudes, sorted_phases
    
    def synthesize_grain_with_dynamic_scaling(self, frequencies: np.ndarray,
                                            magnitudes: np.ndarray,
                                            phases: np.ndarray,
                                            grain_duration_ms: int,
                                            sample_rate: int,
                                            target_level: float) -> np.ndarray:
        """Sintetiza grão com escalonamento dinâmico"""
        grain_samples = int(grain_duration_ms * sample_rate / 1000)

        # Garante tamanho mínimo
        grain_samples = max(grain_samples, 100)

        t = np.linspace(0, grain_duration_ms / 1000, grain_samples, endpoint=False)

        magnitudes = np.maximum(magnitudes, NOISE_FLOOR)
        if np.max(magnitudes) > 0:
            magnitudes = magnitudes / np.max(magnitudes)

        synthesized = np.zeros(grain_samples)
        valid_partials = min(self.num_partials, len(frequencies))

        for i in range(valid_partials):
            freq = frequencies[i]
            mag = magnitudes[i]
            phase = phases[i] if i < len(phases) else 0

            if freq > 20 and freq < sample_rate / 2:
                synthesized += mag * np.sin(2 * np.pi * freq * t + phase)

        if len(synthesized) > 0:
            current_rms = np.sqrt(np.mean(synthesized**2))
            if current_rms > NOISE_FLOOR:
                scale_factor = target_level / current_rms
                synthesized = synthesized * scale_factor
            else:
                synthesized = synthesized * target_level

        max_val = np.max(np.abs(synthesized))
        if max_val > 1.0:
            synthesized = synthesized / max_val * 0.95

        # Garante tamanho exato
        if len(synthesized) != grain_samples:
            # Ajusta para tamanho exato se necessário
            if len(synthesized) < grain_samples:
                padded = np.zeros(grain_samples)
                padded[:len(synthesized)] = synthesized
                synthesized = padded
            else:
                synthesized = synthesized[:grain_samples]

        return synthesized
    
    def synthesize_connection_grain(self, freq1: np.ndarray, mag1: np.ndarray, phase1: np.ndarray,
                                  freq2: np.ndarray, mag2: np.ndarray, phase2: np.ndarray,
                                  connection_duration_ms: int, sample_rate: int) -> np.ndarray:
        """Sintetiza grão de conexão com glissando entre parciais"""
        grain_samples = int(connection_duration_ms * sample_rate / 1000)
        t = np.linspace(0, connection_duration_ms / 1000, grain_samples, endpoint=False)
        
        # Limita número de parciais ao mínimo entre os dois grãos
        num_partials = min(len(freq1), len(freq2), self.num_partials)
        
        # Cria envelope para o grão de conexão (mais suave)
        envelope = self.create_tukey_envelope(grain_samples, alpha=0.8)
        
        synthesized = np.zeros(grain_samples)
        
        for i in range(num_partials):
            # Frequências inicial e final
            f_start = freq1[i] if i < len(freq1) else freq1[-1]
            f_end = freq2[i] if i < len(freq2) else freq2[-1]
            
            # Magnitudes (usa a menor magnitude para evitar picos)
            mag_start = mag1[i] if i < len(mag1) else 0
            mag_end = mag2[i] if i < len(mag2) else 0
            mag = min(mag_start, mag_end)
            
            # Fases (interpola linearmente entre as fases)
            p_start = phase1[i] if i < len(phase1) else 0
            p_end = phase2[i] if i < len(phase2) else 0
            
            # Aplica glissando linear de frequência
            freq_t = f_start + (f_end - f_start) * (t / (connection_duration_ms / 1000))
            
            # Interpola fase
            phase_t = p_start + (p_end - p_start) * (t / (connection_duration_ms / 1000))
            
            if f_start > 20 and f_end > 20 and mag > 0.001:
                synthesized += mag * np.sin(2 * np.pi * freq_t * t + phase_t)
        
        # Aplica envelope
        synthesized = synthesized * envelope
        
        # Normaliza
        max_val = np.max(np.abs(synthesized))
        if max_val > 0:
            synthesized = synthesized / max_val * 0.7
        
        return synthesized
    
    def create_connection_grain_glissando(self, freq1: np.ndarray, mag1: np.ndarray, phase1: np.ndarray,
                                    freq2: np.ndarray, mag2: np.ndarray, phase2: np.ndarray,
                                    connection_duration_ms: int, sample_rate: int) -> np.ndarray:
        """Cria grão de conexão com glissandi entre parciais correspondentes por amplitude - SEM ENVELOPE"""
        grain_samples = int(connection_duration_ms * sample_rate / 1000)
        t = np.linspace(0, connection_duration_ms / 1000, grain_samples, endpoint=False)

        # Limita ao número de parciais disponíveis
        num_partials = min(len(freq1), len(freq2), self.num_partials)

        synthesized = np.zeros(grain_samples)

        # Para cada parcial, cria um glissando da frequência do grão 1 para a do grão 2
        for i in range(num_partials):
            # Frequências inicial e final
            f_start = freq1[i] if i < len(freq1) else 0
            f_end = freq2[i] if i < len(freq2) else 0

            # Magnitudes (usa a média das magnitudes)
            mag_start = mag1[i] if i < len(mag1) else 0
            mag_end = mag2[i] if i < len(mag2) else 0
            mag = (mag_start + mag_end) / 2

            # Fases (interpola linearmente)
            p_start = phase1[i] if i < len(phase1) else 0
            p_end = phase2[i] if i < len(phase2) else 0

            # Verifica se as frequências são válidas
            if f_start > 20 and f_end > 20 and f_start < sample_rate/2 and f_end < sample_rate/2 and mag > 0.001:
                # Glissando linear de frequência
                freq_t = f_start + (f_end - f_start) * (t / (connection_duration_ms / 1000))

                # Interpolação de fase
                phase_t = p_start + (p_end - p_start) * (t / (connection_duration_ms / 1000))

                # Adiciona o parcial com glissando
                synthesized += mag * np.sin(2 * np.pi * freq_t * t + phase_t)

        # NORMALIZAÇÃO SIMPLES (sem envelope)
        max_val = np.max(np.abs(synthesized))
        if max_val > 0:
            synthesized = synthesized / max_val * 0.8  # Ganho ligeiramente reduzido

        return synthesized
    
    def reconstruct_audio_with_dynamics(self, grains: List[np.ndarray], 
                                      envelopes: List[np.ndarray],
                                      hop_size: int,
                                      original_length: int,
                                      dynamic_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstrói áudio preservando dinâmica"""
        if not grains:
            return np.zeros(original_length)
        
        reconstructed = np.zeros(original_length)
        grain_length = len(grains[0])
        
        for i, (grain, envelope) in enumerate(zip(grains, envelopes)):
            start = i * hop_size
            end = start + grain_length
            
            if start >= original_length:
                break
            
            enveloped_grain = grain * envelope
            
            if dynamic_profile is not None and i < len(dynamic_profile) and dynamic_profile[i] > 0:
                current_level = np.sqrt(np.mean(enveloped_grain**2))
                if current_level > NOISE_FLOOR:
                    scale_factor = dynamic_profile[i] / current_level
                    scale_factor = np.clip(scale_factor, 0.1, 10.0)
                    enveloped_grain = enveloped_grain * scale_factor
            
            if end > original_length:
                end = original_length
                enveloped_grain = enveloped_grain[:end-start]
            
            reconstructed[start:end] += enveloped_grain
        
        max_val = np.max(np.abs(reconstructed))
        if max_val > NOISE_FLOOR:
            reconstructed = reconstructed / max_val * 0.9
        
        return reconstructed

# ================= INTERFACE PRINCIPAL =================
class AudioInterpolatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interpolador Granular de Áudios")
        self.root.geometry("1400x900")
        
        # Dados dos áudios
        self.audio_tracks = [
            AudioTrack(name="Áudio 1", color="#4a6fa5"),
            AudioTrack(name="Áudio 2", color="#ff6b6b")
        ]

        # Inicializa referências aos botões
        for track in self.audio_tracks:
            track.save_btn = None
            track.play_orig_btn = None
            track.play_synth_btn = None

        self.grain_processor = GrainProcessor(NUM_PARTIALS)
        
        # Configurações compartilhadas
        self.grain_duration = 40  # ms
        self.overlap = OVERLAP
        self.num_partials = NUM_PARTIALS
        self.dynamic_preservation = DYNAMIC_RANGE_PRESERVATION
        self.normalize_audio = True
        
        # Configurações de interpolação
        self.interpolation_modes = ["Granular Simples", "Com Conexão de Grãos", "Espectral Progressivo"]
        self.interpolation_mode = self.interpolation_modes[0]
        self.interpolation_factor = 50.0  # 0-100%
        self.randomness_factor = 0.0  # 0-100%
        self.connection_grain_duration = 100  # ms (10-5000 ms)
        self.interpolated_audio = None
        self.interpolation_processing = False
        self.interpolation_gain_db = 0.0
        self.interpolation_gain_linear = 1.0

        # Configurações do modo Espectral Progressivo
        self.spectral_duration_mode  = "media"        # "soma" ou "media"
        self.spectral_blend_mode     = "selecao"      # "selecao" ou "amplitude"
        self.spectral_phase_mode     = "natural"      # "natural" ou "proporcional"
        self.spectral_interp_start   = 0.0            # 0.0–1.0 — início da interpolação
        self.spectral_interp_end     = 1.0            # 0.0–1.0 — fim   da interpolação

        # Criar variáveis de controle
        self.grain_var = tk.IntVar(value=self.grain_duration)
        self.overlap_var = tk.DoubleVar(value=self.overlap * 100)
        self.partials_var = tk.IntVar(value=self.num_partials)
        self.interp_var = tk.DoubleVar(value=self.interpolation_factor)
        self.random_var = tk.DoubleVar(value=self.randomness_factor)
        self.connection_var = tk.IntVar(value=self.connection_grain_duration)
        self.interp_gain_var = tk.DoubleVar(value=self.interpolation_gain_db)
        self.mode_var = tk.StringVar(value=self.interpolation_modes[0])
        self.spectral_duration_var = tk.StringVar(value="media")
        self.spectral_blend_var    = tk.StringVar(value="selecao")
        self.spectral_phase_var    = tk.StringVar(value="natural")
        self.dynamic_var = tk.BooleanVar(value=self.dynamic_preservation)
        self.normalize_var = tk.BooleanVar(value=self.normalize_audio)
        
        # Fila para comunicação
        self.audio_queue = queue.Queue()
        
        # Configuração de cores modernas
        self.colors = {
            'bg': '#1e1e2e',
            'card_bg': '#2d2d3d',
            'fg': '#cdd6f4',
            'fg_light': '#a6adc8',
            'primary': '#89b4fa',
            'secondary': '#585b70',
            'accent': '#f5c2e7',
            'interpolation': '#6bcf7f',
            'success': '#a6e3a1',
            'warning': '#f9e2af',
            'error': '#f38ba8',
            'border': '#45475a'
        }
        
        # Configurar fonte
        self.fonts = {
            'title': ('Segoe UI', 16, 'bold'),
            'subtitle': ('Segoe UI', 12),
            'body': ('Segoe UI', 10),
            'small': ('Segoe UI', 9)
        }
        
        # Configurar tema escuro para matplotlib
        self.matplotlib_colors = {
            'bg': '#1e1e2e',
            'fg': '#cdd6f4',
            'grid': '#45475a',
            'spines': '#585b70'
        }
        
        self.setup_ui()

    def setup_ui(self):
        """Configura a interface do usuário moderna"""
        self.setup_styles()

        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 5))

        # Título principal
        title_label = tk.Label(
            header_frame,
            text="🎛️ Interpolador Granular",
            font=self.fonts['title'],
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        title_label.pack(side=tk.LEFT)

        # Subtítulo ao lado
        subtitle_label = tk.Label(
            header_frame,
            text="por síntese aditiva",
            font=('Segoe UI', 10, 'italic'),
            fg=self.colors['fg_light'],
            bg=self.colors['bg'],
            padx=(10,)
        )
        subtitle_label.pack(side=tk.LEFT)

        # Container principal
        container = tk.Frame(main_frame, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True)

        # ========== COLUNA ESQUERDA COM SCROLL ==========
        left_frame = tk.Frame(container, bg=self.colors['bg'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Canvas para scroll
        canvas = tk.Canvas(left_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=370)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 1. CONFIGURAÇÕES GERAIS
        config_card = self.create_card(scrollable_frame, "⚙️ CONFIGURAÇÕES", top_margin=0, compact=True)

        # Sliders em grid 2x2
        sliders_grid = tk.Frame(config_card, bg=self.colors['card_bg'])
        sliders_grid.pack(fill=tk.X, pady=(0, 2))

        slider_configs = [
            ("Grão:", self.grain_var, 5, 100, self.on_grain_change, "ms", 0, 0),
            ("Overlap:", self.overlap_var, 10, 90, self.on_overlap_change, "%", 0, 1),
            ("Parciais:", self.partials_var, 8, 1024, self.on_partials_change, "", 1, 0)
        ]

        self.slider_labels = {}
        for text, var, min_val, max_val, command, unit, row, col in slider_configs:
            frame = tk.Frame(sliders_grid, bg=self.colors['card_bg'])
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=1)

            tk.Label(
                frame,
                text=text,
                font=('Segoe UI', 8),
                fg=self.colors['fg_light'],
                bg=self.colors['card_bg']
            ).pack(anchor=tk.W)

            slider = ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                variable=var,
                command=command,
                style='Modern.Horizontal.TScale',
                length=174
            )
            slider.pack(fill=tk.X, pady=(0, 0))

            value_label = tk.Label(
                frame,
                text=f"{var.get()}{unit}",
                font=('Segoe UI', 7),
                fg=self.colors['primary'],
                bg=self.colors['card_bg']
            )
            value_label.pack(anchor=tk.E)

            key = text.lower().replace(":", "").strip()
            self.slider_labels[key] = value_label

        # Controles extras
        extra_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        extra_frame.pack(fill=tk.X, pady=(2, 0))

        checks_frame = tk.Frame(extra_frame, bg=self.colors['card_bg'])
        checks_frame.pack(expand=True)

        ttk.Checkbutton(
            checks_frame,
            text="Preservar Dinâmica",
            variable=self.dynamic_var,
            style='Modern.TCheckbutton'
        ).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Checkbutton(
            checks_frame,
            text="Normalizar",
            variable=self.normalize_var,
            style='Modern.TCheckbutton'
        ).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(
            checks_frame,
            text="⏹ Parar",
            command=self.stop_playback,
            style='Secondary.TButton',
            width=8
        ).pack(side=tk.LEFT, padx=(10, 0))

        # 2. ÁUDIO 1
        track1_card = self.create_card(scrollable_frame, "🎵 ÁUDIO 1", top_margin=5, compact=True)
        self.create_track_controls_compact(track1_card, 0)

        # 3. ÁUDIO 2
        track2_card = self.create_card(scrollable_frame, "🎵 ÁUDIO 2", top_margin=5, compact=True)
        self.create_track_controls_compact(track2_card, 1)

        # 4. INTERPOLAÇÃO - Agora com mais linhas para o modo conexão
        interp_card = self.create_card(scrollable_frame, "🔄 INTERPOLAÇÃO", top_margin=5, compact=True)

        # ===== LINHA 1: FATOR DE INTERPOLAÇÃO =====
        self.interp_row1 = tk.Frame(interp_card, bg=self.colors['card_bg'])
        self.interp_row1.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            self.interp_row1,
            text="Interpolação:",
            font=('Segoe UI', 8),
            fg=self.colors['interpolation'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        interp_slider = ttk.Scale(
            self.interp_row1,
            from_=0,
            to=100,
            variable=self.interp_var,
            command=self.on_interpolation_change,
            style='Modern.Horizontal.TScale',
            length=200
        )
        interp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))

        self.interp_value_label = tk.Label(
            self.interp_row1,
            text=f"{self.interpolation_factor:.0f}%",
            font=('Segoe UI', 8, 'bold'),
            fg=self.colors['interpolation'],
            bg=self.colors['card_bg'],
            width=5
        )
        self.interp_value_label.pack(side=tk.RIGHT)

        # ===== LINHA 2: ALEATORIEDADE =====
        self.interp_row2 = tk.Frame(interp_card, bg=self.colors['card_bg'])
        self.interp_row2.pack(fill=tk.X, pady=(2, 3))

        tk.Label(
            self.interp_row2,
            text="Aleatoriedade:",
            font=('Segoe UI', 8),
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        random_slider = ttk.Scale(
            self.interp_row2,
            from_=0,
            to=100,
            variable=self.random_var,
            command=self.on_randomness_change,
            style='Modern.Horizontal.TScale',
            length=200
        )
        random_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))

        self.random_label = tk.Label(
            self.interp_row2,
            text=f"{self.randomness_factor:.0f}%",
            font=('Segoe UI', 8),
            fg=self.colors['interpolation'],
            bg=self.colors['card_bg'],
            width=5
        )
        self.random_label.pack(side=tk.RIGHT)

        # ===== LINHA 3: DURAÇÃO DO GRÃO DE CONEXÃO =====
        self.connection_row = tk.Frame(interp_card, bg=self.colors['card_bg'])
        self.connection_row.pack(fill=tk.X, pady=(2, 3))

        tk.Label(
            self.connection_row,
            text="Conexão:",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        self.connection_slider = ttk.Scale(
            self.connection_row,
            from_=10,
            to=5000,
            variable=self.connection_var,
            command=self.on_connection_change,
            style='Modern.Horizontal.TScale',
            length=200
        )
        self.connection_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))

        self.connection_label = tk.Label(
            self.connection_row,
            text=f"{self.connection_grain_duration}ms",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg'],
            width=5
        )
        self.connection_label.pack(side=tk.RIGHT)

        # ===== PAINEL: ESPECTRAL PROGRESSIVO =====
        self.spectral_row = tk.Frame(interp_card, bg=self.colors['card_bg'])
        # (visibilidade gerenciada por update_connection_controls_visibility)

        # Linha A: Duração
        spectral_dur_frame = tk.Frame(self.spectral_row, bg=self.colors['card_bg'])
        spectral_dur_frame.pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            spectral_dur_frame,
            text="Duração:",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg'],
            width=9,
            anchor='w'
        ).pack(side=tk.LEFT)

        for label, value in [("Média", "media"), ("Soma", "soma")]:
            rb = tk.Radiobutton(
                spectral_dur_frame,
                text=label,
                variable=self.spectral_duration_var,
                value=value,
                command=self.on_spectral_options_change,
                font=('Segoe UI', 8),
                fg=self.colors['fg'],
                bg=self.colors['card_bg'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card_bg'],
                activeforeground=self.colors['fg'],
            )
            rb.pack(side=tk.LEFT, padx=(4, 0))

        # Linha B: Mistura espectral
        spectral_blend_frame = tk.Frame(self.spectral_row, bg=self.colors['card_bg'])
        spectral_blend_frame.pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            spectral_blend_frame,
            text="Mistura:",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg'],
            width=9,
            anchor='w'
        ).pack(side=tk.LEFT)

        for label, value in [("Seleção", "selecao"), ("Amplitude", "amplitude")]:
            rb = tk.Radiobutton(
                spectral_blend_frame,
                text=label,
                variable=self.spectral_blend_var,
                value=value,
                command=self.on_spectral_options_change,
                font=('Segoe UI', 8),
                fg=self.colors['fg'],
                bg=self.colors['card_bg'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card_bg'],
                activeforeground=self.colors['fg'],
            )
            rb.pack(side=tk.LEFT, padx=(4, 0))

        # Linha B2: Modo das fases A e C
        spectral_phase_frame = tk.Frame(self.spectral_row, bg=self.colors['card_bg'])
        spectral_phase_frame.pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            spectral_phase_frame,
            text="Fases A/C:",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg'],
            width=9,
            anchor='w'
        ).pack(side=tk.LEFT)

        for label, value in [("Natural", "natural"), ("Proporcional", "proporcional")]:
            rb = tk.Radiobutton(
                spectral_phase_frame,
                text=label,
                variable=self.spectral_phase_var,
                value=value,
                command=self.on_spectral_options_change,
                font=('Segoe UI', 8),
                fg=self.colors['fg'],
                bg=self.colors['card_bg'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card_bg'],
                activeforeground=self.colors['fg'],
            )
            rb.pack(side=tk.LEFT, padx=(4, 0))

        # Linha C: Slider duplo — régua com três zonas
        spectral_range_label = tk.Label(
            self.spectral_row,
            text="Janela:",
            font=('Segoe UI', 8),
            fg=self.colors['accent'],
            bg=self.colors['card_bg'],
            anchor='w'
        )
        spectral_range_label.pack(fill=tk.X, pady=(4, 0))

        # Canvas único: trilha + handles + legendas das três zonas
        _SLIDER_H   = 38   # altura suficiente para trilha + texto das zonas
        _HANDLE_R   = 7
        _TRACK_Y    = 12   # centro vertical da trilha
        self._rs_H  = _HANDLE_R
        self._rs_TY = _TRACK_Y

        self.range_canvas = tk.Canvas(
            self.spectral_row,
            height=_SLIDER_H,
            bg=self.colors['card_bg'],
            highlightthickness=0,
            cursor="hand2"
        )
        self.range_canvas.pack(fill=tk.X, pady=(2, 4))

        # Bindings
        self.range_canvas.bind("<Configure>",       self._on_range_canvas_configure)
        self.range_canvas.bind("<ButtonPress-1>",   self._on_range_press)
        self.range_canvas.bind("<B1-Motion>",       self._on_range_drag)
        self.range_canvas.bind("<ButtonRelease-1>", self._on_range_release)

        # Estado interno do slider duplo
        self._range_drag_target  = None   # 'start' | 'end' | None
        self._range_max_start    = 1.0    # atualizado quando áudios carregam
        self._range_min_end      = 0.0    # atualizado quando áudios carregam

        # ===== LINHA 4: GANHO =====
        interp_row4 = tk.Frame(interp_card, bg=self.colors['card_bg'])
        interp_row4.pack(fill=tk.X, pady=(2, 5))

        tk.Label(
            interp_row4,
            text="Ganho:",
            font=('Segoe UI', 8),
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        gain_slider = ttk.Scale(
            interp_row4,
            from_=-24,
            to=24,
            variable=self.interp_gain_var,
            command=self.on_interpolation_gain_change,
            style='Modern.Horizontal.TScale',
            length=200
        )
        gain_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))

        self.interp_gain_label = tk.Label(
            interp_row4,
            text="0.0 dB",
            font=('Segoe UI', 8),
            fg=self.colors['primary'],
            bg=self.colors['card_bg'],
            width=5
        )
        self.interp_gain_label.pack(side=tk.RIGHT)

        # ===== LINHA 5: MODO DE INTERPOLAÇÃO =====
        mode_frame = tk.Frame(interp_card, bg=self.colors['card_bg'])
        mode_frame.pack(fill=tk.X, pady=(2, 5))

        tk.Label(
            mode_frame,
            text="Modo:",
            font=('Segoe UI', 8),
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=self.interpolation_modes,
            state='readonly',
            font=('Segoe UI', 8),
            height=4,
            width=20
        )
        mode_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        mode_combo.bind('<<ComboboxSelected>>', self.on_interpolation_mode_change)

        # ===== BOTÕES DE AÇÃO =====
        action_frame = tk.Frame(interp_card, bg=self.colors['card_bg'])
        action_frame.pack(fill=tk.X, pady=(5, 0))

        self.save_interp_btn = ttk.Button(
            action_frame,
            text="💾",
            command=self.save_interpolation,
            style='Secondary.TButton',
            width=4,
            state='disabled'
        )
        self.save_interp_btn.pack(side=tk.RIGHT)

        self.play_interp_btn = ttk.Button(
            action_frame,
            text="▶",
            command=self.play_interpolation,
            style='Secondary.TButton',
            width=8,
            state='disabled'
        )
        self.play_interp_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.generate_btn = ttk.Button(
            action_frame,
            text="🎲 Gerar Interpolação",
            command=self.generate_interpolation_auto_process,
            style='Primary.TButton',
            width=20,
            state='disabled'
        )
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Status da interpolação
        self.interp_status_var = tk.StringVar(value="")
        self.interp_status_label = tk.Label(
            interp_card,
            textvariable=self.interp_status_var,
            font=('Segoe UI', 8),
            fg=self.colors['success'],
            bg=self.colors['card_bg']
        )
        self.interp_status_label.pack(anchor=tk.W, pady=(3, 0))

        # ========== COLUNA DIREITA - GRÁFICOS ==========
        right_column = tk.Frame(container, bg=self.colors['bg'])
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Criar figura matplotlib - 3 linhas
        self.fig = Figure(figsize=(12, 12), facecolor=self.matplotlib_colors['bg'])
        self.fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.95, bottom=0.05, left=0.08, right=0.97)

        self.ax_original = []
        self.ax_synthesized = []

        for i in range(2):
            # Gráfico original
            ax_orig = self.fig.add_subplot(3, 2, i*2 + 1)
            ax_orig.set_title(f"{self.audio_tracks[i].name} - Original",
                            color=self.matplotlib_colors['fg'],
                            fontsize=10,
                            pad=6)
            self.setup_axes(ax_orig, show_labels=False)
            self.ax_original.append(ax_orig)

            # Gráfico sintetizado
            ax_synth = self.fig.add_subplot(3, 2, i*2 + 2)
            ax_synth.set_title(f"{self.audio_tracks[i].name} - Sintetizado",
                            color=self.matplotlib_colors['fg'],
                            fontsize=10,
                            pad=6)
            self.setup_axes(ax_synth, show_labels=False)
            self.ax_synthesized.append(ax_synth)

        # Gráfico interpolado
        self.ax_interpolated = self.fig.add_subplot(3, 2, (5, 6))
        self.ax_interpolated.set_title("Áudio Interpolado",
                                    color=self.matplotlib_colors['fg'],
                                    fontsize=11,
                                    pad=8)
        self.setup_axes(self.ax_interpolated, show_labels=True)

        # Canvas para gráficos
        self.canvas = FigureCanvasTkAgg(self.fig, right_column)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barra de status
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Segoe UI', 8),
            fg=self.colors['fg_light'],
            bg=self.colors['card_bg'],
            relief=tk.FLAT,
            padx=5,
            pady=2
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        # Configura visibilidade inicial do controle de conexão
        self.update_connection_controls_visibility()

        # Iniciar monitoramento
        self.root.after(100, self.monitor_queue)

    def update_connection_controls_visibility(self):
        """Atualiza visibilidade dos controles baseado no modo selecionado"""
        mode = self.interpolation_mode
        is_spectral = "Espectral" in mode

        # Barras de Interpolação e Aleatoriedade — ocultas no modo Espectral
        if is_spectral:
            self.interp_row1.pack_forget()
            self.interp_row2.pack_forget()
        else:
            self.interp_row1.pack(fill=tk.X, pady=(0, 3))
            self.interp_row2.pack(fill=tk.X, pady=(2, 3))

        # Painel de conexão de grãos
        if "Conexão" in mode:
            self.connection_row.pack(fill=tk.X, pady=(2, 3))
        else:
            self.connection_row.pack_forget()

        # Painel espectral progressivo
        if is_spectral:
            self.spectral_row.pack(fill=tk.X, pady=(2, 3))
        else:
            self.spectral_row.pack_forget()

    # ── Slider duplo (janela de interpolação espectral) ───────────────────

    def _range_canvas_width(self):
        w = self.range_canvas.winfo_width()
        return max(w, 100)

    def _val_to_x(self, val):
        r = self._rs_H
        w = self._range_canvas_width()
        return r + val * (w - 2 * r)

    def _x_to_val(self, x):
        r = self._rs_H
        w = self._range_canvas_width()
        return max(0.0, min(1.0, (x - r) / (w - 2 * r)))

    def _draw_range_slider(self):
        """Redesenha régua: trilha, faixas de zona, handles e legendas."""
        c  = self.range_canvas
        c.delete("all")
        w  = self._range_canvas_width()
        ty = self._rs_TY   # centro vertical da trilha (12)
        r  = self._rs_H    # raio dos handles (7)
        lh = 26            # y-base das legendas das zonas

        xs = self._val_to_x(self.spectral_interp_start)
        xe = self._val_to_x(self.spectral_interp_end)

        # ── Faixas coloridas das zonas (espessura 4, abaixo da trilha) ──
        TH = 4   # espessura da trilha
        # Zona A — Áudio 1 (azul primário, esquerda até xs)
        if xs > r:
            c.create_line(r, ty, xs, ty,
                          fill=self.colors['primary'], width=TH,
                          capstyle=tk.ROUND)
        # Zona B — Interpolação (verde, xs até xe)
        if xe > xs:
            c.create_line(xs, ty, xe, ty,
                          fill=self.colors['interpolation'], width=TH,
                          capstyle=tk.BUTT)
        # Zona C — Áudio 2 (rosa accent, xe até fim)
        if w - r > xe:
            c.create_line(xe, ty, w - r, ty,
                          fill=self.colors['accent'], width=TH,
                          capstyle=tk.ROUND)

        # ── Handles ──
        # Handle START (azul)
        c.create_oval(xs - r, ty - r, xs + r, ty + r,
                      fill=self.colors['primary'],
                      outline=self.colors['bg'], width=2,
                      tags="handle_start")
        # Handle END (rosa)
        c.create_oval(xe - r, ty - r, xe + r, ty + r,
                      fill=self.colors['accent'],
                      outline=self.colors['bg'], width=2,
                      tags="handle_end")

        # ── Legendas das três zonas ──
        s  = self.spectral_interp_start
        e  = self.spectral_interp_end
        fs = ('Segoe UI', 7)

        # Zona A: "Áudio 1  0%→{s%}"
        cx_a = (r + xs) / 2
        c.create_text(cx_a, lh,
                      text=f"Áudio 1\n0%→{s*100:.0f}%",
                      fill=self.colors['primary'],
                      font=fs, anchor='n', justify='center')

        # Zona B: "Interp  {s%}→{e%}"
        cx_b = (xs + xe) / 2
        c.create_text(cx_b, lh,
                      text=f"Interp\n{s*100:.0f}%→{e*100:.0f}%",
                      fill=self.colors['interpolation'],
                      font=fs, anchor='n', justify='center')

        # Zona C: "Áudio 2  {e%}→100%"
        cx_c = (xe + w - r) / 2
        c.create_text(cx_c, lh,
                      text=f"Áudio 2\n{e*100:.0f}%→100%",
                      fill=self.colors['accent'],
                      font=fs, anchor='n', justify='center')

    def _compute_range_limits(self):
        """
        Calcula max_start e min_end em função das durações reais dos áudios
        e do modo de duração (soma / média).
        Deve ser chamado sempre que um áudio é carregado ou o modo muda.
        """
        t1 = self.audio_tracks[0]
        t2 = self.audio_tracks[1]

        if t1.original is None or t2.original is None:
            # Sem áudios carregados: limites totais
            self._range_max_start = 0.99
            self._range_min_end   = 0.01
            return

        dur1 = len(t1.original) / t1.sample_rate
        dur2 = len(t2.original) / t2.sample_rate

        if self.spectral_duration_mode == "soma":
            dur_total = dur1 + dur2
        else:
            dur_total = (dur1 + dur2) / 2.0

        # marker_start não pode ultrapassar 99% da duração do Áudio 1
        self._range_max_start = min(0.99, (dur1 * 0.99) / dur_total)
        # marker_end não pode começar antes de 1% da duração do Áudio 2
        self._range_min_end   = max(0.01, (dur_total - dur2 * 0.99) / dur_total)

        # Garante que os valores atuais respeitam os novos limites
        MIN_GAP = 0.02
        self.spectral_interp_start = min(self.spectral_interp_start,
                                         self._range_max_start)
        self.spectral_interp_end   = max(self.spectral_interp_end,
                                         self._range_min_end)
        # Mantém gap mínimo
        if self.spectral_interp_end - self.spectral_interp_start < MIN_GAP:
            mid = (self.spectral_interp_start + self.spectral_interp_end) / 2
            self.spectral_interp_start = max(0.0, mid - MIN_GAP / 2)
            self.spectral_interp_end   = min(1.0, mid + MIN_GAP / 2)

    def _on_range_canvas_configure(self, event):
        self._draw_range_slider()

    def _on_range_press(self, event):
        xs = self._val_to_x(self.spectral_interp_start)
        xe = self._val_to_x(self.spectral_interp_end)
        r  = self._rs_H + 4

        dist_s = abs(event.x - xs)
        dist_e = abs(event.x - xe)

        if dist_s <= r and dist_s <= dist_e:
            self._range_drag_target = 'start'
        elif dist_e <= r:
            self._range_drag_target = 'end'
        else:
            self._range_drag_target = None

    def _on_range_drag(self, event):
        if self._range_drag_target is None:
            return
        val     = self._x_to_val(event.x)
        MIN_GAP = 0.02

        if self._range_drag_target == 'start':
            val = max(0.0, min(val, self._range_max_start))
            val = min(val, self.spectral_interp_end - MIN_GAP)
            self.spectral_interp_start = val
        else:
            val = min(1.0, max(val, self._range_min_end))
            val = max(val, self.spectral_interp_start + MIN_GAP)
            self.spectral_interp_end = val

        self._draw_range_slider()

    def _on_range_release(self, event):
        self._range_drag_target = None

    def on_spectral_options_change(self):
        """Atualiza opções do modo Espectral Progressivo"""
        self.spectral_duration_mode = self.spectral_duration_var.get()
        self.spectral_blend_mode    = self.spectral_blend_var.get()
        self.spectral_phase_mode    = self.spectral_phase_var.get()
        # Recalcula limites dos handles pois a duração total pode ter mudado
        self._compute_range_limits()
        self._draw_range_slider()

    def on_interpolation_mode_change(self, event=None):
        """Atualiza quando modo de interpolação muda"""
        self.interpolation_mode = self.mode_var.get()
        self.update_connection_controls_visibility()

    def on_connection_change(self, value):
        """Atualiza quando duração do grão de conexão muda"""
        try:
            self.connection_grain_duration = int(float(value))
            self.connection_var.set(self.connection_grain_duration)
            self.connection_label.config(text=f"{self.connection_grain_duration}ms")
        except Exception as e:
            print(f"Erro ao atualizar duração da conexão: {e}")

    def _generate_interpolation_with_connection(self):
        """Gera interpolação com grãos de conexão entre áudios diferentes"""
        try:
            track1 = self.audio_tracks[0]
            track2 = self.audio_tracks[1]

            # VERIFICAÇÃO DE DADOS
            if not track1.grain_frequencies or not track2.grain_frequencies:
                raise ValueError("Dados espectrais insuficientes. Processe ambos os áudios primeiro!")

            # 1. PASSO: Cria sequência granular simples (como no modo Granular Simples)
            grains1 = track1.synthesized_grains
            grains2 = track2.synthesized_grains

            if not grains1 or not grains2:
                raise ValueError("Ambos os áudios precisam ser processados primeiro!")

            # Fator de interpolação (0-1)
            alpha = self.interpolation_factor / 100.0

            # Número de grãos disponíveis
            num_grains1 = len(grains1)
            num_grains2 = len(grains2)

            # total_grains baseado no áudio mais longo, para não cortar o final
            max_grains = max(num_grains1, num_grains2)

            # Para qualquer fator (incluindo 0% e 100%)
            grains_from_1 = int(max_grains * (1 - alpha))
            grains_from_2 = int(max_grains * alpha)

            # Ajusta casos extremos
            if alpha == 0.0:
                grains_from_1 = num_grains1
                grains_from_2 = 0
            elif alpha == 1.0:
                grains_from_1 = 0
                grains_from_2 = num_grains2

            total_grains = grains_from_1 + grains_from_2

            # Seleciona grãos distribuídos uniformemente
            # O ganho de cada track é aplicado aqui, sobre os grãos selecionados
            gain1 = track1.gain  # gain linear do Áudio 1
            gain2 = track2.gain  # gain linear do Áudio 2

            if grains_from_1 > 0:
                # Índices cíclicos caso grains_from_1 > num_grains1
                indices1 = [i % num_grains1 for i in range(grains_from_1)]
                selected_grains1 = [grains1[i] * gain1 for i in indices1]
                selected_freqs1 = [track1.grain_frequencies[i] for i in indices1]
                selected_mags1 = [track1.grain_magnitudes[i] * gain1 for i in indices1]
                selected_phases1 = [track1.grain_phases[i] for i in indices1]
                selected_indices1 = indices1
            else:
                selected_grains1 = []
                selected_freqs1 = []
                selected_mags1 = []
                selected_phases1 = []
                selected_indices1 = []

            if grains_from_2 > 0:
                # Índices cíclicos caso grains_from_2 > num_grains2
                indices2 = [i % num_grains2 for i in range(grains_from_2)]
                selected_grains2 = [grains2[i] * gain2 for i in indices2]
                selected_freqs2 = [track2.grain_frequencies[i] for i in indices2]
                selected_mags2 = [track2.grain_magnitudes[i] * gain2 for i in indices2]
                selected_phases2 = [track2.grain_phases[i] for i in indices2]
                selected_indices2 = indices2
            else:
                selected_grains2 = []
                selected_freqs2 = []
                selected_mags2 = []
                selected_phases2 = []
                selected_indices2 = []

            if total_grains == 0:
                raise ValueError("Nenhum grão selecionado para interpolação!")

            # 2. PASSO: Monta sequência base com identificação de origem
            sequence = []
            source_info = []  # (track_id, original_index)

            # Determina qual áudio tem mais grãos selecionados
            if grains_from_1 >= grains_from_2:
                # Mais grãos do áudio 1
                if grains_from_2 > 0:
                    step = max(1, grains_from_1 // grains_from_2)

                    idx1 = 0
                    idx2 = 0
                    for i in range(total_grains):
                        if i % (step + 1) < step and idx1 < len(selected_grains1):
                            # Pega grão do áudio 1
                            sequence.append(selected_grains1[idx1])
                            source_info.append((0, selected_indices1[idx1]))
                            idx1 += 1
                        elif idx2 < len(selected_grains2):
                            # Pega grão do áudio 2
                            sequence.append(selected_grains2[idx2])
                            source_info.append((1, selected_indices2[idx2]))
                            idx2 += 1
                        elif idx1 < len(selected_grains1):
                            # Se acabou áudio 2, pega do áudio 1
                            sequence.append(selected_grains1[idx1])
                            source_info.append((0, selected_indices1[idx1]))
                            idx1 += 1
                else:
                    # Só tem áudio 1
                    for i, grain in enumerate(selected_grains1):
                        sequence.append(grain)
                        source_info.append((0, selected_indices1[i]))
            else:
                # Mais grãos do áudio 2
                if grains_from_1 > 0:
                    step = max(1, grains_from_2 // grains_from_1)

                    idx1 = 0
                    idx2 = 0
                    for i in range(total_grains):
                        if i % (step + 1) < step and idx2 < len(selected_grains2):
                            # Pega grão do áudio 2
                            sequence.append(selected_grains2[idx2])
                            source_info.append((1, selected_indices2[idx2]))
                            idx2 += 1
                        elif idx1 < len(selected_grains1):
                            # Pega grão do áudio 1
                            sequence.append(selected_grains1[idx1])
                            source_info.append((0, selected_indices1[idx1]))
                            idx1 += 1
                        elif idx2 < len(selected_grains2):
                            # Se acabou áudio 1, pega do áudio 2
                            sequence.append(selected_grains2[idx2])
                            source_info.append((1, selected_indices2[idx2]))
                            idx2 += 1
                else:
                    # Só tem áudio 2
                    for i, grain in enumerate(selected_grains2):
                        sequence.append(grain)
                        source_info.append((1, selected_indices2[i]))

            # 3. PASSO: Aplica aleatoriedade (se configurada)
            if self.randomness_factor > 0 and len(sequence) > 1:
                num_swaps = int(len(sequence) * (self.randomness_factor / 200.0))
                for _ in range(num_swaps):
                    i, j = random.randint(0, len(sequence) - 1), random.randint(0, len(sequence) - 1)
                    if i != j:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                        source_info[i], source_info[j] = source_info[j], source_info[i]

            # 4. PASSO: Identifica transições que precisam de grãos de conexão
            # Precisamos dos dados espectrais para cada grão
            spectral_data_for_sequence = []
            for track_id, orig_idx in source_info:
                if track_id == 0:
                    idx_in_selected = selected_indices1.index(orig_idx)
                    spectral_data_for_sequence.append((
                        selected_freqs1[idx_in_selected],
                        selected_mags1[idx_in_selected],
                        selected_phases1[idx_in_selected]
                    ))
                else:
                    idx_in_selected = selected_indices2.index(orig_idx)
                    spectral_data_for_sequence.append((
                        selected_freqs2[idx_in_selected],
                        selected_mags2[idx_in_selected],
                        selected_phases2[idx_in_selected]
                    ))

            # 5. PASSO: Cria sequência FINAL com grãos de conexão intercalados
            final_sequence = []  # Grãos intercalados (normal, conexão, normal, conexão, ...)
            final_sequence_info = []  # Informação sobre cada grão: ('normal', track_id) ou ('connection', -1)
            connection_count = 0

            for i in range(len(sequence)):
                # Adiciona o grão normal atual
                final_sequence.append(sequence[i])
                final_sequence_info.append(('normal', source_info[i][0]))

                # Verifica se precisa adicionar grão de conexão após este grão
                if i < len(sequence) - 1:
                    current_track_id = source_info[i][0]
                    next_track_id = source_info[i + 1][0]

                    # Verifica se é uma transição entre áudios diferentes
                    if current_track_id != next_track_id:
                        # Obtém dados espectrais para criar o grão de conexão
                        freq1, mag1, phase1 = spectral_data_for_sequence[i]
                        freq2, mag2, phase2 = spectral_data_for_sequence[i + 1]

                        # Cria grão de conexão SEM envelope
                        connection_grain = self.grain_processor.create_connection_grain_glissando(
                            freq1, mag1, phase1,
                            freq2, mag2, phase2,
                            self.connection_grain_duration,
                            track1.sample_rate
                        )

                        # Adiciona grão de conexão à sequência
                        final_sequence.append(connection_grain)
                        final_sequence_info.append(('connection', -1))
                        connection_count += 1

                # Atualiza progresso
                if len(sequence) > 0:
                    progress = (i + 1) / len(sequence) * 50
                    self.audio_queue.put(('interpolation_progress', progress))

            # 6. PASSO: Reconstrói áudio com CROSSFADE mantendo a ordem correta
            print(f"DEBUG: Grãos normais: {len(sequence)}, Grãos de conexão: {connection_count}, Total na sequência final: {len(final_sequence)}")
            print(f"DEBUG: Sequência: {[info[0] for info in final_sequence_info]}")

            # Configurações para reconstrução
            normal_grain_length = len(grains1[0]) if grains1 else len(grains2[0])
            connection_grain_length = int(self.connection_grain_duration * track1.sample_rate / 1000)

            # Parâmetros de crossfade (5ms como sugerido)
            crossfade_samples = int(0.005 * track1.sample_rate)  # 5ms em samples
            crossfade_samples = min(crossfade_samples, normal_grain_length // 4)  # No máximo 1/4 do grão
            crossfade_samples = max(10, crossfade_samples)  # Mínimo de 10 samples

            # Hop size para grãos normais
            normal_hop_size = int(normal_grain_length * (1 - self.overlap))

            # Calcula duração total estimada
            total_duration_samples = 0
            for i, (grain, info) in enumerate(zip(final_sequence, final_sequence_info)):
                grain_type, _ = info

                if grain_type == 'connection':
                    # Grão de conexão ocupa sua duração completa
                    total_duration_samples += len(grain)
                else:
                    # Grão normal
                    if i == len(final_sequence) - 1:
                        # Último grão ocupa sua duração completa
                        total_duration_samples += normal_grain_length
                    else:
                        # Grão normal usa hop size
                        total_duration_samples += normal_hop_size

            print(f"DEBUG: Duração total estimada: {total_duration_samples/track1.sample_rate:.2f}s")

            # Cria array para o áudio interpolado
            interpolated = np.zeros(total_duration_samples)

            # Envelopes para diferentes tipos de grãos
            normal_envelope = self.grain_processor.create_tukey_envelope(normal_grain_length, alpha=0.4)

            # Crossfade para grãos normais em transição de fonte (mesmo esquema do modo GS)
            sr_cg     = track1.sample_rate
            cf_samples_cg = int(0.005 * sr_cg)
            cf_samples_cg = max(10, min(cf_samples_cg, normal_hop_size // 2))
            fade_out_cg   = np.linspace(1.0, 0.0, cf_samples_cg)
            fade_in_cg    = np.linspace(0.0, 1.0, cf_samples_cg)

            # Para crossfade linear
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)

            # Posição atual no áudio
            current_position = 0
            prev_cg_source = None   # rastreia fonte do grão normal anterior

            # Processa a sequência FINAL mantendo a ordem correta
            for i in range(len(final_sequence)):
                grain = final_sequence[i]
                grain_type, track_id = final_sequence_info[i]

                if grain_type == 'normal':
                    # Grão normal
                    # Ajusta tamanho se necessário
                    if len(grain) != len(normal_envelope):
                        if len(grain) < len(normal_envelope):
                            padded = np.zeros(len(normal_envelope))
                            padded[:len(grain)] = grain
                            grain = padded
                        else:
                            grain = grain[:len(normal_envelope)]

                    # Aplica envelope
                    enveloped_grain = grain * normal_envelope
                    grain_length = len(enveloped_grain)

                    # Verifica se o próximo grão é de conexão
                    next_is_connection = (i + 1 < len(final_sequence) and
                                        final_sequence_info[i + 1][0] == 'connection')

                    if next_is_connection:
                        # Este grão normal é seguido por um grão de conexão

                        # ── Crossfade adaptativo se a fonte mudou ──────────
                        if prev_cg_source is not None and track_id != prev_cg_source:
                            n_cf = cf_samples_cg
                            if current_position > 0:
                                ov_start = current_position
                                ov_end   = min(ov_start + n_cf, len(interpolated))
                                n = ov_end - ov_start
                                if n > 0:
                                    interpolated[ov_start:ov_end] *= fade_out_cg[:n]
                                    enveloped_grain[:n] *= fade_in_cg[:n]

                        # 1. Adiciona o grão normal inteiro primeiro
                        end_pos = current_position + grain_length
                        if end_pos <= len(interpolated):
                            interpolated[current_position:end_pos] += enveloped_grain
                        else:
                            available = len(interpolated) - current_position
                            if available > 0:
                                interpolated[current_position:] += enveloped_grain[:available]

                        # 2. Aplica fade-out nos últimos crossfade_samples (antes do grão de conexão)
                        if grain_length >= crossfade_samples:
                            fade_out_start = current_position + grain_length - crossfade_samples
                            if fade_out_start >= 0 and fade_out_start + crossfade_samples <= len(interpolated):
                                interpolated[fade_out_start:fade_out_start + crossfade_samples] *= fade_out

                        # 3. Avança pelo hop size (não pela duração completa)
                        current_position += normal_hop_size

                    else:
                        # Grão normal sem conexão seguinte

                        # ── Crossfade adaptativo se a fonte mudou ──────────
                        if prev_cg_source is not None and track_id != prev_cg_source:
                            n_cf = cf_samples_cg
                            if current_position > 0:
                                ov_start = current_position
                                ov_end   = min(ov_start + n_cf, len(interpolated))
                                n = ov_end - ov_start
                                if n > 0:
                                    interpolated[ov_start:ov_end] *= fade_out_cg[:n]
                                    enveloped_grain[:n] *= fade_in_cg[:n]

                        # Adiciona normalmente
                        end_pos = current_position + grain_length
                        if end_pos <= len(interpolated):
                            interpolated[current_position:end_pos] += enveloped_grain
                        else:
                            available = len(interpolated) - current_position
                            if available > 0:
                                interpolated[current_position:] += enveloped_grain[:available]

                        # Avança
                        if i < len(final_sequence) - 1:
                            current_position += normal_hop_size
                        else:
                            current_position += grain_length

                    prev_cg_source = track_id   # atualiza fonte do grão anterior

                else:  # grain_type == 'connection'
                    # Grão de conexão
                    grain_length = len(grain)

                    # Verifica se o grão anterior era normal
                    prev_is_normal = (i > 0 and final_sequence_info[i - 1][0] == 'normal')

                    if prev_is_normal:
                        # Este grão de conexão segue um grão normal
                        # A posição já foi ajustada pelo hop size do grão normal anterior
                        # Adiciona o grão de conexão
                        end_pos = current_position + grain_length
                        if end_pos <= len(interpolated):
                            interpolated[current_position:end_pos] += grain
                        else:
                            available = len(interpolated) - current_position
                            if available > 0:
                                interpolated[current_position:] += grain[:available]

                        # Avança pela duração completa do grão de conexão
                        current_position += grain_length

                        # Verifica se o próximo grão é normal
                        next_is_normal = (i + 1 < len(final_sequence) and
                                        final_sequence_info[i + 1][0] == 'normal')

                        if next_is_normal:
                            # O próximo grão normal aplicará fade-in nos primeiros crossfade_samples
                            # Não precisamos fazer nada aqui
                            pass
                    else:
                        # Grão de conexão no início ou após outro grão de conexão (improvável)
                        # Adiciona normalmente
                        end_pos = current_position + grain_length
                        if end_pos <= len(interpolated):
                            interpolated[current_position:end_pos] += grain
                        else:
                            available = len(interpolated) - current_position
                            if available > 0:
                                interpolated[current_position:] += grain[:available]

                        current_position += grain_length

                # Atualiza progresso
                if len(final_sequence) > 0:
                    progress = 50 + ((i + 1) / len(final_sequence) * 50)
                    self.audio_queue.put(('interpolation_progress', progress))

            # 7. PASSO: Aplica configurações finais
            max_val = np.max(np.abs(interpolated))

            # Normalização automática
            if self.normalize_var.get() and max_val > 0:
                interpolated = interpolated / max_val * 0.9
            elif max_val > 1.0:
                interpolated = interpolated / max_val * 0.95

            # Preservar dinâmica
            if self.dynamic_var.get():
                interpolated = self._apply_dynamics_to_interpolation(
                    interpolated, track1, track2, self.interpolation_factor
                )

            # Aplica ganho da interpolação
            interpolated = interpolated * self.interpolation_gain_linear

            # Aplica fade-out de 80ms no final para evitar click
            interpolated = self._apply_fadeout(interpolated, track1.sample_rate, fadeout_ms=80)

            # Limiter final
            max_amp = np.max(np.abs(interpolated))
            if max_amp > 1.0:
                interpolated = np.tanh(interpolated * 0.8) * 1.2

            self.interpolated_audio = interpolated
            self.interpolated_audio_raw = interpolated.copy()

            # Atualiza interface
            self.audio_queue.put(('interpolation_done', interpolated))

        except Exception as e:
            import traceback
            print(f"Erro na interpolação com conexão: {str(e)}")
            print(traceback.format_exc())
            self.audio_queue.put(('error', -1, str(e)))

    def _generate_spectral_progressive_interpolation(self):
        """
        Modo Espectral Progressivo
        ──────────────────────────
        Cria um áudio interpolado onde cada grão é sintetizado misturando
        parciais do Áudio 1 e do Áudio 2 de forma progressiva.

        Posição p (0→1) controla simultaneamente:
          • qual grão de cada áudio é consultado (percurso proporcional)
          • a proporção de parciais de cada fonte (probabilidade estocástica)

        Modos de mistura (blend):
          selecao  – cada parcial vem INTEIRO de uma fonte (sorteio por probabilidade)
          amplitude – cada parcial tem magnitude interpolada entre as duas fontes
        """
        try:
            track1 = self.audio_tracks[0]
            track2 = self.audio_tracks[1]

            if not track1.grain_frequencies or not track2.grain_frequencies:
                raise ValueError("Processe ambos os áudios primeiro!")

            sr = track1.sample_rate
            grain_samples = int(self.grain_duration * sr / 1000)
            hop_size      = int(grain_samples * (1 - self.overlap))
            envelope      = self.grain_processor.create_tukey_envelope(grain_samples, alpha=0.4)

            num_grains1 = len(track1.grain_frequencies)
            num_grains2 = len(track2.grain_frequencies)

            # ── 1. Duração do áudio interpolado ───────────────────────────
            dur1_samples = len(track1.original)
            dur2_samples = len(track2.original)

            if self.spectral_duration_mode == "soma":
                total_samples = dur1_samples + dur2_samples
            else:  # "media"
                total_samples = (dur1_samples + dur2_samples) // 2

            # Número de grãos necessários para cobrir total_samples
            N = max(2, (total_samples - grain_samples) // hop_size + 1)

            # Ajusta total_samples para ser exatamente o que N grãos produzem
            total_samples = (N - 1) * hop_size + grain_samples

            interpolated = np.zeros(total_samples)

            # Ganhos individuais dos tracks
            gain1 = track1.gain
            gain2 = track2.gain

            # ── 2. Loop grão a grão ────────────────────────────────────────
            marker_start = self.spectral_interp_start   # 0.0–1.0
            marker_end   = self.spectral_interp_end     # 0.0–1.0
            if marker_end <= marker_start:
                marker_end = min(marker_start + 0.02, 1.0)

            # Número de slots em cada fase
            N_a = int(round(marker_start * (N - 1)))   # slots da Fase A
            N_c = N - 1 - int(round(marker_end * (N - 1)))  # slots da Fase C
            N_b = N - N_a - N_c                         # slots da Fase B
            N_b = max(N_b, 1)

            # Pré-computa o vetor t (igual para todos os grãos)
            t = np.linspace(0, self.grain_duration / 1000, grain_samples, endpoint=False)

            use_natural = (self.spectral_phase_mode == "natural")

            for i in range(N):
                pos = i / (N - 1)   # posição 0.0→1.0 no áudio total

                # ── Calcula idx1, idx2 e p conforme a fase ─────────────────
                if pos <= marker_start:
                    # ── FASE A: só Áudio 1 ──────────────────────────────────
                    p = 0.0
                    if use_natural:
                        # Reprodução 1-para-1: grão i → grão i do Áudio 1
                        idx1 = min(i, num_grains1 - 1)
                    else:
                        # Proporcional: escala os N_a slots sobre num_grains1
                        frac_a = (i / N_a) if N_a > 0 else 0.0
                        idx1 = int(round(frac_a * (num_grains1 - 1)))
                        idx1 = min(idx1, num_grains1 - 1)
                    idx2 = 0   # não usado com p=0

                elif pos >= marker_end:
                    # ── FASE C: só Áudio 2 ──────────────────────────────────
                    p = 1.0
                    if use_natural:
                        # Grão relativo dentro da Fase C
                        i_c = i - (N - N_c)          # posição dentro da fase C (0-based)
                        # Retoma do grão do Áudio 2 onde a Fase B terminou
                        idx2_b_end = min(int(round(marker_end * (num_grains2 - 1))),
                                         num_grains2 - 1)
                        idx2 = min(idx2_b_end + i_c, num_grains2 - 1)
                    else:
                        span_c = 1.0 - marker_end
                        frac_c = ((pos - marker_end) / span_c) if span_c > 0 else 1.0
                        idx2_start = int(round(marker_end * (num_grains2 - 1)))
                        idx2 = idx2_start + int(round(frac_c * (num_grains2 - 1 - idx2_start)))
                        idx2 = min(idx2, num_grains2 - 1)
                    idx1 = num_grains1 - 1   # não usado com p=1

                else:
                    # ── FASE B: interpolação progressiva ────────────────────
                    p = (pos - marker_start) / (marker_end - marker_start)
                    # Áudio 1: continua de onde a Fase A parou até o último grão
                    idx1_start = int(round(marker_start * (num_grains1 - 1)))
                    idx1 = idx1_start + int(round(p * (num_grains1 - 1 - idx1_start)))
                    idx1 = min(idx1, num_grains1 - 1)
                    # Áudio 2: do início até marker_end
                    idx2_end = int(round(marker_end * (num_grains2 - 1)))
                    idx2 = int(round(p * idx2_end))
                    idx2 = min(idx2, num_grains2 - 1)

                freq1  = track1.grain_frequencies[idx1]
                mag1   = track1.grain_magnitudes[idx1] * gain1
                phase1 = track1.grain_phases[idx1]

                freq2  = track2.grain_frequencies[idx2]
                mag2   = track2.grain_magnitudes[idx2] * gain2
                phase2 = track2.grain_phases[idx2]

                # ── 3. Síntese do grão interpolado ─────────────────────────
                n_total  = self.num_partials
                grain_out = np.zeros(grain_samples)

                if self.spectral_blend_mode == "selecao":
                    # ── Modo Seleção: cada slot é sorteado para uma fonte ──
                    # Normaliza magnitudes para usar como peso dentro de cada fonte
                    w1 = mag1 / (np.max(mag1) + 1e-9)
                    w2 = mag2 / (np.max(mag2) + 1e-9)

                    for slot in range(n_total):
                        # Sorteio: prob (1-p) → Áudio 1, prob p → Áudio 2
                        use_src2 = (random.random() < p)

                        if use_src2:
                            if slot < len(freq2) and freq2[slot] > 20 and freq2[slot] < sr / 2:
                                grain_out += w2[slot] * np.sin(
                                    2 * np.pi * freq2[slot] * t + phase2[slot])
                        else:
                            if slot < len(freq1) and freq1[slot] > 20 and freq1[slot] < sr / 2:
                                grain_out += w1[slot] * np.sin(
                                    2 * np.pi * freq1[slot] * t + phase1[slot])

                else:
                    # ── Modo Amplitude: lerp das magnitudes, freq da fonte majoritária ──
                    # Garante vetores de mesmo tamanho para interpolação
                    n_common = min(len(freq1), len(freq2), n_total)

                    # Normaliza magnitudes antes de interpolar
                    m1_norm = mag1[:n_common] / (np.max(mag1[:n_common]) + 1e-9)
                    m2_norm = mag2[:n_common] / (np.max(mag2[:n_common]) + 1e-9)
                    mag_interp = (1 - p) * m1_norm + p * m2_norm

                    for slot in range(n_common):
                        # Frequência: sorteio ponderado pela prob p
                        use_src2 = (random.random() < p)
                        freq_use  = freq2[slot] if use_src2 else freq1[slot]
                        phase_use = phase2[slot] if use_src2 else phase1[slot]

                        if freq_use > 20 and freq_use < sr / 2:
                            grain_out += mag_interp[slot] * np.sin(
                                2 * np.pi * freq_use * t + phase_use)

                # ── 4. Escala RMS e aplica envelope ────────────────────────
                rms = np.sqrt(np.mean(grain_out ** 2))
                if rms > NOISE_FLOOR:
                    grain_out = grain_out / rms * 0.5   # nível alvo fixo
                grain_out *= envelope

                # ── 5. Overlap-add ─────────────────────────────────────────
                start = i * hop_size
                end   = start + grain_samples
                if start >= total_samples:
                    break
                if end > total_samples:
                    end = total_samples
                    grain_out = grain_out[:end - start]
                interpolated[start:end] += grain_out

                # Progresso
                if N > 0 and i % max(1, N // 20) == 0:
                    self.audio_queue.put(('interpolation_progress', (i + 1) / N * 100))

            # ── 6. Pós-processamento ───────────────────────────────────────
            # Normalização
            max_val = np.max(np.abs(interpolated))
            if self.normalize_var.get() and max_val > 0:
                interpolated = interpolated / max_val * 0.9
            elif max_val > 1.0:
                interpolated = interpolated / max_val * 0.95

            # Dinâmica
            if self.dynamic_var.get():
                interpolated = self._apply_dynamics_to_interpolation(
                    interpolated, track1, track2, self.interpolation_factor)

            # Ganho da interpolação
            interpolated = interpolated * self.interpolation_gain_linear

            # Fade-out final (evita click)
            interpolated = self._apply_fadeout(interpolated, sr, fadeout_ms=80)

            # Limiter
            max_amp = np.max(np.abs(interpolated))
            if max_amp > 1.0:
                interpolated = np.tanh(interpolated * 0.8) * 1.2

            self.interpolated_audio     = interpolated
            self.interpolated_audio_raw = interpolated.copy()
            self.audio_queue.put(('interpolation_done', interpolated))

        except Exception as e:
            import traceback
            print(f"Erro na interpolação espectral progressiva: {str(e)}")
            print(traceback.format_exc())
            self.audio_queue.put(('error', -1, str(e)))

    def _generate_interpolation_thread(self):
        """Thread para geração da interpolação - agora com suporte a conexão"""
        try:
            # Verifica se temos os dados necessários
            track1 = self.audio_tracks[0]
            track2 = self.audio_tracks[1]

            if track1.original is None or track2.original is None:
                raise ValueError("Carregue ambos os áudios primeiro!")

            # VERIFICAÇÃO DE TAXAS DE AMOSTRAGEM
            if track1.sample_rate != track2.sample_rate:
                warning_msg = (
                    f"⚠️ AVISO: Taxas de amostragem diferentes!\n"
                    f"Áudio 1: {track1.sample_rate} Hz\n"
                    f"Áudio 2: {track2.sample_rate} Hz\n"
                    f"A interpolação será feita em {track1.sample_rate} Hz (taxa do primeiro áudio).\n\n"
                    f"Implicações:\n"
                    f"• O áudio 2 será reproduzido em velocidade diferente\n"
                    f"• Pode haver alteração no timbre, altura e duração"
                )
                self.audio_queue.put(('warning', warning_msg))

            # Escolhe o método baseado no modo
            if "Conexão" in self.interpolation_mode:
                # Modo com Conexão de Grãos
                self._generate_interpolation_with_connection()
            elif "Espectral" in self.interpolation_mode:
                # Modo Espectral Progressivo
                self._generate_spectral_progressive_interpolation()
            else:
                # Modo Granular Simples (original)
                self._generate_simple_granular_interpolation()

        except Exception as e:
            import traceback
            print(f"Erro geral na interpolação: {str(e)}")
            print(traceback.format_exc())
            self.audio_queue.put(('error', -1, str(e)))

    def _generate_simple_granular_interpolation(self):
        """Gera interpolação granular simples (método original)"""
        try:
            track1 = self.audio_tracks[0]
            track2 = self.audio_tracks[1]

            grains1 = track1.synthesized_grains
            grains2 = track2.synthesized_grains

            if not grains1 or not grains2:
                raise ValueError("Ambos os áudios precisam ser processados primeiro!")

            # Fator de interpolação (0-1)
            alpha = self.interpolation_factor / 100.0

            # Número de grãos disponíveis
            num_grains1 = len(grains1)
            num_grains2 = len(grains2)

            # total_grains baseado no áudio mais longo, para não cortar o final
            max_grains = max(num_grains1, num_grains2)

            # Para qualquer fator (incluindo 0% e 100%)
            grains_from_1 = int(max_grains * (1 - alpha))
            grains_from_2 = int(max_grains * alpha)

            # Ajusta casos extremos
            if alpha == 0.0:
                grains_from_1 = num_grains1
                grains_from_2 = 0
            elif alpha == 1.0:
                grains_from_1 = 0
                grains_from_2 = num_grains2

            total_grains = grains_from_1 + grains_from_2

            # Seleciona grãos distribuídos uniformemente
            # O ganho de cada track é aplicado aqui, sobre os grãos selecionados
            gain1 = track1.gain  # gain linear do Áudio 1
            gain2 = track2.gain  # gain linear do Áudio 2

            if grains_from_1 > 0:
                # Usa índices cíclicos se grains_from_1 > num_grains1
                indices1 = [i % num_grains1 for i in range(grains_from_1)]
                selected_grains1 = [grains1[i] * gain1 for i in indices1]
            else:
                selected_grains1 = []

            if grains_from_2 > 0:
                # Usa índices cíclicos se grains_from_2 > num_grains2
                indices2 = [i % num_grains2 for i in range(grains_from_2)]
                selected_grains2 = [grains2[i] * gain2 for i in indices2]
            else:
                selected_grains2 = []

            if total_grains == 0:
                raise ValueError("Nenhum grão selecionado para interpolação!")

            # Encontra o tamanho mais comum entre os grãos
            grain_lengths1 = [len(g) for g in grains1]
            grain_lengths2 = [len(g) for g in grains2]
            all_grain_lengths = grain_lengths1 + grain_lengths2

            if not all_grain_lengths:
                raise ValueError("Nenhum grão com tamanho válido!")

            from collections import Counter
            length_counts = Counter(all_grain_lengths)
            standard_grain_length = length_counts.most_common(1)[0][0]

            # Cria sequência de grãos como (grain, source_id) para rastrear origem
            sequence = []   # lista de (grain, source_id)  — source_id: 0=áudio1, 1=áudio2
            idx1 = 0
            idx2 = 0

            # Determina qual áudio tem mais grãos selecionados
            if grains_from_1 >= grains_from_2:
                if grains_from_2 > 0:
                    step = max(1, grains_from_1 // grains_from_2)
                    for i in range(total_grains):
                        if i % (step + 1) < step and idx1 < len(selected_grains1):
                            sequence.append((selected_grains1[idx1], 0))
                            idx1 += 1
                        elif idx2 < len(selected_grains2):
                            sequence.append((selected_grains2[idx2], 1))
                            idx2 += 1
                        elif idx1 < len(selected_grains1):
                            sequence.append((selected_grains1[idx1], 0))
                            idx1 += 1
                else:
                    for grain in selected_grains1:
                        sequence.append((grain, 0))
            else:
                if grains_from_1 > 0:
                    step = max(1, grains_from_2 // grains_from_1)
                    for i in range(total_grains):
                        if i % (step + 1) < step and idx2 < len(selected_grains2):
                            sequence.append((selected_grains2[idx2], 1))
                            idx2 += 1
                        elif idx1 < len(selected_grains1):
                            sequence.append((selected_grains1[idx1], 0))
                            idx1 += 1
                        elif idx2 < len(selected_grains2):
                            sequence.append((selected_grains2[idx2], 1))
                            idx2 += 1
                else:
                    for grain in selected_grains2:
                        sequence.append((grain, 1))

            if len(sequence) == 0:
                raise ValueError("Nenhum grão válido após padronização!")

            # Aplica aleatoriedade (se configurada) — troca pares (grain, source_id)
            if self.randomness_factor > 0 and len(sequence) > 1:
                num_swaps = int(len(sequence) * (self.randomness_factor / 200.0))
                for _ in range(num_swaps):
                    i, j = random.randint(0, len(sequence) - 1), random.randint(0, len(sequence) - 1)
                    if i != j:
                        sequence[i], sequence[j] = sequence[j], sequence[i]

            # Reconstrói áudio
            grain_length = standard_grain_length
            hop_size = int(grain_length * (1 - self.overlap))

            # Duração total
            if alpha == 0.0 and track1.original is not None:
                total_length = len(track1.original)
            elif alpha == 1.0 and track2.original is not None:
                total_length = len(track2.original)
            else:
                total_length = (len(sequence) - 1) * hop_size + grain_length

            interpolated = np.zeros(total_length)

            # Envelope Tukey alpha=0.4
            envelope = self.grain_processor.create_tukey_envelope(grain_length, alpha=0.4)

            # Crossfade adaptativo: aplicado na região de overlap quando a fonte muda
            # N amostras = 5 ms, limitado a metade do hop_size
            cf_samples = int(0.005 * track1.sample_rate)
            cf_samples = max(10, min(cf_samples, hop_size // 2))
            fade_out_cf = np.linspace(1.0, 0.0, cf_samples)
            fade_in_cf  = np.linspace(0.0, 1.0, cf_samples)

            prev_source = None   # rastreia fonte do grão anterior

            for i, (grain, src_id) in enumerate(sequence):
                start = i * hop_size
                end   = start + grain_length

                if start >= total_length:
                    break

                # Ajusta tamanho do grão ao envelope
                if len(grain) != grain_length:
                    if len(grain) < grain_length:
                        padded = np.zeros(grain_length)
                        padded[:len(grain)] = grain
                        grain = padded
                    else:
                        grain = grain[:grain_length]

                enveloped_grain = grain * envelope

                if end > total_length:
                    end = total_length
                    enveloped_grain = enveloped_grain[:end - start]

                # ── Crossfade na junção entre fontes diferentes ────────────
                # Quando a fonte muda, suaviza a região de overlap já no buffer
                # (fade-out no final do acumulado) e aplica fade-in no novo grão
                if prev_source is not None and src_id != prev_source and start > 0:
                    overlap_start = start
                    overlap_end   = min(start + cf_samples, total_length)
                    n_cf = overlap_end - overlap_start
                    if n_cf > 0:
                        # Fade-out sobre o que já está no buffer (final do grão anterior)
                        interpolated[overlap_start:overlap_end] *= fade_out_cf[:n_cf]
                        # Fade-in no novo grão antes de somar
                        enveloped_grain[:n_cf] *= fade_in_cf[:n_cf]

                interpolated[start:end] += enveloped_grain
                prev_source = src_id

                # Atualiza progresso
                if len(sequence) > 0 and i % max(1, len(sequence) // 10) == 0:
                    progress = (i + 1) / len(sequence) * 100
                    self.audio_queue.put(('interpolation_progress', progress))

            # Aplica configurações
            max_val = np.max(np.abs(interpolated))

            # Normalização automática
            if self.normalize_var.get() and max_val > 0:
                interpolated = interpolated / max_val * 0.9
            elif max_val > 1.0:
                interpolated = interpolated / max_val * 0.95

            # Preservar dinâmica
            if self.dynamic_var.get():
                interpolated = self._apply_dynamics_to_interpolation(
                    interpolated, track1, track2, self.interpolation_factor
                )

            # Aplica ganho da interpolação
            interpolated = interpolated * self.interpolation_gain_linear

            # Aplica fade-out de 80ms no final para evitar click
            interpolated = self._apply_fadeout(interpolated, track1.sample_rate, fadeout_ms=80)

            # Limiter final
            max_amp = np.max(np.abs(interpolated))
            if max_amp > 1.0:
                interpolated = np.tanh(interpolated * 0.8) * 1.2

            self.interpolated_audio = interpolated
            self.interpolated_audio_raw = interpolated.copy()

            # Atualiza interface
            self.audio_queue.put(('interpolation_done', interpolated))

        except Exception as e:
            import traceback
            print(f"Erro na interpolação granular simples: {str(e)}")
            print(traceback.format_exc())
            self.audio_queue.put(('error', -1, str(e)))

    def create_card(self, parent, title, top_margin=0, compact=False):
        """Cria um card moderno"""
        card = tk.Frame(parent, bg=self.colors['card_bg'], relief=tk.FLAT)
        card.pack(fill=tk.X, pady=(top_margin, 0))

        title_label = tk.Label(
            card,
            text=title,
            font=('Segoe UI', 9, 'bold') if compact else self.fonts['subtitle'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg'],
            pady=2 if compact else 5
        )
        title_label.pack(anchor=tk.W, padx=8 if compact else 15)

        separator = tk.Frame(card, height=1, bg=self.colors['border'])
        separator.pack(fill=tk.X, padx=8 if compact else 15)

        content = tk.Frame(card, bg=self.colors['card_bg'],
                        padx=8 if compact else 15,
                        pady=6 if compact else 10)
        content.pack(fill=tk.X)

        return content

    def create_track_controls_compact(self, parent, track_index):
        """Cria controles compactos para um track"""
        track = self.audio_tracks[track_index]

        top_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        top_frame.pack(fill=tk.X, pady=(0, 3))

        ttk.Button(
            top_frame,
            text="📂 Abrir",
            command=lambda idx=track_index: self.load_audio(idx),
            style='Secondary.TButton',
            width=10
        ).pack(side=tk.LEFT, padx=(0, 2))

        track.process_btn = ttk.Button(
            top_frame,
            text="⚡ Processar",
            command=lambda idx=track_index: self.process_track(idx),
            style='Primary.TButton',
            width=13
        )
        track.process_btn.pack(side=tk.LEFT, padx=(0, 2))

        track.info_var = tk.StringVar(value="Nenhum arquivo")
        info_label = tk.Label(
            top_frame,
            textvariable=track.info_var,
            font=('Segoe UI', 7),
            fg=self.colors['fg_light'],
            bg=self.colors['card_bg'],
            anchor=tk.W
        )
        info_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        track.status_var = tk.StringVar(value="")
        track.status_label = tk.Label(
            top_frame,
            textvariable=track.status_var,
            font=('Segoe UI', 7),
            fg=self.colors['success'],
            bg=self.colors['card_bg']
        )
        track.status_label.pack(side=tk.RIGHT)

        gain_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        gain_frame.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            gain_frame,
            text="Ganho:",
            font=('Segoe UI', 8),
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        track.gain_var = tk.DoubleVar(value=0)
        gain_slider = ttk.Scale(
            gain_frame,
            from_=-24,
            to=24,
            variable=track.gain_var,
            command=lambda v, idx=track_index: self.on_track_gain_change(idx, v),
            style='Modern.Horizontal.TScale',
            length=120
        )
        gain_slider.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)

        track.gain_label = tk.Label(
            gain_frame,
            text="0.0 dB",
            font=('Segoe UI', 7),
            fg=self.colors['primary'],
            bg=self.colors['card_bg'],
            width=6
        )
        track.gain_label.pack(side=tk.RIGHT)

        action_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        action_frame.pack(fill=tk.X)

        track.play_orig_btn = ttk.Button(
            action_frame,
            text="▶ Original",
            command=lambda idx=track_index: self.play_track(idx, 'original'),
            style='Secondary.TButton',
            width=12,
            state='disabled'
        )
        track.play_orig_btn.pack(side=tk.LEFT, padx=(5, 2))

        track.play_synth_btn = ttk.Button(
            action_frame,
            text="▶ Sintetizado",
            command=lambda idx=track_index: self.play_track(idx, 'synthesized'),
            style='Primary.TButton',
            width=15,
            state='disabled'
        )
        track.play_synth_btn.pack(side=tk.LEFT, padx=(0, 2))

        track.save_btn = ttk.Button(
            action_frame,
            text="💾",
            command=lambda idx=track_index: self.save_track(idx),
            style='Secondary.TButton',
            width=4,
            state='disabled'
        )
        track.save_btn.pack(side=tk.RIGHT)

    def setup_styles(self):
            """Configura estilos modernos para ttk"""
            style = ttk.Style()
            style.theme_use('clam')

            # Configurar cores base
            style.configure('.',
                        background=self.colors['card_bg'],
                        foreground=self.colors['fg'],
                        fieldbackground=self.colors['card_bg'],
                        troughcolor=self.colors['secondary'])

            # Botão secundário
            style.configure('Secondary.TButton',
                        background=self.colors['secondary'],
                        foreground=self.colors['fg'],
                        borderwidth=0,
                        focusthickness=0,
                        padding=2,
                        font=('Segoe UI', 8))

            # MAPEAMENTO
            style.map('Secondary.TButton',
                    background=[
                        ('active', '#6a6b80'),
                        ('pressed', '#4a5a70'),
                        ('disabled', '#585b70')
                    ],
                    foreground=[
                        ('disabled', '#a6adc8')  # Apenas texto mais claro
                    ])

            # Botão primário
            style.configure('Primary.TButton',
                        background=self.colors['primary'],
                        foreground=self.colors['bg'],
                        borderwidth=0,
                        focusthickness=0,
                        padding=2,
                        font=('Segoe UI', 8))
            style.map('Primary.TButton',
                    background=[('active', '#7aa3e0'), ('disabled', '#4a5a70')])

            # Botão de destaque
            style.configure('Accent.TButton',
                        background=self.colors['accent'],
                        foreground=self.colors['bg'],
                        borderwidth=0,
                        focusthickness=0,
                        padding=3,
                        font=('Segoe UI', 8))
            style.map('Accent.TButton',
                    background=[('active', '#e5b2d7'), ('disabled', '#4a5a70')])

            # Slider moderno
            style.configure('Modern.Horizontal.TScale',
                        background=self.colors['card_bg'])

            # Checkbox moderno
            style.configure('Modern.TCheckbutton',
                        background=self.colors['card_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 8))
            style.map('Modern.TCheckbutton',
                    background=[('active', self.colors['card_bg'])])

            # Combobox
            style.configure('TCombobox',
                        fieldbackground=self.colors['card_bg'],  # Fundo cinza igual aos cards
                        background=self.colors['card_bg'],
                        foreground=self.colors['fg'],
                        borderwidth=1,
                        relief='flat',
                        padding=3,
                        arrowsize=12,  # Tamanho das setinhas
                        font=('Segoe UI', 8))  # Fonte igual aos sliders

            # Mapeamento para estados
            style.map('TCombobox',
                    fieldbackground=[
                        ('readonly', self.colors['card_bg']),
                        ('active', self.colors['card_bg']),
                        ('disabled', '#4a5a70')
                    ],
                    background=[
                        ('readonly', self.colors['card_bg']),
                        ('active', self.colors['card_bg']),
                        ('disabled', '#4a5a70')
                    ],
                    foreground=[
                        ('readonly', self.colors['fg']),
                        ('active', self.colors['fg']),
                        ('disabled', self.colors['fg_light'])
                    ],
                    selectbackground=[('readonly', self.colors['card_bg'])],  # Cor quando selecionado
                    selectforeground=[('readonly', self.colors['fg'])]  # Texto quando selecionado
                    )

            # Listbox da combobox
            style.configure('TCombobox.Listbox',
                        background=self.colors['card_bg'],
                        foreground=self.colors['fg'],
                        selectbackground=self.colors['primary'],  # Item selecionado
                        selectforeground=self.colors['bg'],
                        borderwidth=0,
                        relief='flat',
                        font=('Segoe UI', 8))

    def create_track_controls(self, parent, track_index):
        """Cria controles para um track específico"""
        track = self.audio_tracks[track_index]

        # Frame para controles do áudio
        track_controls = tk.Frame(parent, bg=self.colors['card_bg'])
        track_controls.pack(fill=tk.X, pady=(0, 10))

        # Botão carregar
        load_btn = ttk.Button(
            track_controls,
            text="📂 Carregar",
            command=lambda idx=track_index: self.load_audio(idx),
            style='Secondary.TButton',
            width=8
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Botão processar individual
        self.audio_tracks[track_index].process_btn = ttk.Button(
            track_controls,
            text="⚡ Processar",
            command=lambda idx=track_index: self.process_track(idx),
            style='Primary.TButton',
            width=8
        )
        self.audio_tracks[track_index].process_btn.pack(side=tk.LEFT)

        # Info do arquivo
        self.create_info_display(parent, track_index)

        # Controle de ganho individual
        self.create_gain_control(parent, track_index)

        # Botões de ação
        self.create_action_buttons(parent, track_index)

    def create_info_display(self, parent, track_index):
        """Cria display de informações do arquivo"""
        info_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        info_frame.pack(fill=tk.X, pady=(10, 0))

        # Labels de informação
        self.audio_tracks[track_index].info_var = tk.StringVar(value="Nenhum arquivo carregado")
        info_label = tk.Label(
            info_frame,
            textvariable=self.audio_tracks[track_index].info_var,
            font=self.fonts['small'],
            fg=self.colors['fg_light'],
            bg=self.colors['card_bg'],
            wraplength=280,
            justify=tk.LEFT
        )
        info_label.pack(anchor=tk.W)

        # Status de processamento
        self.audio_tracks[track_index].status_var = tk.StringVar(value="")
        self.audio_tracks[track_index].status_label = tk.Label(
            info_frame,
            textvariable=self.audio_tracks[track_index].status_var,
            font=self.fonts['small'],
            fg=self.colors['success'],
            bg=self.colors['card_bg']
        )
        self.audio_tracks[track_index].status_label.pack(anchor=tk.W, pady=(2, 0))

    def create_gain_control(self, parent, track_index):
        """Cria controle de ganho para um track"""
        gain_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        gain_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            gain_frame,
            text="Ganho:",
            font=self.fonts['body'],
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(side=tk.LEFT)

        # Variável de ganho
        self.audio_tracks[track_index].gain_var = tk.DoubleVar(value=0)

        # Slider de ganho
        gain_slider = ttk.Scale(
            gain_frame,
            from_=-24,
            to=24,
            variable=self.audio_tracks[track_index].gain_var,
            command=lambda v, idx=track_index: self.on_track_gain_change(idx, v),
            style='Modern.Horizontal.TScale',
            length=180
        )
        gain_slider.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        # Label do valor
        self.audio_tracks[track_index].gain_label = tk.Label(
            gain_frame,
            text="0.0 dB",
            font=self.fonts['small'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg'],
            width=8
        )
        self.audio_tracks[track_index].gain_label.pack(side=tk.RIGHT)

        # Botões rápidos de ganho
        quick_gain_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        quick_gain_frame.pack(fill=tk.X, pady=(5, 0))

        for db in [-6, 0, 6]:
            btn = ttk.Button(
                quick_gain_frame,
                text=f"{db:+d} dB",
                command=lambda d=db, idx=track_index: self.set_track_gain(idx, d),
                style='Secondary.TButton',
                width=6
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))

    def create_action_buttons(self, parent, track_index):
        """Cria botões de ação para um track"""
        action_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        action_frame.pack(fill=tk.X, pady=(10, 0))

        # Botão reproduzir original
        play_orig_btn = ttk.Button(
            action_frame,
            text="▶ Original",
            command=lambda idx=track_index: self.play_track(idx, 'original'),
            style='Secondary.TButton',
            width=8
        )
        play_orig_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Botão reproduzir sintetizado
        play_synth_btn = ttk.Button(
            action_frame,
            text="▶ Sintetizado",
            command=lambda idx=track_index: self.play_track(idx, 'synthesized'),
            style='Primary.TButton',
            width=8
        )
        play_synth_btn.pack(side=tk.LEFT)

        # Botão salvar
        save_btn = ttk.Button(
            action_frame,
            text="💾 Salvar",
            command=lambda idx=track_index: self.save_track(idx),
            style='Secondary.TButton',
            width=6
        )
        save_btn.pack(side=tk.RIGHT)

    def setup_axes(self, ax, show_labels=True):
        """Configura eixos do matplotlib com tema escuro"""
        ax.set_facecolor(self.matplotlib_colors['bg'])
        ax.tick_params(colors=self.matplotlib_colors['fg'], labelsize=8)

        # Configura spines (bordas) mais finas
        for spine in ax.spines.values():
            spine.set_color(self.matplotlib_colors['spines'])
            spine.set_linewidth(0.8)

        # Grid mais sutil
        ax.grid(True, color=self.matplotlib_colors['grid'], alpha=0.2, linestyle='--', linewidth=0.5)

        # Labels com fonte menor - APENAS SE show_labels=True
        if show_labels:
            ax.set_xlabel('Tempo (s)', color=self.matplotlib_colors['fg'], fontsize=9)
            ax.set_ylabel('Amplitude', color=self.matplotlib_colors['fg'], fontsize=9)
        else:
            # Remove labels para evitar sobreposição (SOLICITADO)
            ax.set_xlabel('')
            ax.set_ylabel('Amplitude', color=self.matplotlib_colors['fg'], fontsize=9)
            # Remove alguns ticks também para limpar mais
            #ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Configura ticks do eixo Y mais compactos
        ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8)

        # Para estado vazio, configuramos limites razoáveis
        #ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)

    def on_grain_change(self, value):
        """Atualiza quando duração do grão muda"""
        try:
            self.grain_duration = int(float(value))
            # Atualiza label no grid
            if hasattr(self, 'slider_labels') and 'grão' in self.slider_labels:
                self.slider_labels['grão'].config(text=f"{self.grain_duration}ms")
        except Exception as e:
            print(f"Erro ao atualizar grão: {e}")

    def on_overlap_change(self, value):
        """Atualiza quando overlap muda"""
        self.overlap = float(value) / 100.0
        # Atualiza label no grid
        if hasattr(self, 'slider_labels') and 'overlap' in self.slider_labels:
            self.slider_labels['overlap'].config(text=f"{float(value):.0f}%")

    def on_partials_change(self, value):
        """Atualiza quando número de parciais muda"""
        try:
            num_partials = int(float(value))
            self.num_partials = max(8, min(num_partials, 2048))
            self.partials_var.set(self.num_partials)

            # Atualiza label no grid
            if hasattr(self, 'slider_labels') and 'parciais' in self.slider_labels:
                self.slider_labels['parciais'].config(text=f"{self.num_partials}")

            self.grain_processor.set_num_partials(self.num_partials)

        except Exception as e:
            print(f"Erro ao atualizar parciais: {e}")

    def on_partials_entry_change(self):
        """Atualiza quando número de parciais é alterado via entry"""
        try:
            value = self.partials_var.get()
            num_partials = int(value)
            num_partials = max(8, min(num_partials, 2048))
            self.num_partials = num_partials
            self.partials_var.set(num_partials)
            self.partials_label.config(text=f"{num_partials} parciais")

            # Atualiza o processador
            self.grain_processor.set_num_partials(num_partials)

        except Exception as e:
            print(f"Erro ao atualizar parciais via entry: {e}")

    def set_partials(self, num_partials):
        """Define número específico de parciais"""
        self.partials_var.set(num_partials)
        self.on_partials_change(num_partials)

    def on_interpolation_change(self, value):
        """Atualiza quando fator de interpolação muda"""
        try:
            self.interpolation_factor = float(value)
            self.interp_var.set(self.interpolation_factor)

            # Atualiza label no painel de interpolação (se existir)
            if hasattr(self, 'interp_value_label'):
                self.interp_value_label.config(text=f"{self.interpolation_factor:.0f}%")

            # # Atualiza label no grid
            # if hasattr(self, 'slider_labels') and 'interpolação' in self.slider_labels:
            #     self.slider_labels['interpolação'].config(text=f"{self.interpolation_factor:.0f}%")

        except Exception as e:
            print(f"Erro ao atualizar interpolação: {e}")

    def on_interpolation_entry_change(self):
        """Atualiza quando fator de interpolação é alterado via entry"""
        try:
            value = self.interp_var.get()
            self.interpolation_factor = max(0, min(float(value), 100))
            self.interp_var.set(self.interpolation_factor)
            self.interp_label.config(text=f"{self.interpolation_factor:.1f}% (0=Áudio1, 100=Áudio2)")
        except Exception as e:
            print(f"Erro ao atualizar interpolação via entry: {e}")

    def on_interpolation_gain_change(self, value):
        """Atualiza quando ganho da interpolação muda"""
        try:
            gain_db = float(value)
            self.interpolation_gain_db = gain_db
            self.interpolation_gain_linear = 10 ** (gain_db / 20.0)
            self.interp_gain_label.config(text=f"{gain_db:.2f} dB")

            # Aplica ganho ao áudio interpolado se existir
            if self.interpolated_audio is not None:
                # Gera novamente a interpolação com novo ganho
                self.apply_gain_to_interpolation()

        except Exception as e:
            print(f"Erro ao atualizar ganho da interpolação: {e}")

    def on_interpolation_gain_entry_change(self):
        """Atualiza quando ganho da interpolação é alterado via entry"""
        try:
            value = self.interp_gain_var.get()
            gain_db = max(-24, min(float(value), 24))
            self.interp_gain_var.set(gain_db)
            self.on_interpolation_gain_change(gain_db)
        except Exception as e:
            print(f"Erro ao atualizar ganho via entry: {e}")

    def set_interpolation_gain(self, db_value):
        """Define ganho específico para a interpolação"""
        self.interp_gain_var.set(db_value)
        self.on_interpolation_gain_change(db_value)

    def apply_gain_to_interpolation(self):
        """Aplica ganho atual ao áudio interpolado e atualiza tudo"""
        if self.interpolated_audio is None:
            return

        try:
            # Guarda o áudio raw original (sem ganho)
            if not hasattr(self, 'interpolated_audio_raw') or self.interpolated_audio_raw is None:
                self.interpolated_audio_raw = self.interpolated_audio.copy()

            # Aplica ganho ao áudio raw
            audio_with_gain = self.interpolated_audio_raw * self.interpolation_gain_linear

            # Aplica limiter se necessário
            max_amplitude = np.max(np.abs(audio_with_gain))
            if max_amplitude > 1.0:
                audio_with_gain = np.tanh(audio_with_gain * 0.8) * 1.2

            # Atualiza áudio interpolado com ganho
            self.interpolated_audio = audio_with_gain

            # Atualiza gráfico
            self.plot_interpolated_waveform(audio_with_gain)

        except Exception as e:
            print(f"Erro ao aplicar ganho à interpolação: {e}")

    def on_randomness_change(self, value):
        """Atualiza quando fator de aleatoriedade muda"""
        try:
            self.randomness_factor = float(value)
            self.random_var.set(self.randomness_factor)
            self.random_label.config(text=f"{self.randomness_factor:.1f}%")
        except Exception as e:
            print(f"Erro ao atualizar aleatoriedade: {e}")

    def on_randomness_entry_change(self):
        """Atualiza quando fator de aleatoriedade é alterado via entry"""
        try:
            value = self.random_var.get()
            self.randomness_factor = max(0, min(float(value), 100))
            self.random_var.set(self.randomness_factor)
            self.random_label.config(text=f"{self.randomness_factor:.1f}%")
        except Exception as e:
            print(f"Erro ao atualizar aleatoriedade via entry: {e}")

    def on_connection_grain_change(self):
        """Atualiza quando duração do grão de conexão muda"""
        try:
            value = self.connection_var.get()
            self.connection_grain_duration = max(5, min(int(value), 10000))
            self.connection_var.set(self.connection_grain_duration)
        except Exception as e:
            print(f"Erro ao atualizar grão de conexão: {e}")

    def on_track_gain_change(self, track_index, value):
        """Atualiza ganho de um track específico"""
        try:
            gain_db = float(value)
            gain_linear = 10 ** (gain_db / 20.0)

            track = self.audio_tracks[track_index]
            track.gain = gain_linear

            # Aplica ganho ao áudio sintetizado se existir
            if track.synthesized_raw is not None:
                synthesized_with_gain = track.synthesized_raw * gain_linear

                # Verifica clipping
                max_amplitude = np.max(np.abs(synthesized_with_gain))
                if max_amplitude > 1.0:
                    synthesized_with_gain = np.tanh(synthesized_with_gain * 0.8) * 1.2

                track.synthesized = synthesized_with_gain

                # Atualiza gráfico
                self.plot_track_waveform(track_index)

            # Atualiza label
            track.gain_label.config(text=f"{gain_db:.1f} dB")

        except Exception as e:
            print(f"Erro ao aplicar ganho: {e}")

    def set_track_gain(self, track_index, db_value):
        """Define ganho específico para um track"""
        track = self.audio_tracks[track_index]
        track.gain_var.set(db_value)
        self.on_track_gain_change(track_index, db_value)

    def load_audio(self, track_index):
        """Carrega arquivo de áudio para track específico"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Arquivos de áudio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac")]
        )

        if not file_path:
            return

        try:
            # Lê arquivo de áudio
            audio, sample_rate = sf.read(file_path)

            # Converte para mono se necessário
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Normaliza
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                if max_val > 1.0:
                    audio = audio / max_val * 0.95

            # Atualiza track
            track = self.audio_tracks[track_index]
            track.original = audio
            track.sample_rate = sample_rate
            track.file_path = file_path
            track.gain = 1.0
            track.gain_var.set(0)

            # CORREÇÃO: Habilita botão play original imediatamente
            if hasattr(track, 'play_orig_btn') and track.play_orig_btn:
                track.play_orig_btn.config(state='normal')

            # Desabilita outros botões até processar
            if hasattr(track, 'play_synth_btn') and track.play_synth_btn:
                track.play_synth_btn.config(state='disabled')
            if hasattr(track, 'save_btn') and track.save_btn:
                track.save_btn.config(state='disabled')

            # Reseta áudio sintetizado se existir
            track.synthesized = None
            track.synthesized_raw = None
            track.synthesized_grains = None

            # Atualiza informações
            filename = os.path.basename(file_path)
            duration = len(audio) / sample_rate
            track.info_var.set(f"{filename}\n{duration:.2f}s @ {sample_rate}Hz")

            # Plota forma de onda original
            self.plot_track_waveform(track_index, is_original=True)

            # Verifica se AMBOS os áudios estão carregados
            audio1_loaded = self.audio_tracks[0].original is not None
            audio2_loaded = self.audio_tracks[1].original is not None

            if audio1_loaded and audio2_loaded:
                # Ambos áudios carregados, habilita botão Gerar
                self.generate_btn.config(state='normal')
                # Recalcula limites do slider duplo com as durações reais
                self._compute_range_limits()
                self._draw_range_slider()

            # Atualiza status
            self.status_var.set(f"{track.name} carregado: {filename}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar áudio: {str(e)}")

    def process_track(self, track_index):
        """Processa síntese para track específico"""
        track = self.audio_tracks[track_index]

        if track.original is None:
            messagebox.showwarning("Aviso", f"Carregue um áudio para {track.name} primeiro!")
            return

        # Verifica se já está processando
        if track.processing:
            return

        # Marca como processando
        track.processing = True
        track.status_var.set("Processando...")
        track.status_label.config(fg=self.colors['warning'])

        # Desabilita botão de processar
        track.process_btn.config(state='disabled')

        # Atualiza status com número de parciais
        self.status_var.set(f"Processando {track.name} com {self.num_partials} parciais...")

        # Processa em thread separada
        thread = threading.Thread(
            target=self._process_track_thread,
            args=(track_index,)
        )
        thread.daemon = True
        thread.start()

    def process_all_tracks(self):
        """Processa síntese para todos os tracks"""
        for i in range(len(self.audio_tracks)):
            if self.audio_tracks[i].original is not None:
                self.process_track(i)

    def _process_track_thread(self, track_index):
        """Thread para processamento de um track"""
        try:
            track = self.audio_tracks[track_index]
            grain_duration = self.grain_duration
            overlap = self.overlap
            preserve_dynamics = self.dynamic_var.get()

            # Extrai grãos e calcula perfil dinâmico
            grains, dynamic_profile, hop_size = self.grain_processor.extract_grains_with_dynamic_profile(
                track.original,
                grain_duration,
                track.sample_rate,
                overlap
            )
            track.grains = grains

            # Armazena perfil dinâmico
            if preserve_dynamics:
                track.dynamic_profile = dynamic_profile
                if dynamic_profile is not None and np.max(dynamic_profile) > 0:
                    dynamic_profile_normalized = dynamic_profile / np.max(dynamic_profile)
                else:
                    dynamic_profile_normalized = None
            else:
                track.dynamic_profile = None
                dynamic_profile_normalized = None

            synthesized_grains = []
            envelopes = []
            grain_samples = int(grain_duration * track.sample_rate / 1000)

            # Cria envelope
            envelope = self.grain_processor.create_tukey_envelope(grain_samples, alpha=0.4)

            # Limpa arrays de análise espectral
            track.grain_frequencies = []
            track.grain_magnitudes = []
            track.grain_phases = []

            # Processa cada grão
            total_grains = len(grains)
            for i, grain in enumerate(grains):
                # Aplica envelope
                enveloped_grain = grain * envelope

                # Obtém nível dinâmico
                grain_level = dynamic_profile_normalized[i] if dynamic_profile_normalized is not None and i < len(dynamic_profile_normalized) else 0.5

                # Analisa grão
                frequencies, magnitudes, phases = self.grain_processor.analyze_grain_with_adaptive_threshold(
                    enveloped_grain,
                    track.sample_rate,
                    grain_level
                )

                # Armazena análise espectral para uso em interpolação
                track.grain_frequencies.append(frequencies)
                track.grain_magnitudes.append(magnitudes)
                track.grain_phases.append(phases)

                # Determina nível alvo
                if preserve_dynamics and dynamic_profile_normalized is not None:
                    target_level = dynamic_profile_normalized[i] if i < len(dynamic_profile_normalized) else 0.5
                else:
                    target_level = 0.5

                # Sintetiza grão
                synthesized_grain = self.grain_processor.synthesize_grain_with_dynamic_scaling(
                    frequencies,
                    magnitudes,
                    phases,
                    grain_duration,
                    track.sample_rate,
                    target_level
                )

                synthesized_grain = synthesized_grain * envelope
                synthesized_grains.append(synthesized_grain)
                envelopes.append(envelope)

                # Atualiza progresso a cada 10%
                if total_grains > 0 and i % max(1, total_grains // 10) == 0:
                    progress = (i + 1) / total_grains * 100
                    self.audio_queue.put(('progress', track_index, progress))

            # Reconstrói áudio
            if preserve_dynamics:
                synthesized_raw = self.grain_processor.reconstruct_audio_with_dynamics(
                    synthesized_grains,
                    envelopes,
                    hop_size,
                    len(track.original),
                    dynamic_profile_normalized
                )
            else:
                synthesized_raw = self.grain_processor.reconstruct_audio_with_dynamics(
                    synthesized_grains,
                    envelopes,
                    hop_size,
                    len(track.original),
                    None
                )

            # Aplica filtro passa-baixas
            if len(synthesized_raw) > 100:
                nyquist = track.sample_rate / 2
                cutoff = min(nyquist * 0.85, 16000)
                b, a = signal.butter(2, cutoff / nyquist, 'low')
                synthesized_raw = signal.filtfilt(b, a, synthesized_raw)

            # Normalização automática
            if self.normalize_var.get():
                max_val = np.max(np.abs(synthesized_raw))
                if max_val > 0:
                    synthesized_raw = synthesized_raw / max_val * 0.9

            # Salva versão raw
            track.synthesized_raw = synthesized_raw

            # Aplica ganho atual
            gain_linear = track.gain
            synthesized_with_gain = synthesized_raw * gain_linear

            # Aplica limiter se necessário
            max_amplitude = np.max(np.abs(synthesized_with_gain))
            if max_amplitude > 1.0:
                synthesized_with_gain = np.tanh(synthesized_with_gain * 0.8) * 1.2

            track.synthesized = synthesized_with_gain
            track.synthesized_grains = synthesized_grains
            track.grain_envelopes = envelopes

            # Envia resultado para a fila
            self.audio_queue.put(('done', track_index, synthesized_with_gain))

            # Envia sinal para habilitar botões
            self.audio_queue.put(('enable_buttons', track_index))

            # Salva parâmetros usados para processamento
            track.processing_params = {
                'grain_duration': grain_duration,
                'overlap': overlap,
                'num_partials': self.num_partials,
                'dynamic_preservation': preserve_dynamics,
                'normalize': self.normalize_var.get()
            }

        except Exception as e:
            self.audio_queue.put(('error', track_index, str(e)))

    def _apply_dynamics_to_interpolation(self, interpolated_audio, track1, track2, interpolation_factor):
        """Aplica perfil dinâmico ao áudio interpolado"""
        if not self.dynamic_var.get() or interpolated_audio is None:
            return interpolated_audio

        # Verifica se temos perfis dinâmicos
        if (track1.dynamic_profile is None or len(track1.dynamic_profile) == 0 or
            track2.dynamic_profile is None or len(track2.dynamic_profile) == 0):
            return interpolated_audio

        try:
            # Interpola os perfis dinâmicos baseado no fator de interpolação
            min_len = min(len(track1.dynamic_profile), len(track2.dynamic_profile))

            # Normaliza os perfis
            profile1 = track1.dynamic_profile[:min_len]
            profile2 = track2.dynamic_profile[:min_len]

            if np.max(profile1) > 0:
                profile1 = profile1 / np.max(profile1)
            if np.max(profile2) > 0:
                profile2 = profile2 / np.max(profile2)

            # Interpola linearmente os perfis
            alpha = interpolation_factor / 100.0
            interpolated_profile = (1 - alpha) * profile1 + alpha * profile2

            # Suaviza o perfil interpolado
            interpolated_profile = gaussian_filter1d(interpolated_profile, sigma=2)

            # Normaliza o perfil para evitar amplificação excessiva
            if np.max(interpolated_profile) > 0:
                interpolated_profile = interpolated_profile / np.max(interpolated_profile)

            # Divide o áudio interpolado em segmentos para aplicar dinâmica
            segment_length = len(interpolated_audio) // min_len

            result = np.zeros_like(interpolated_audio)

            for i in range(min_len):
                start = i * segment_length
                end = start + segment_length if i < min_len - 1 else len(interpolated_audio)

                segment = interpolated_audio[start:end]
                if len(segment) > 0 and interpolated_profile[i] > NOISE_FLOOR:
                    # Calcula RMS atual do segmento
                    current_rms = np.sqrt(np.mean(segment**2))
                    if current_rms > NOISE_FLOOR:
                        # Aplica scaling baseado no perfil dinâmico
                        scale_factor = interpolated_profile[i] / (current_rms + 0.01)
                        scale_factor = np.clip(scale_factor, 0.2, 2.0)
                        segment = segment * scale_factor

                result[start:end] = segment

            # Aplica normalização final para garantir que não há clipping
            max_val = np.max(np.abs(result))
            if max_val > 0.9:
                result = result / max_val * 0.9

            return result

        except Exception as e:
            print(f"Erro ao aplicar dinâmica à interpolação: {e}")
            return interpolated_audio

    def _resample_audio(self, audio, target_length):
        """Redimensiona áudio para comprimento alvo"""
        if len(audio) == target_length:
            return audio.copy()

        # Cria array de índices de origem
        source_indices = np.linspace(0, len(audio) - 1, target_length)

        # Interpolação linear simples
        resampled = np.interp(source_indices, np.arange(len(audio)), audio)

        return resampled

    def _apply_fadeout(self, audio, sample_rate, fadeout_ms=80):
        """Aplica fade-out cossenoidal no final do áudio para evitar click"""
        if audio is None or len(audio) == 0:
            return audio
        fadeout_samples = int(fadeout_ms * sample_rate / 1000)
        fadeout_samples = min(fadeout_samples, len(audio))
        if fadeout_samples <= 0:
            return audio
        # Envelope cossenoidal suave (0 -> pi/2 mapeado para 1 -> 0)
        fade = 0.5 * (1 + np.cos(np.pi * np.arange(fadeout_samples) / fadeout_samples))
        result = audio.copy()
        result[-fadeout_samples:] *= fade
        return result

    def _process_then_interpolate(self):
        """Processa os áudios e depois gera a interpolação"""
        # Processa ambos os áudios
        self.process_all_tracks()

        # Verifica periodicamente se o processamento terminou
        def check_processing():
            all_processed = True
            for track in self.audio_tracks:
                if track.processing:
                    all_processed = False
                    break

            if all_processed:
                # Todos processados, agora gera interpolação
                self.root.after(500, self.generate_interpolation)
            else:
                # Ainda processando, verifica novamente em 100ms
                self.root.after(100, check_processing)

        # Inicia verificação
        self.root.after(500, check_processing)

    def generate_interpolation(self):
        """Gera áudio interpolado com base nas configurações"""
        if self.interpolation_processing:
            return

        # Verifica se AMBOS os áudios estão carregados
        if self.audio_tracks[0].original is None or self.audio_tracks[1].original is None:
            messagebox.showwarning("Aviso", "Carregue ambos os áudios primeiro!")
            return

        # Verifica se ambos os áudios estão processados com parâmetros atuais
        needs_processing = self._check_if_reprocessing_needed()

        if needs_processing:
            # Se precisa processar, chama a versão que processa automaticamente
            self.generate_interpolation_auto_process()
            return

        # Marca como processando
        self.interpolation_processing = True
        self.interp_status_var.set("Processando...")
        self.interp_status_label.config(fg=self.colors['warning'])
        self.generate_btn.config(state='disabled')

        # Atualiza status
        self.status_var.set(f"Gerando interpolação...")

        # Processa em thread separada
        thread = threading.Thread(target=self._generate_interpolation_thread)
        thread.daemon = True
        thread.start()

    def generate_interpolation_auto_process(self):
        """Processa automaticamente e gera interpolação com verificação inteligente"""
        # Verifica se os parâmetros de síntese mudaram
        needs_reprocessing = self._check_if_reprocessing_needed()

        if needs_reprocessing:
            # Atualiza status
            self.interp_status_var.set("Processando áudios...")
            self.interp_status_label.config(fg=self.colors['warning'])

            # Processa primeiro, depois gera interpolação
            self._process_then_interpolate()
        else:
            # Já estão processados, gera direto
            self.generate_interpolation()

    def _check_if_reprocessing_needed(self):
        """Verifica se é necessário reprocessar os áudios"""
        needs_reprocessing = False
        reasons = []

        # Parâmetros que afetam a síntese
        current_params = {
            'grain_duration': self.grain_duration,
            'overlap': self.overlap,
            'num_partials': self.num_partials,
            'dynamic_preservation': self.dynamic_var.get(),
            'normalize': self.normalize_var.get()
        }

        # Verifica cada track
        for i, track in enumerate(self.audio_tracks):
            if track.original is None:
                continue

            # Se nunca foi processado
            if track.synthesized_grains is None or len(track.synthesized_grains) == 0:
                needs_reprocessing = True
                reasons.append(f"Áudio {i+1} não processado")
                continue

            # Verifica se os parâmetros atuais são diferentes dos usados no processamento anterior
            # (Podemos armazenar os parâmetros usados no objeto AudioTrack)
            if not hasattr(track, 'processing_params'):
                needs_reprocessing = True
                reasons.append(f"Áudio {i+1} precisa de parâmetros atualizados")
                continue

            # Compara parâmetros
            for param, value in current_params.items():
                if param in track.processing_params and track.processing_params[param] != value:
                    needs_reprocessing = True
                    reasons.append(f"Parâmetro '{param}' mudou no Áudio {i+1}")
                    break

        if reasons and needs_reprocessing:
            print(f"DEBUG: Reprocessamento necessário: {', '.join(reasons)}")

        return needs_reprocessing

    def plot_track_waveform(self, track_index, is_original=False):
        """Plota forma de onda de um track com duração total"""
        track = self.audio_tracks[track_index]

        if is_original:
            ax = self.ax_original[track_index]
            audio = track.original
            title_suffix = " - Original"
        else:
            ax = self.ax_synthesized[track_index]
            audio = track.synthesized
            title_suffix = " - Sintetizado"

        ax.clear()

        if audio is not None and len(audio) > 0:
            # Calcula duração total em segundos
            total_duration = len(audio) / track.sample_rate

            # Cria array de tempo para todo o áudio
            time_array = np.linspace(0, total_duration, len(audio))

            # Para áudios muito longos, usamos downsampling para visualização
            max_display_points = 5000  # Limite de pontos para visualização

            if len(audio) > max_display_points:
                # Downsampling para visualização
                step = len(audio) // max_display_points
                audio_display = audio[::step]
                time_display = time_array[::step]
            else:
                audio_display = audio
                time_display = time_array

            # Encontra amplitude máxima
            max_abs = np.max(np.abs(audio_display))

            # Plota forma de onda
            linewidth = 0.5 if len(audio_display) > 20000 else 0.8
            ax.plot(time_display, audio_display, color=track.color,
                   linewidth=linewidth, alpha=0.8)

            # Linha de referência zero
            ax.axhline(y=0, color=self.matplotlib_colors['fg'],
                      linestyle='-', alpha=0.2, linewidth=0.5)

            # Linhas de clipping se necessário
            if max_abs > 0.95:
                ax.axhline(y=1.0, color=self.colors['error'],
                          linestyle='--', alpha=0.4, linewidth=0.5)
                ax.axhline(y=-1.0, color=self.colors['error'],
                          linestyle='--', alpha=0.4, linewidth=0.5)

            # Configura limites dos eixos - MOSTRANDO O ÁUDIO INTEIRO
            ax.set_xlim(0, total_duration)

            # Ajusta eixo Y
            y_margin = 0.05
            y_max = min(1.0, max_abs + y_margin)
            ax.set_ylim(-y_max, y_max)

            # Configura ticks do eixo X de forma inteligente
            # Calcula intervalos apropriados baseados na duração total
            if total_duration > 60:  # Mais de 1 minuto
                interval = 10  # A cada 10 segundos
            elif total_duration > 10:  # Mais de 10 segundos
                interval = 2  # A cada 2 segundos
            elif total_duration > 1.5:  # Mais de 2 segundos
                interval = 0.5  # A cada 0.5 segundos
            else:  # Menos de 2 segundos
                interval = 0.1  # A cada 0.1 segundos

            x_ticks = np.arange(0, total_duration + interval/2, interval)
            ax.set_xticks(x_ticks)

            # Formata os labels do eixo X
            if total_duration > 60:
                # Formato MM:SS para áudios longos
                tick_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in x_ticks]
            elif total_duration > 10:
                # Apenas segundos inteiros
                tick_labels = [f"{int(t)}" if t == int(t) else f"{t:.1f}" for t in x_ticks]
            else:
                # Segundos com uma casa decimal
                tick_labels = [f"{t:.1f}" for t in x_ticks]

            ax.set_xticklabels(tick_labels, fontsize=8)

            # Configura ticks do eixo Y
            y_ticks = np.linspace(-y_max, y_max, 5)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=8)

        else:
            ax.text(0.5, 0.5, "Nenhum áudio\ncarregado" if is_original else "Não processado",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color=self.matplotlib_colors['fg'],
                   fontsize=10,
                   alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=8)
            ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
            ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8)

        # Atualiza título
        if audio is not None and len(audio) > 0:
            total_duration = len(audio) / track.sample_rate
            duration_text = f" ({total_duration:.2f}s)"
        else:
            duration_text = ""

        ax.set_title(f"{track.name}{title_suffix}{duration_text}",
                    color=self.matplotlib_colors['fg'],
                    fontsize=11,
                    pad=8)

        # Reconfigura estilo básico
        #self.setup_axes(ax, show_labels=False)  # SEMPRE False conforme solicitado

        # Atualiza canvas
        self.canvas.draw()

    def plot_interpolated_waveform(self, audio):
        """Plota forma de onda do áudio interpolado"""
        ax = self.ax_interpolated
        ax.clear()

        # Usa áudio interpolado se não for especificado
        if audio is None:
            audio = self.interpolated_audio

        if audio is not None and len(audio) > 0:
            # Assume sample rate do primeiro áudio
            sample_rate = self.audio_tracks[0].sample_rate
            total_duration = len(audio) / sample_rate

            # Cria array de tempo para todo o áudio
            time_array = np.linspace(0, total_duration, len(audio))

            # Downsampling para visualização se necessário
            max_display_points = 5000
            if len(audio) > max_display_points:
                step = len(audio) // max_display_points
                audio_display = audio[::step]
                time_display = time_array[::step]
            else:
                audio_display = audio
                time_display = time_array

            max_abs = np.max(np.abs(audio_display))

            # Plota com cor específica para interpolação
            linewidth = 0.5 if len(audio_display) > 20000 else 0.8
            ax.plot(time_display, audio_display, color=self.colors['interpolation'],
                   linewidth=linewidth, alpha=0.8)

            ax.axhline(y=0, color=self.matplotlib_colors['fg'],
                      linestyle='-', alpha=0.2, linewidth=0.5)

            if max_abs > 0.95:
                ax.axhline(y=1.0, color=self.colors['error'],
                          linestyle='--', alpha=0.4, linewidth=0.5)
                ax.axhline(y=-1.0, color=self.colors['error'],
                          linestyle='--', alpha=0.4, linewidth=0.5)

            # MOSTRANDO O ÁUDIO INTEIRO
            ax.set_xlim(0, total_duration)

            y_margin = 0.05
            y_max = min(1.0, max_abs + y_margin)
            ax.set_ylim(-y_max, y_max)

            # Configura ticks do eixo X
            if total_duration > 60:
                interval = 10
                x_ticks = np.arange(0, total_duration + interval/2, interval)
                tick_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in x_ticks]
            elif total_duration > 10:
                interval = 2
                x_ticks = np.arange(0, total_duration + interval/2, interval)
                tick_labels = [f"{int(t)}" if t == int(t) else f"{t:.1f}" for t in x_ticks]
            elif total_duration > 4:
                interval = 0.5
                x_ticks = np.arange(0, total_duration + interval/2, interval)
                tick_labels = [f"{t:.1f}" for t in x_ticks]
            else:
                interval = 0.1
                x_ticks = np.arange(0, total_duration + interval/2, interval)
                tick_labels = [f"{t:.1f}" for t in x_ticks]

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(tick_labels, fontsize=8)

            y_ticks = np.linspace(-y_max, y_max, 5)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=8)

        else:
            ax.text(0.5, 0.5, "Nenhuma interpolação\ngerada",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color=self.matplotlib_colors['fg'],
                   fontsize=10,
                   alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=8)
            ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
            ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8)

        # Título com formato solicitado
        title = self.get_interpolation_title()
        ax.set_title(title,
                    color=self.matplotlib_colors['fg'],
                    fontsize=12,
                    pad=8)

        #self.setup_axes(ax, show_labels=True)
        self.canvas.draw()

    def get_interpolation_title(self):
        """Gera título para o painel do gráfico (inclui duração)"""
        if self.interpolated_audio is None:
            return "Áudio Interpolado"

        sample_rate     = self.audio_tracks[0].sample_rate
        total_duration  = len(self.interpolated_audio) / sample_rate
        title           = self._build_interp_name(include_duration=True)
        return f"{title}  [{total_duration:.1f}s]"

    def _build_interp_name(self, include_duration=False):
        """
        Monta o nome base da interpolação (usado no título do gráfico
        e como nome de arquivo).

        Estrutura:
          Modo + Configurações Gerais + Configurações Específicas [+ duração]

        GS  → GS_G{grão}ms_O{overlap}p_P{parciais}_I{interp}p_A{random}p
        CG  → CG_G{grão}ms_O{overlap}p_P{parciais}_I{interp}p_A{random}p_C{conexão}ms
        EP  → EP_G{grão}ms_O{overlap}p_P{parciais}_{dur}{blend}{fases}_J{start}p{end}p
        """
        g  = self.grain_duration
        ov = int(round(self.overlap * 100))
        p  = self.num_partials

        if "Espectral" in self.interpolation_mode:
            mode    = "EP"
            dur_tag   = "S" if self.spectral_duration_mode == "soma"         else "M"
            blend_tag = "A" if self.spectral_blend_mode    == "amplitude"    else "Sel"
            phase_tag = "P" if self.spectral_phase_mode    == "proporcional" else "N"
            js = int(round(self.spectral_interp_start * 100))
            je = int(round(self.spectral_interp_end   * 100))
            specific = f"_{dur_tag}{blend_tag}{phase_tag}_J{js}p{je}p"
        elif "Conexão" in self.interpolation_mode:
            mode     = "CG"
            i_pct    = int(round(self.interpolation_factor))
            a_pct    = int(round(self.randomness_factor))
            c_ms     = self.connection_grain_duration
            specific = f"_I{i_pct}p_A{a_pct}p_C{c_ms}ms"
        else:
            mode     = "GS"
            i_pct    = int(round(self.interpolation_factor))
            a_pct    = int(round(self.randomness_factor))
            specific = f"_I{i_pct}p_A{a_pct}p"

        name = f"{mode}_G{g}ms_O{ov}p_P{p}{specific}"

        if include_duration and self.interpolated_audio is not None:
            sr  = self.audio_tracks[0].sample_rate
            dur = len(self.interpolated_audio) / sr
            name += f"_D{dur:.1f}s"

        return name

    def play_interpolation(self):
        """Reproduz áudio interpolado"""
        if self.interpolated_audio is None:
            messagebox.showwarning("Aviso", "Gere uma interpolação primeiro!")
            return

        try:
            sd.stop()
            sample_rate = self.audio_tracks[0].sample_rate
            sd.play(self.interpolated_audio, sample_rate)
            duration = len(self.interpolated_audio) / sample_rate
            self.status_var.set(f"Reproduzindo áudio interpolado ({duration:.1f}s)...")
        except Exception as e:
            self.status_var.set(f"Erro na reprodução: {str(e)}")

    def save_interpolation(self):
        """Salva áudio interpolado"""
        if self.interpolated_audio is None:
            messagebox.showwarning("Aviso", "Gere uma interpolação primeiro!")
            return

        # Usa o nome estruturado (sem duração) como base para o arquivo
        base_name  = self._build_interp_name(include_duration=False)
        default_name = f"{base_name}.wav"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("Arquivos WAV", "*.wav"), ("Arquivos FLAC", "*.flac")],
            initialfile=default_name
        )

        if file_path:
            try:
                audio_to_save = self.interpolated_audio
                sample_rate = self.audio_tracks[0].sample_rate

                # Garante que está no range [-1, 1]
                max_val = np.max(np.abs(audio_to_save))
                if max_val > 1.0:
                    audio_to_save = audio_to_save / max_val * 0.99

                # Converte para int16
                audio_int16 = (audio_to_save * 32767).astype(np.int16)

                sf.write(file_path, audio_int16, sample_rate)

                filename = os.path.basename(file_path)
                self.status_var.set(f"Interpolação salva: {filename}")

                messagebox.showinfo("Sucesso",
                                  f"Áudio interpolado salvo com sucesso!\n"
                                  f"Arquivo: {filename}\n"
                                  f"Duração: {len(audio_to_save)/sample_rate:.2f}s")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar áudio: {str(e)}")

    def stop_playback(self):
        """Interrompe qualquer reprodução em andamento"""
        try:
            sd.stop()
            self.status_var.set("Reprodução interrompida.")
        except Exception as e:
            self.status_var.set(f"Erro ao parar reprodução: {str(e)}")

    def play_track(self, track_index, audio_type):
        """Reproduz áudio de um track"""
        try:
            sd.stop()

            track = self.audio_tracks[track_index]

            if audio_type == 'original' and track.original is not None:
                sd.play(track.original, track.sample_rate)
                self.status_var.set(f"Reproduzindo {track.name} original ({len(track.original)/track.sample_rate:.1f}s)...")
            elif audio_type == 'synthesized' and track.synthesized is not None:
                sd.play(track.synthesized, track.sample_rate)
                self.status_var.set(f"Reproduzindo {track.name} sintetizado ({len(track.synthesized)/track.sample_rate:.1f}s)...")

        except Exception as e:
            self.status_var.set(f"Erro na reprodução: {str(e)}")

    def save_track(self, track_index):
        """Salva áudio sintetizado de um track"""
        track = self.audio_tracks[track_index]

        if track.synthesized is None:
            messagebox.showwarning("Aviso", f"Processe {track.name} primeiro!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("Arquivos WAV", "*.wav"), ("Arquivos FLAC", "*.flac")],
            initialfile=f"{track.name.lower().replace(' ', '_')}_sintetizado.wav"
        )

        if file_path:
            try:
                audio_to_save = track.synthesized

                max_val = np.max(np.abs(audio_to_save))
                if max_val > 1.0:
                    audio_to_save = audio_to_save / max_val * 0.99

                audio_int16 = (audio_to_save * 32767).astype(np.int16)
                sf.write(file_path, audio_int16, track.sample_rate)

                filename = os.path.basename(file_path)
                self.status_var.set(f"{track.name} salvo: {filename}")

                messagebox.showinfo("Sucesso",
                                  f"{track.name} salvo com sucesso!\n"
                                  f"Arquivo: {filename}\n"
                                  f"Duração: {len(audio_to_save)/track.sample_rate:.2f}s\n"
                                  f"Parciais usadas: {self.num_partials}")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar áudio: {str(e)}")

    def monitor_queue(self):
        """Monitora a fila de processamento"""
        try:
            while True:
                msg_type, *data = self.audio_queue.get_nowait()

                if msg_type == 'progress':
                    track_index, progress = data
                    track = self.audio_tracks[track_index]
                    track.status_var.set(f"Processando... {progress:.0f}%")

                elif msg_type == 'done':
                    track_index, audio = data
                    track = self.audio_tracks[track_index]
                    track.processing = False
                    track.status_var.set("✓ Pronto")
                    track.status_label.config(fg=self.colors['success'])

                    track.process_btn.config(state='normal')
                    self.plot_track_waveform(track_index, is_original=False)

                    #  Habilita botões de reprodução e salvar
                    if hasattr(track, 'play_synth_btn'):
                        track.play_synth_btn.config(state='normal')
                    if hasattr(track, 'save_btn'):
                        track.save_btn.config(state='normal')

                    self.status_var.set(f"{track.name} processado com {self.num_partials} parciais!")

                elif msg_type == 'enable_buttons':
                    # Mensagem explícita para habilitar botões
                    track_index = data[0]
                    track = self.audio_tracks[track_index]
                    if track.play_synth_btn:
                        track.play_synth_btn.config(state='normal')
                    if track.save_btn:
                        track.save_btn.config(state='normal')

                elif msg_type == 'interpolation_progress':
                    progress = data[0]
                    self.interp_status_var.set(f"Processando... {progress:.0f}%")

                elif msg_type == 'interpolation_done':
                    audio = data[0]
                    self.interpolated_audio = audio
                    self.interpolation_processing = False
                    self.interp_status_var.set("✓ Pronto")
                    self.interp_status_label.config(fg=self.colors['success'])
                    self.generate_btn.config(state='normal')

                    self.plot_interpolated_waveform(audio)

                    self.save_interp_btn.config(state='normal')
                    self.play_interp_btn.config(state='normal')

                    duration = len(audio) / self.audio_tracks[0].sample_rate
                    self.status_var.set(f"Interpolação gerada! Duração: {duration:.2f}s")

                elif msg_type == 'error':
                    track_index, error_msg = data
                    if track_index >= 0:
                        track = self.audio_tracks[track_index]
                        track.processing = False
                        track.status_var.set("✗ Erro")
                        track.status_label.config(fg=self.colors['error'])
                        track.process_btn.config(state='normal')

                    if track_index == -1:  # Erro na interpolação
                        self.interpolation_processing = False
                        self.interp_status_var.set("✗ Erro")
                        self.interp_status_label.config(fg=self.colors['error'])
                        self.generate_btn.config(state='normal')

                    self.status_var.set(f"Erro: {error_msg}")
                    messagebox.showerror("Erro", f"Erro: {error_msg}")

                elif msg_type == 'warning':
                    warning_msg = data[0]
                    # Mostra aviso na interface
                    self.status_var.set(f"Aviso: Taxas de amostragem diferentes")
                    # Opcional: mostra messagebox
                    messagebox.showwarning("Aviso de Amostragem", warning_msg)

        except queue.Empty:
            pass

        self.root.after(100, self.monitor_queue)


# ================= PONTO DE ENTRADA =================
def main():
    root = tk.Tk()
    app = AudioInterpolatorApp(root)
    
    def on_closing():
        try:
            sd.stop()
        except:
            pass
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
