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
    dynamic_profile: Optional[np.ndarray] = None
    gain: float = DEFAULT_GAIN
    file_path: str = ""
    color: str = ""
    processing: bool = False
    
    def __post_init__(self):
        if self.grains is None:
            self.grains = []
        if self.synthesized_grains is None:
            self.synthesized_grains = []
        if self.grain_envelopes is None:
            self.grain_envelopes = []
        if self.dynamic_profile is None:
            self.dynamic_profile = np.array([])

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
        self.root.title("Dual Audio Interpolator - Síntese Aditiva")
        self.root.geometry("1400x1100")
        
        # Dados dos áudios
        self.audio_tracks = [
            AudioTrack(name="Áudio 1", color="#4a6fa5"),
            AudioTrack(name="Áudio 2", color="#ff6b6b")
        ]
        self.grain_processor = GrainProcessor(NUM_PARTIALS)
        
        # Configurações compartilhadas
        self.grain_duration = 40  # ms
        self.overlap = OVERLAP
        self.num_partials = NUM_PARTIALS
        self.dynamic_preservation = DYNAMIC_RANGE_PRESERVATION
        self.normalize_audio = True
        
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
            'success': '#a6e3a1',
            'warning': '#f9e2af',
            'error': '#f38ba8',
            'border': '#45475a'
        }
        
        # Configurar fonte
        self.fonts = {
            'title': ('Segoe UI', 16, 'bold'),
            'subtitle': ('Segoe UI', 12, 'bold'),
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
        # Configurar estilo
        self.setup_styles()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🎛️ Dual Audio Interpolator",
            font=self.fonts['title'],
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Síntese Aditiva de Dois Áudios Simultâneos",
            font=self.fonts['subtitle'],
            fg=self.colors['fg_light'],
            bg=self.colors['bg']
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Container principal com layout em grid
        container = tk.Frame(main_frame, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Coluna esquerda - Controles
        left_column = tk.Frame(container, bg=self.colors['bg'])
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Scrollbar para a coluna esquerda
        left_canvas = tk.Canvas(left_column, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_column, orient="vertical", command=left_canvas.yview)
        scrollable_frame = tk.Frame(left_canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configuração do mouse wheel
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Card de configurações globais
        config_card = self.create_card(scrollable_frame, "⚙️ Configurações Globais")
        
        # Duração do grão
        grain_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        grain_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            grain_frame,
            text="Duração do Grão:",
            font=self.fonts['body'],
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(anchor=tk.W)
        
        self.grain_var = tk.IntVar(value=self.grain_duration)
        grain_scale = ttk.Scale(
            grain_frame,
            from_=10,
            to=100,
            variable=self.grain_var,
            command=self.on_grain_change,
            style='Modern.Horizontal.TScale'
        )
        grain_scale.pack(fill=tk.X, pady=(5, 0))
        
        self.grain_label = tk.Label(
            grain_frame,
            text=f"{self.grain_duration} ms",
            font=self.fonts['small'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg']
        )
        self.grain_label.pack(anchor=tk.E)
        
        # Overlap
        overlap_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        overlap_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            overlap_frame,
            text="Overlap:",
            font=self.fonts['body'],
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(anchor=tk.W)
        
        self.overlap_var = tk.DoubleVar(value=self.overlap * 100)
        overlap_scale = ttk.Scale(
            overlap_frame,
            from_=10,
            to=90,
            variable=self.overlap_var,
            command=self.on_overlap_change,
            style='Modern.Horizontal.TScale'
        )
        overlap_scale.pack(fill=tk.X, pady=(5, 0))
        
        self.overlap_label = tk.Label(
            overlap_frame,
            text=f"{self.overlap*100:.0f}%",
            font=self.fonts['small'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg']
        )
        self.overlap_label.pack(anchor=tk.E)
        
        # Número de Parciais
        partials_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        partials_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            partials_frame,
            text="Parciais de Síntese:",
            font=self.fonts['body'],
            fg=self.colors['fg'],
            bg=self.colors['card_bg']
        ).pack(anchor=tk.W)
        
        # Frame para slider e entry
        partials_control_frame = tk.Frame(partials_frame, bg=self.colors['card_bg'])
        partials_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Slider para número de parciais
        self.partials_var = tk.IntVar(value=self.num_partials)
        partials_slider = ttk.Scale(
            partials_control_frame,
            from_=8,
            to=2048,
            variable=self.partials_var,
            command=self.on_partials_change,
            style='Modern.Horizontal.TScale'
        )
        partials_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Entry para valor numérico
        partials_entry_frame = tk.Frame(partials_control_frame, bg=self.colors['card_bg'])
        partials_entry_frame.pack(side=tk.RIGHT)
        
        partials_entry = ttk.Entry(
            partials_entry_frame,
            textvariable=self.partials_var,
            width=8,
            font=self.fonts['body'],
            justify='center'
        )
        partials_entry.pack(side=tk.LEFT)
        partials_entry.bind('<Return>', lambda e: self.on_partials_entry_change())
        
        # Label descritivo
        self.partials_label = tk.Label(
            partials_frame,
            text=f"{self.num_partials} parciais",
            font=self.fonts['small'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg']
        )
        self.partials_label.pack(anchor=tk.E)
        
        # Frame para botões rápidos de parciais
        partials_quick_frame = tk.Frame(partials_frame, bg=self.colors['card_bg'])
        partials_quick_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Botões rápidos para valores comuns
        quick_partials = [32, 64, 128, 256, 512, 1024]
        for num in quick_partials:
            btn = ttk.Button(
                partials_quick_frame,
                text=f"{num}",
                command=lambda n=num: self.set_partials(n),
                style='Secondary.TButton',
                width=4
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Checkboxes de processamento
        options_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        options_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.dynamic_var = tk.BooleanVar(value=self.dynamic_preservation)
        dynamic_check = ttk.Checkbutton(
            options_frame,
            text="Preservar Dinâmica",
            variable=self.dynamic_var,
            style='Modern.TCheckbutton'
        )
        dynamic_check.pack(anchor=tk.W, pady=(0, 5))
        
        self.normalize_var = tk.BooleanVar(value=self.normalize_audio)
        normalize_check = ttk.Checkbutton(
            options_frame,
            text="Normalização Automática",
            variable=self.normalize_var,
            style='Modern.TCheckbutton'
        )
        normalize_check.pack(anchor=tk.W)
        
        # Botão processar todos
        process_all_btn = ttk.Button(
            config_card,
            text="🔄 Processar Ambos",
            command=self.process_all_tracks,
            style='Accent.TButton'
        )
        process_all_btn.pack(fill=tk.X, pady=(15, 0))
        
        # Info de configurações
        info_frame = tk.Frame(config_card, bg=self.colors['card_bg'])
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_text = ("🔧 Dica: Mais parciais = mais detalhe espectral\n"
                    "Menos parciais = processamento mais rápido")
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=self.fonts['small'],
            fg=self.colors['fg_light'],
            bg=self.colors['card_bg'],
            justify=tk.LEFT,
            wraplength=280
        )
        info_label.pack(anchor=tk.W)
        
        # Cards para cada áudio
        for i, track in enumerate(self.audio_tracks):
            track_card = self.create_card(
                scrollable_frame, 
                f"🎵 {track.name}",
                top_margin=20
            )
            
            # Frame para controles do áudio
            track_controls = tk.Frame(track_card, bg=self.colors['card_bg'])
            track_controls.pack(fill=tk.X, pady=(5, 0))
            
            # Botão carregar
            load_btn = ttk.Button(
                track_controls,
                text="📂 Carregar",
                command=lambda idx=i: self.load_audio(idx),
                style='Secondary.TButton',
                width=10
            )
            load_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            # Botão processar individual
            self.audio_tracks[i].process_btn = ttk.Button(
                track_controls,
                text="⚡ Processar",
                command=lambda idx=i: self.process_track(idx),
                style='Primary.TButton',
                width=10
            )
            self.audio_tracks[i].process_btn.pack(side=tk.LEFT)
            
            # Info do arquivo
            self.create_info_display(track_card, i)
            
            # Controle de ganho individual
            self.create_gain_control(track_card, i)
            
            # Botões de ação
            self.create_action_buttons(track_card, i)
        
        # Coluna direita - Visualizações
        right_column = tk.Frame(container, bg=self.colors['bg'])
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Container para gráficos
        graphs_container = tk.Frame(right_column, bg=self.colors['bg'])
        graphs_container.pack(fill=tk.BOTH, expand=True)
        
        # Criar figura matplotlib
        self.fig = Figure(figsize=(12, 10), facecolor=self.matplotlib_colors['bg'])
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # 2x2 grid de gráficos
        self.ax_original = []
        self.ax_synthesized = []
        
        for i in range(2):
            # Gráfico original
            ax_orig = self.fig.add_subplot(2, 2, i*2 + 1)
            ax_orig.set_title(f"{self.audio_tracks[i].name} - Original", 
                            color=self.matplotlib_colors['fg'],
                            fontsize=11,
                            pad=10)
            self.setup_axes(ax_orig)
            self.ax_original.append(ax_orig)
            
            # Gráfico sintetizado
            ax_synth = self.fig.add_subplot(2, 2, i*2 + 2)
            ax_synth.set_title(f"{self.audio_tracks[i].name} - Sintetizado", 
                             color=self.matplotlib_colors['fg'],
                             fontsize=11,
                             pad=10)
            self.setup_axes(ax_synth)
            self.ax_synthesized.append(ax_synth)
        
        # Canvas para gráficos
        self.canvas = FigureCanvasTkAgg(self.fig, graphs_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barra de status
        self.status_var = tk.StringVar(value="Pronto para carregar áudios")
        status_bar = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=self.fonts['small'],
            fg=self.colors['fg_light'],
            bg=self.colors['card_bg'],
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        
        # Iniciar monitoramento da fila
        self.root.after(100, self.monitor_queue)
    
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
        
        # Botão primário
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground=self.colors['bg'],
                       borderwidth=0,
                       focusthickness=0,
                       padding=8,
                       font=self.fonts['body'])
        style.map('Primary.TButton',
                 background=[('active', '#7aa3e0'), ('disabled', '#4a5a70')])
        
        # Botão secundário
        style.configure('Secondary.TButton',
                       background=self.colors['secondary'],
                       foreground=self.colors['fg'],
                       borderwidth=0,
                       focusthickness=0,
                       padding=8,
                       font=self.fonts['body'])
        style.map('Secondary.TButton',
                 background=[('active', '#6a6b80'), ('disabled', '#4a5a70')])
        
        # Botão de destaque
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['bg'],
                       borderwidth=0,
                       focusthickness=0,
                       padding=10,
                       font=self.fonts['body'])
        style.map('Accent.TButton',
                 background=[('active', '#e5b2d7'), ('disabled', '#4a5a70')])
        
        # Slider moderno
        style.configure('Modern.Horizontal.TScale',
                       background=self.colors['card_bg'])
        
        # Checkbox moderno
        style.configure('Modern.TCheckbutton',
                       background=self.colors['card_bg'],
                       foreground=self.colors['fg'],
                       font=self.fonts['body'])
        style.map('Modern.TCheckbutton',
                 background=[('active', self.colors['card_bg'])])
    
    def create_card(self, parent, title, top_margin=0):
        """Cria um card moderno com bordas arredondadas"""
        card = tk.Frame(parent, bg=self.colors['card_bg'], relief=tk.FLAT)
        card.pack(fill=tk.X, pady=(top_margin, 0))
        
        # Título do card
        title_label = tk.Label(
            card,
            text=title,
            font=self.fonts['subtitle'],
            fg=self.colors['primary'],
            bg=self.colors['card_bg'],
            pady=10
        )
        title_label.pack(anchor=tk.W, padx=15)
        
        # Linha divisória
        separator = tk.Frame(card, height=1, bg=self.colors['border'])
        separator.pack(fill=tk.X, padx=15)
        
        # Conteúdo do card
        content = tk.Frame(card, bg=self.colors['card_bg'], padx=15, pady=15)
        content.pack(fill=tk.X)
        
        return content
    
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
            wraplength=250,
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
            width=10
        )
        play_orig_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Botão reproduzir sintetizado
        play_synth_btn = ttk.Button(
            action_frame,
            text="▶ Sintetizado",
            command=lambda idx=track_index: self.play_track(idx, 'synthesized'),
            style='Primary.TButton',
            width=10
        )
        play_synth_btn.pack(side=tk.LEFT)
        
        # Botão salvar
        save_btn = ttk.Button(
            action_frame,
            text="💾 Salvar",
            command=lambda idx=track_index: self.save_track(idx),
            style='Secondary.TButton',
            width=8
        )
        save_btn.pack(side=tk.RIGHT)
    
    def setup_axes(self, ax):
        """Configura eixos do matplotlib com tema escuro"""
        ax.set_facecolor(self.matplotlib_colors['bg'])
        ax.tick_params(colors=self.matplotlib_colors['fg'], labelsize=8)
        
        # Configura spines (bordas) mais finas
        for spine in ax.spines.values():
            spine.set_color(self.matplotlib_colors['spines'])
            spine.set_linewidth(0.8)
        
        # Grid mais sutil
        ax.grid(True, color=self.matplotlib_colors['grid'], alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Labels com fonte menor
        ax.set_xlabel('Tempo (s)', color=self.matplotlib_colors['fg'], fontsize=9)
        ax.set_ylabel('Amplitude', color=self.matplotlib_colors['fg'], fontsize=9)
        
        # Configura ticks do eixo Y mais compactos
        ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8)
        
        # Para estado vazio, configuramos limites razoáveis
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)
    
    def on_grain_change(self, value):
        """Atualiza quando duração do grão muda"""
        self.grain_duration = int(float(value))
        self.grain_label.config(text=f"{self.grain_duration} ms")
    
    def on_overlap_change(self, value):
        """Atualiza quando overlap muda"""
        self.overlap = float(value) / 100.0
        self.overlap_label.config(text=f"{float(value):.0f}%")
    
    def on_partials_change(self, value):
        """Atualiza quando número de parciais muda"""
        try:
            num_partials = int(float(value))
            self.num_partials = max(8, min(num_partials, 2048))
            self.partials_var.set(self.num_partials)
            self.partials_label.config(text=f"{self.num_partials} parciais")
            
            # Atualiza o processador
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
            
            # Atualiza informações
            filename = os.path.basename(file_path)
            duration = len(audio) / sample_rate
            track.info_var.set(f"{filename}\n{duration:.2f}s @ {sample_rate}Hz")
            
            # Plota forma de onda original
            self.plot_track_waveform(track_index, is_original=True)
            
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
            
        except Exception as e:
            self.audio_queue.put(('error', track_index, str(e)))
    
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
            
            # Para áudios muito longos (>30 segundos), limitamos a visualização
            # para manter a performance, mas mostramos uma mensagem
            display_duration = total_duration
            if total_duration > 30:
                display_duration = 30  # Limita a 30 segundos para visualização
                show_warning = True
            else:
                show_warning = False
            
            # Cria array de tempo para visualização
            if display_duration < total_duration:
                # Corta o áudio para visualização
                display_samples = int(display_duration * track.sample_rate)
                audio_display = audio[:display_samples]
                time_display = np.linspace(0, display_duration, display_samples)
            else:
                # Usa todo o áudio
                audio_display = audio
                time_display = np.linspace(0, total_duration, len(audio))
            
            # Encontra amplitude máxima para ajustar eixo Y
            max_abs = np.max(np.abs(audio_display))
            
            # Plota forma de onda
            linewidth = 0.5 if len(audio_display) > 20000 else 0.8
            ax.plot(time_display, audio_display, color=track.color, 
                   linewidth=linewidth, alpha=0.8)
            
            # Linha de referência zero
            ax.axhline(y=0, color=self.matplotlib_colors['fg'], 
                      linestyle='-', alpha=0.2, linewidth=0.5)
            
            # Linhas de clipping se amplitude exceder 0.95
            if max_abs > 0.95:
                ax.axhline(y=1.0, color=self.colors['error'], 
                          linestyle='--', alpha=0.4, linewidth=0.5)
                ax.axhline(y=-1.0, color=self.colors['error'], 
                          linestyle='--', alpha=0.4, linewidth=0.5)
            
            # Configura limites dos eixos
            ax.set_xlim(0, display_duration)
            
            # Ajusta eixo Y baseado na amplitude real
            y_margin = 0.05
            y_max = min(1.0, max_abs + y_margin)
            ax.set_ylim(-y_max, y_max)
            
            # Configura ticks do eixo X
            if display_duration > 10:
                # Para áudios longos: ticks a cada 5 segundos
                x_ticks = np.arange(0, display_duration + 1, 5)
            elif display_duration > 2:
                # Para áudios médios: ticks a cada segundo
                x_ticks = np.arange(0, display_duration + 0.5, 1)
            else:
                # Para áudios curtos: ticks a cada 0.5 segundos
                x_ticks = np.arange(0, display_duration + 0.25, 0.5)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{t:.0f}' if t == int(t) else f'{t:.1f}' for t in x_ticks], fontsize=8)
            
            # Configura ticks do eixo Y
            y_ticks = np.linspace(-y_max, y_max, 5)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=8)
            
            # Adiciona mensagem se áudio foi truncado para visualização
            if show_warning:
                ax.text(0.98, 0.02, f"Visualizando primeiros {display_duration}s",
                       transform=ax.transAxes,
                       color=self.colors['warning'],
                       fontsize=7,
                       alpha=0.7,
                       horizontalalignment='right',
                       verticalalignment='bottom')
            
        else:
            ax.text(0.5, 0.5, "Nenhum áudio\ncarregado" if is_original else "Não processado",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color=self.matplotlib_colors['fg'],
                   fontsize=10,
                   alpha=0.5)
            
            # Configurações padrão para gráficos vazios
            ax.set_xlim(0, 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=8)
            ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
            ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8)
        
        # Atualiza título com duração
        if audio is not None and len(audio) > 0:
            total_duration = len(audio) / track.sample_rate
            duration_text = f" ({total_duration:.2f}s)"
        else:
            duration_text = ""
        
        ax.set_title(f"{track.name}{title_suffix}{duration_text}", 
                    color=self.matplotlib_colors['fg'],
                    fontsize=11,
                    pad=8)
        
        # Reconfigura estilo básico dos eixos
        ax.set_facecolor(self.matplotlib_colors['bg'])
        ax.tick_params(colors=self.matplotlib_colors['fg'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.matplotlib_colors['spines'])
            spine.set_linewidth(0.8)
        ax.grid(True, color=self.matplotlib_colors['grid'], alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Tempo (s)', color=self.matplotlib_colors['fg'], fontsize=9)
        ax.set_ylabel('Amplitude', color=self.matplotlib_colors['fg'], fontsize=9)
        
        # Atualiza canvas
        self.canvas.draw()
    
    def play_track(self, track_index, audio_type):
        """Reproduz áudio de um track"""
        try:
            sd.stop()  # Para qualquer reprodução anterior
            
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
                
                # Garante que está no range [-1, 1]
                max_val = np.max(np.abs(audio_to_save))
                if max_val > 1.0:
                    audio_to_save = audio_to_save / max_val * 0.99
                
                # Converte para int16
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
                    
                    # Reabilita botão de processar
                    track.process_btn.config(state='normal')
                    
                    # Atualiza gráfico
                    self.plot_track_waveform(track_index, is_original=False)
                    
                    # Atualiza status
                    self.status_var.set(f"{track.name} processado com {self.num_partials} parciais!")
                    
                elif msg_type == 'error':
                    track_index, error_msg = data
                    track = self.audio_tracks[track_index]
                    track.processing = False
                    track.status_var.set("✗ Erro")
                    track.status_label.config(fg=self.colors['error'])
                    
                    # Reabilita botão de processar
                    track.process_btn.config(state='normal')
                    
                    self.status_var.set(f"Erro em {track.name}: {error_msg}")
                    messagebox.showerror("Erro", f"Erro ao processar {track.name}: {error_msg}")
                    
        except queue.Empty:
            pass
        
        # Agenda próxima verificação
        self.root.after(100, self.monitor_queue)

# ================= PONTO DE ENTRADA =================
def main():
    root = tk.Tk()
    app = AudioInterpolatorApp(root)
    
    # Configuração para fechar corretamente
    def on_closing():
        try:
            sd.stop()
        except:
            pass
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Centraliza a janela
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()