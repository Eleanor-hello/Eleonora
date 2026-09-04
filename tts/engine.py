# -*- coding: utf-8 -*-
"""TtsEngine — озвучка ответов: Silero v5.5 + winsound.

Как работает:
  speak(text) мгновенно возвращается: синтез и воспроизведение идут в
  фоновом потоке по чанкам (chunker). Воспроизведение блокирующее
  PlaySound(SND_MEMORY) — но блокирует только воркер, не UI.

  Новый speak() или stop() обрывает текущее воспроизведение (SND_PURGE)
  и помечает старый синтез устаревшим через token-счётчик: недосинтезированные
  чанки просто не проигрываются.

Почему winsound: stdlib без зависимостей; SND_MEMORY играет WAV из памяти
без temp-файлов; SND_PURGE штатно прерывает воспроизведение из другого потока.
SND_ASYNC с SND_MEMORY запрещён в winsound — поэтому блокирующий вариант.
"""

import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import torch

try:
    import winsound
except ImportError:
    winsound = None  # не-Windows: движок недоступен

from tts.chunker import chunk_for_tts
from tts.preprocessor import apply_stress, yoficate
from tts.sanitizer import sanitize_for_tts

logger = logging.getLogger(__name__)


class TtsEngine:
    """Silero TTS + winsound. speak()/stop() потокобезопасны."""

    def __init__(
        self,
        model_path: Path,
        voice: str,
        sample_rate: int = 48000,
        device: str = "cpu",
        max_chunk_chars: int = 800,
    ):
        if winsound is None:
            raise RuntimeError("TtsEngine требует Windows (winsound)")
        if not model_path.is_file():
            raise FileNotFoundError(f"Нет TTS-модели: {model_path}")

        self.voice = voice
        self.sample_rate = sample_rate
        self.max_chunk_chars = max_chunk_chars

        logger.info(f"Загрузка Silero TTS: {model_path.name}...")
        # Читаем сами и отдаём байты: torch на Windows не открывает
        # не-ASCII пути (например, папку проекта с кириллицей).
        importer = torch.package.PackageImporter(io.BytesIO(model_path.read_bytes()))
        self._model = importer.load_pickle("tts_models", "model")
        self._model.to(torch.device(device))

        self._token_lock = threading.Lock()
        self._current_token = 0
        self._synth_lock = threading.Lock()   # Silero не любит параллельные apply_tts
        self._speaking = threading.Event()
        logger.info(f"TTS готов: voice={voice}, {sample_rate}Hz, {device}")

    # ── публичный API ──

    def speak(self, text: str) -> None:
        """Озвучить текст асинхронно, прервав текущее воспроизведение."""
        self.stop()
        with self._token_lock:
            self._current_token += 1
            token = self._current_token
        threading.Thread(
            target=self._worker, args=(text, token), daemon=True, name=f"TTS#{token}"
        ).start()

    def stop(self) -> None:
        """Оборвать текущее воспроизведение."""
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # ── внутренности ──

    def _stale(self, token: int) -> bool:
        with self._token_lock:
            return token != self._current_token

    def _worker(self, text: str, token: int) -> None:
        self._speaking.set()
        try:
            clean = sanitize_for_tts(text)
            clean = yoficate(clean)
            clean = apply_stress(clean)   # строго после ёфикации
            chunks = chunk_for_tts(clean, max_chars=self.max_chunk_chars)

            for chunk in chunks:
                if self._stale(token):
                    return
                with self._synth_lock:
                    if self._stale(token):
                        return
                    audio = self._model.apply_tts(
                        text=chunk,
                        speaker=self.voice,
                        sample_rate=self.sample_rate,
                        put_accent=True,
                        put_stress_homo=True,
                        put_yo=False,
                        put_yo_homo=False,
                    )
                wav = self._to_wav(audio)
                if self._stale(token):
                    return
                # Блокирует до конца чанка; stop() из другого потока прервёт.
                winsound.PlaySound(wav, winsound.SND_MEMORY)
        except Exception:
            logger.exception(f"TTS #{token} упал")
        finally:
            self._speaking.clear()

    def _to_wav(self, audio: torch.Tensor) -> bytes:
        """Tensor [-1..1] → mono 16-bit WAV в памяти."""
        pcm = (
            np.clip(audio.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
            * 32767.0
        ).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
