"""
Monitor de rendiment: temps d'entrenament, memòria i overhead de privadesa.

DISSENY (mínima intrusió)
-------------------------
Aquest mòdul és autònom i no depèn de cap altre del projecte. Exposa un
recol·lector lleuger (PerfMonitor) que acumula durades cronometrades i el pic
de memòria del procés, i una funció per desar-ho tot en un fitxer
(performance.txt + performance.csv) dins del directori de resultats.

MÈTRIQUES QUE ES CAPTUREN
-------------------------
- Temps d'entrenament del baseline (sense DP), per run i mitjana.
- Temps de perturbació DP (aplicar el mecanisme al graf), per (mecanisme,
  epsilon), que és el cost específic de la privadesa a la fase d'entrada.
- Temps d'entrenament sota DP, per (mecanisme, epsilon).
- Overhead de privadesa: temps de perturbació + (entrenament DP - baseline).
  La part de perturbació és el cost net afegit per la privadesa; el temps
  d'entrenament sol ser comparable amb i sense DP perquè l'arquitectura no
  canvia, però es reporta per si hi ha diferències.
- Pic de memòria RAM del procés (RSS) durant tota l'execució.

DEPENDÈNCIES
------------
- time (estàndard).
- resource (estàndard a Linux/macOS) per al pic de RSS; si no està disponible
  (p. ex. Windows), s'intenta psutil i, si tampoc, es reporta 'N/A' sense
  trencar res.
"""

import os
import csv
import time
from datetime import datetime


def _peak_rss_mb():
    """
    Retorna el pic de memòria resident (RSS) del procés en MB, o None si no es
    pot determinar a la plataforma actual.
    """
    # Opció 1: resource (Linux/macOS)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # A Linux ru_maxrss és en KB; a macOS, en bytes. S'heuristitza:
        # si el valor és molt gran (> 1e9) probablement són bytes.
        if ru > 1_000_000_000:
            return ru / (1024 ** 2)        # bytes -> MB
        return ru / 1024                   # KB -> MB
    except Exception:
        pass
    # Opció 2: psutil (multiplataforma, inclòs Windows)
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


class PerfMonitor:
    """
    Recol·lector lleuger de durades i memòria. Ús típic:

        perf = PerfMonitor()
        with perf.timer('baseline_train'):
            ... entrenar baseline ...
        with perf.timer('dp_perturb', mechanism='rr_dp', epsilon=1.0):
            ... perturbar graf ...
        with perf.timer('dp_train', mechanism='rr_dp', epsilon=1.0):
            ... entrenar sota DP ...
        perf.snapshot_memory()
        save_performance(results_dir, perf)
    """

    def __init__(self):
        self.records = []          # llista de dicts: {tag, mechanism, epsilon, seconds}
        self.peak_rss_mb = None
        self._t0 = time.perf_counter()

    def timer(self, tag, mechanism=None, epsilon=None):
        return _Timer(self, tag, mechanism, epsilon)

    def _add(self, tag, mechanism, epsilon, seconds):
        self.records.append({
            'tag': tag,
            'mechanism': mechanism if mechanism is not None else '',
            'epsilon': epsilon if epsilon is not None else '',
            'seconds': seconds,
        })

    def snapshot_memory(self):
        """Actualitza el pic de RAM observat fins ara."""
        rss = _peak_rss_mb()
        if rss is not None:
            self.peak_rss_mb = rss if self.peak_rss_mb is None else max(self.peak_rss_mb, rss)

    # --- Agregats útils per al resum ---
    def total_seconds(self, tag=None):
        return sum(r['seconds'] for r in self.records
                   if tag is None or r['tag'] == tag)

    def mean_seconds(self, tag):
        vals = [r['seconds'] for r in self.records if r['tag'] == tag]
        return sum(vals) / len(vals) if vals else 0.0

    def wall_clock_seconds(self):
        return time.perf_counter() - self._t0


class _Timer:
    def __init__(self, monitor, tag, mechanism, epsilon):
        self.monitor = monitor
        self.tag = tag
        self.mechanism = mechanism
        self.epsilon = epsilon

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self.monitor._add(self.tag, self.mechanism, self.epsilon, elapsed)
        self.monitor.snapshot_memory()
        return False  # no silenciar excepcions


def save_performance(results_dir, perf, dataset_name=None):
    """
    Desa les mètriques de rendiment a performance.txt (llegible) i
    performance.csv (per processar). No falla mai el pipeline: si hi ha
    qualsevol error d'E/S, l'informa per stdout i continua.
    """
    try:
        os.makedirs(results_dir, exist_ok=True)

        # --- Agregats ---
        baseline_mean = perf.mean_seconds('baseline_train')
        dp_train_mean = perf.mean_seconds('dp_train')
        perturb_total = perf.total_seconds('dp_perturb')
        perturb_mean = perf.mean_seconds('dp_perturb')
        # Overhead net de privadesa per run (perturbació + diferència d'entrenament)
        train_overhead = max(0.0, dp_train_mean - baseline_mean)
        privacy_overhead_mean = perturb_mean + train_overhead

        txt = os.path.join(results_dir, 'performance.txt')
        with open(txt, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PERFORMANCE & RESOURCE USAGE\n")
            f.write("=" * 70 + "\n\n")
            if dataset_name:
                f.write(f"Dataset:                      {dataset_name}\n")
            f.write(f"Timestamp:                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total wall-clock time:        {perf.wall_clock_seconds():.1f} s\n")
            rss = f"{perf.peak_rss_mb:.1f} MB" if perf.peak_rss_mb is not None else "N/A"
            f.write(f"Peak RAM (RSS):               {rss}\n\n")

            f.write("Training time (per run, mean)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Baseline (no DP):           {baseline_mean:.3f} s\n")
            f.write(f"  Under DP:                   {dp_train_mean:.3f} s\n\n")

            f.write("Privacy cost (per run, mean)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  DP perturbation time:       {perturb_mean:.3f} s\n")
            f.write(f"  Extra training time:        {train_overhead:.3f} s\n")
            f.write(f"  Total privacy overhead:     {privacy_overhead_mean:.3f} s\n")
            if baseline_mean > 0:
                pct = 100.0 * privacy_overhead_mean / baseline_mean
                f.write(f"  Overhead vs baseline:       {pct:.1f}%\n")
            f.write(f"\n  Total DP perturbation time (all runs): {perturb_total:.1f} s\n")

        # --- CSV detallat (una fila per durada cronometrada) ---
        csv_path = os.path.join(results_dir, 'performance.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['tag', 'mechanism', 'epsilon', 'seconds'])
            for r in perf.records:
                w.writerow([r['tag'], r['mechanism'], r['epsilon'], f"{r['seconds']:.6f}"])

        print(f"  - performance.txt     (temps, RAM, overhead de privadesa)")
        print(f"  - performance.csv     (durades detallades)")
    except Exception as e:  # pragma: no cover
        print(f"[perf_monitor] No s'han pogut desar les mètriques de rendiment: {e}")
