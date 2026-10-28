# Bachelorarbeit: GPU Co-Scheduling

Co-Scheduling einer compute-intensiven (CI) und einer memory-intensiven (MI) Aufgabe auf einer NVIDIA-GPU.

**Leitfrage:** Lässt sich durch manuelle Steuerung der Aufgabenverteilung mehr Überlappung erreichen als durch den Hardware-Scheduler, und auf welcher Granularitätsebene?

**Ergebnis:** Statische, anwendungsseitig gesteuerte Zuweisung erreicht 96,3 % des theoretischen Optimums und ist 1,38× schneller als Hardware-Scheduling mit zwei Streams.

---

## Ergebnisse (RTX 4090)

| Phase | Zeit | Speedup |
|---|---|---|
| Baseline 0, sequenziell | 6,944 s | 1,00× |
| Ansatz 1, blockweise Fusion, hardware-verteilt | 6,020 s | 1,15× |
| Baseline 1, zwei Streams | 5,152 s | 1,35× |
| Ansatz 0, Green Contexts, 120/8 SMs | 3,830 s | 1,81× |
| Ansatz 3, Warp-Level, 4:4 | 3,788 s | 1,83× |
| **Ansatz 2, SM-zentrisch, MI/SM = 4** | **3,730 s** | **1,86×** |
| *Optimum `max(t_CI, t_MI)`* | *3,590 s* | *1,93×* |

---

## Workload

| | CI | MI |
|---|---|---|
| Operation | 4 unabhängige FMA-Ketten | Pointer-Chasing über Zufallspermutation |
| Konstruktionsziel | maximale ALU-Last, hohe ILP | Prefetching und Koaleszierung aushebeln |
| Solo-Laufzeit | 3,585 s | 3,354 s |
| Compute / Memory Throughput | 98,20 % / 0,05 % | 0,15 % / 25,63 % |
| Register/Thread | 21 | 38 |
| Warp Cycles / Instruktion | 7,67 | 14.430 |
| Dominanter Stall | Not Selected | Long Scoreboard 56 %, LG Throttle 44 % |

* Die Kernel beanspruchen komplementäre Ressourcen. Keiner sättigt die Ressource des anderen.
* MI ist latenz-, nicht bandbreitengebunden. Das ist beabsichtigt: ein bandbreitengesättigter Kernel böte keinen Spielraum.
* Kalibrierung: `Iterations_CI = 270.000`, `Iterations_MI = 100`, `NumElems = 134.217.728`, Laufzeitverhältnis etwa 1,07:1.
* `rank`/`count`: jede Aufgabe bearbeitet in jeder Phase alle Elemente, unabhängig von der zugewiesenen Threadzahl. Nur so sind die Phasen vergleichbar.

---

## Aufbau

```
benchmarks/
  include/kernels_strided.cuh   Kernel + __device__-Bodies mit rank/count
  src/baseline_0.cu             sequenziell
  src/baseline_1.cu             zwei Streams
  src/approach_0.cu             Green Contexts, Achse CI_SM_Percent
  src/approach_1.cu             blockweise Fusion + %smid-Instrumentierung
  src/approach_2.cu             SM-zentrisch, Achsen MI_Blocks_Per_SM + Iterations_CI
  src/approach_3.cu             Warp-Level, Achse CI_Warps_Per_Block
  src/scaling.cu                Skalierungskurven, Achsen NumBlocks + Permuted
```

Build: CMake + CPM + nvbench. Hardware: RTX 4090 (128 SMs, CC 8.9), Zielplattform für die Endmessung: RTX 5090.

---

## Konventionen und Fallstricke

* `CUDA_DEVICE_MAX_CONNECTIONS=2` setzen
* `cudaFuncGetAttributes()` vor jedem Launch. Sonst serialisieren Kernel in getrennten Streams scheinbar (Lazy Loading, CUDA ab 11.7)
* Grid-Größen zur Laufzeit aus `cudaDeviceGetAttribute` ableiten, keine 4090-Konstanten
* Permutation über eine `static`-Cache-Funktion, sonst 10 bis 15 s Hostzeit pro Konfiguration
* Messung mit `--timeout 120 --min-samples 15`, darunter meldet nvbench `inf%` Noise
* Durchgängig Cold-Werte, Batch liegt etwa 1 % niedriger, weil der L2 nicht geleert wird
* Baseline 1 ist reihenfolgeabhängig, gemessener Startversatz 858 ms

---

## Literatur

* Wu et al. (2015), *SM-Centric Program Transformations*, Grundlage für Ansatz 2
* Gupta et al. (2012), *A Study of Persistent Threads*, Uberkernel und Job-Queues
* Volkov, GTC 2010, Occupancy-Debatte