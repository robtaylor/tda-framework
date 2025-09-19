"""Visualization tools for signal plotting and performance analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

from ..core.circuit import Circuit
from ..core.signals import Signal
from .timing import CircuitTimingReport, TimingResult


class SignalTracer:
    """Records and visualizes signal waveforms."""

    def __init__(self) -> None:
        self._traces: Dict[str, List[Tuple[float, float]]] = {}

    def add_trace_point(self, signal_name: str, timestamp: float, value: float) -> None:
        """Add a trace point for a signal."""
        if signal_name not in self._traces:
            self._traces[signal_name] = []
        self._traces[signal_name].append((timestamp, value))

    def trace_signal(self, signal: Signal) -> None:
        """Add all historical data from a signal."""
        history = signal.get_history()
        self._traces[signal.name] = [
            (float(sample.timestamp), sample.value) for sample in history
        ]

    def plot_signals(
        self,
        signal_names: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        title: str = "Signal Waveforms",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot signal waveforms."""
        if signal_names is None:
            signal_names = list(self._traces.keys())

        fig, ax = plt.subplots(figsize=(12, 6))

        for signal_name in signal_names:
            if signal_name in self._traces:
                times, values = zip(*self._traces[signal_name])
                times = np.array(times)
                values = np.array(values)

                # Apply time range filter if specified
                if time_range:
                    mask = (times >= time_range[0]) & (times <= time_range[1])
                    times = times[mask]
                    values = values[mask]

                ax.plot(times, values, label=signal_name, linewidth=2)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if time_range:
            ax.set_xlim(time_range)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_frequency_spectrum(
        self,
        signal_name: str,
        sampling_rate: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot frequency spectrum of a signal using FFT."""
        if signal_name not in self._traces:
            raise ValueError(f"Signal '{signal_name}' not found in traces")

        times, values = zip(*self._traces[signal_name])
        times = np.array(times)
        values = np.array(values)

        # Estimate sampling rate if not provided
        if sampling_rate is None and len(times) > 1:
            dt = np.mean(np.diff(times))
            sampling_rate = 1.0 / dt

        # Perform FFT
        fft_values = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values), d=1.0/sampling_rate)

        # Plot positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_values[:len(fft_values)//2])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(positive_freqs[1:], positive_fft[1:])  # Skip DC component
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Frequency Spectrum: {signal_name}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_data(self, signal_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Export signal data as numpy arrays."""
        if signal_names is None:
            signal_names = list(self._traces.keys())

        export_data = {}
        for signal_name in signal_names:
            if signal_name in self._traces:
                times, values = zip(*self._traces[signal_name])
                export_data[signal_name] = np.column_stack([times, values])

        return export_data


class TimingVisualizer:
    """Visualization tools for timing analysis results."""

    @staticmethod
    def plot_timing_comparison(
        reports: Dict[str, CircuitTimingReport],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot timing comparison between different implementations."""
        implementations = list(reports.keys())
        execution_times = [float(report.total_execution_time) for report in reports.values()]
        power_consumption = [report.total_power_consumption for report in reports.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Execution time comparison
        bars1 = ax1.bar(implementations, execution_times, color='skyblue', alpha=0.7)
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time Comparison")
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, time in zip(bars1, execution_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{time:.2e}', ha='center', va='bottom')

        # Power consumption comparison
        bars2 = ax2.bar(implementations, power_consumption, color='lightcoral', alpha=0.7)
        ax2.set_ylabel("Power Consumption (W)")
        ax2.set_title("Power Consumption Comparison")
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, power in zip(bars2, power_consumption):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{power:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_component_timing_breakdown(
        report: CircuitTimingReport,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot timing breakdown by component."""
        if not report.component_results:
            raise ValueError("No component timing data available")

        component_names = [r.component_name for r in report.component_results]
        delays = [float(r.propagation_delay) for r in report.component_results]
        powers = [r.power_consumption for r in report.component_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Propagation delay breakdown
        bars1 = ax1.barh(component_names, delays, color='lightgreen', alpha=0.7)
        ax1.set_xlabel("Propagation Delay (s)")
        ax1.set_title("Component Propagation Delays")

        # Add value labels
        for bar, delay in zip(bars1, delays):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{delay:.2e}', ha='left', va='center')

        # Power consumption breakdown
        bars2 = ax2.barh(component_names, powers, color='orange', alpha=0.7)
        ax2.set_xlabel("Power Consumption (W)")
        ax2.set_title("Component Power Consumption")

        # Add value labels
        for bar, power in zip(bars2, powers):
            ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{power:.4f}', ha='left', va='center')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_performance_metrics(
        report: CircuitTimingReport,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot key performance metrics."""
        metrics = report.performance_metrics

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Events per second
        ax1.bar(['Events/sec'], [metrics.get('events_per_second', 0)], color='skyblue')
        ax1.set_title("Simulation Performance")
        ax1.set_ylabel("Events per Second")

        # Average component delay
        ax2.bar(['Avg Delay'], [metrics.get('average_component_delay', 0)], color='lightcoral')
        ax2.set_title("Average Component Delay")
        ax2.set_ylabel("Delay (s)")

        # Power efficiency
        ax3.bar(['Power Eff.'], [metrics.get('power_efficiency', 0)], color='lightgreen')
        ax3.set_title("Power Efficiency")
        ax3.set_ylabel("Power/Time (W/s)")

        # Accuracy range
        accuracy_range = metrics.get('accuracy_range', (0, 0))
        ax4.bar(['Min Accuracy', 'Max Accuracy'], accuracy_range, color='orange')
        ax4.set_title("Accuracy Range")
        ax4.set_ylabel("Relative Error")
        ax4.set_yscale('log')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class CircuitVisualizer:
    """Visualization tools for circuit topology and connections."""

    @staticmethod
    def plot_circuit_graph(
        circuit: Circuit,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot circuit connectivity graph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required for circuit graph visualization")

        # Create graph
        G = nx.DiGraph()

        # Add nodes for components
        for comp_name in circuit._components:
            G.add_node(comp_name)

        # Add edges for connections
        for connection in circuit._connections:
            G.add_edge(connection.from_component, connection.to_component,
                      label=f"{connection.from_port}→{connection.to_port}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=3000, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrows=True, arrowsize=20, ax=ax)

        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title(f"Circuit Graph: {circuit.name}")
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def generate_timing_report_html(
        reports: Dict[str, CircuitTimingReport],
        output_path: str
    ) -> None:
        """Generate HTML report with timing analysis results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TDA Framework Timing Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .report-section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>TDA Framework Timing Analysis Report</h1>
        """

        for impl_name, report in reports.items():
            html_content += f"""
            <div class="report-section">
                <h2>Implementation: {impl_name}</h2>
                <h3>Circuit: {report.circuit_name}</h3>

                <div class="metric">
                    <strong>Total Execution Time:</strong> {report.total_execution_time:.2e} s
                </div>
                <div class="metric">
                    <strong>Critical Path Delay:</strong> {report.critical_path_delay:.2e} s
                </div>
                <div class="metric">
                    <strong>Total Power:</strong> {report.total_power_consumption:.4f} W
                </div>

                <h4>Component Details</h4>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Hardware Model</th>
                        <th>Propagation Delay (s)</th>
                        <th>Power (W)</th>
                        <th>Accuracy</th>
                    </tr>
            """

            for result in report.component_results:
                html_content += f"""
                    <tr>
                        <td>{result.component_name}</td>
                        <td>{result.hardware_model}</td>
                        <td>{result.propagation_delay:.2e}</td>
                        <td>{result.power_consumption:.4f}</td>
                        <td>{result.accuracy:.2e}</td>
                    </tr>
                """

            html_content += """
                </table>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)