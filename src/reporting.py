import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.plot_utils as pu
from src.utils import get_project_results_dir


def produce_report(init_method, dataset, experiment_data):
    results_dir = get_project_results_dir()
    report_file = results_dir / dataset / f"{init_method}.csv"
    report_file.parent.mkdir(exist_ok=True)
    report_file.touch(exist_ok=True)
    with report_file.open('a') as f:
        writer = csv.DictWriter(f, experiment_data[0].keys())
        writer.writeheader()
        writer.writerows(experiment_data)


def table3():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    table = LatexTable()

    table.add_caption(
        "Resultados para comparação com a Tabela 3 do artigo original.")

    table.add_line()
    table.add_header(['CI-values (initial)'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1',
              'a2', 'a3', 'unb', 'b1', 'b2', 'dim32', 'Aver.', ]
    table.add_header(header)
    averages = []
    for init_method in init_methods:
        row = []
        row.append(init_method)
        means = []

        for dataset in header[1:-1]:
            filepath = results_dir / dataset / f"{init_method}.csv"

            if not filepath.exists():
                means.append(0.0)
                row.append('0.0')
                continue

            df = pd.read_csv(filepath)
            mean = df['ci_initial'].mean()
            means.append(mean)
            row.append(f"{mean:.1f}")

        average = np.mean(means)
        averages.append(average)
        row.append(f"\\bfseries{{{average:.1f}}}")
        table.add_row(row)

    table.add_line()
    table.add_header(['CI-values (final)'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1', 'a2',
              'a3', 'unb', 'b1', 'b2', 'dim32', 'Aver.', 'Impr.', ]
    table.add_header(header)
    for init_method, initial_average in zip(init_methods, averages):
        row = []
        row.append(init_method)
        means = []

        for dataset in header[1:-2]:
            filepath = results_dir / dataset / f"{init_method}.csv"

            if not filepath.exists():
                means.append(0.0)
                row.append('0.0')
                continue

            df = pd.read_csv(filepath)
            mean = df['ci_final'].mean()
            means.append(mean)
            row.append(f"{mean:.1f}")

        average = np.mean(means)
        row.append(f"\\bfseries{{{average:.1f}}}")
        improvement = 100.0 - (average / initial_average)*100
        row.append(f"\\bfseries{{{improvement:.0f}\\%}}")
        table.add_row(row)

    table.add_line()
    table.add_header(['Success-\%'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1', 'a2',
              'a3', 'unb', 'b1', 'b2', 'dim32', 'Aver.', 'Fails', ]
    table.add_header(header)
    for init_method, initial_average in zip(init_methods, averages):
        row = []
        row.append(init_method)
        percentages = []

        for dataset in header[1:-2]:
            filepath = results_dir / dataset / f"{init_method}.csv"

            if not filepath.exists():
                percentages.append(0)
                row.append('0\%')
                continue

            df = pd.read_csv(filepath)
            zeros = df['ci_final'].value_counts().get(0, 0)
            n_rows = df.shape[0]
            percentage = (zeros / n_rows) * 100
            percentages.append(percentage)
            row.append(f"{percentage:.0f}\\%")

        average = np.mean(percentages)
        row.append(f"\\bfseries{{{average:.0f}\\%}}")
        row.append(f"\\bfseries{{{Counter(percentages)[0]}}}")
        table.add_row(row)

    table.add_line()
    table.add_header(['Number of iterations'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1',
              'a2', 'a3', 'unb', 'b1', 'b2', 'dim32', 'Aver.', ]
    table.add_header(header)
    for init_method, initial_average in zip(init_methods, averages):
        row = []
        row.append(init_method)
        means = []

        for dataset in header[1:-1]:
            filepath = results_dir / dataset / f"{init_method}.csv"

            if not filepath.exists():
                percentages.append(0)
                row.append('0\%')
                continue

            df = pd.read_csv(filepath)
            mean = df['iterations'].mean()
            means.append(mean)
            row.append(f"{mean:.0f}")

        average = np.mean(means)
        row.append(f"\\bfseries{{{average:.0f}}}")
        table.add_row(row)

    table.add_line()
    result = table.to_str()
    print(result)
    return result


def table5():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    table = LatexTable()

    table.add_caption(
        "Resultados para comparação com a Tabela 5 do artigo original.")

    table.add_line()
    table.add_header(['CI-values'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1', 'a2',
              'a3', 'unb', 'b1', 'b2', 'dim32', 'KM', 'RKM', ]
    table.add_header(header)
    km_averages = []
    for init_method in init_methods:
        means = []
        for dataset in header[1:-2]:
            filepath = results_dir / dataset / f"{init_method}.csv"
            df = pd.read_csv(filepath)
            mean = df['ci_final'].mean()
            means.append(mean)

        average = np.mean(means)
        km_averages.append(average)

    for init_method, km_average in zip(init_methods, km_averages):
        row = []
        row.append(init_method)
        means = []

        for dataset in header[1:-2]:
            filepath = results_dir / f"{dataset}-r100" / f"{init_method}.csv"
            df = pd.read_csv(filepath)
            mean = df['ci_final'].mean()
            means.append(mean)
            row.append(f"{mean:.1f}")

        row.append(f"\\bfseries{{{km_average:.1f}}}")
        average = np.mean(means)
        row.append(f"\\bfseries{{{average:.1f}}}")
        table.add_row(row)

    table.add_line()
    table.add_header(['Success-\%'])
    header = ['Method', 's1', 's2', 's3', 's4', 'a1', 'a2',
              'a3', 'unb', 'b1', 'b2', 'dim32', 'Aver.', 'Fails', ]
    table.add_header(header)
    for init_method in init_methods:
        row = []
        row.append(init_method)
        percentages = []

        for dataset in header[1:-2]:
            filepath = results_dir / f"{dataset}-r100" / f"{init_method}.csv"
            df = pd.read_csv(filepath)
            zeros = df['ci_final'].value_counts().get(0, 0)
            n_rows = df.shape[0]
            percentage = (zeros / n_rows) * 100
            percentages.append(percentage)
            row.append(f"{percentage:.0f}\\%")

        average = np.mean(percentages)
        row.append(f"\\bfseries{{{average:.0f}\\%}}")
        row.append(f"\\bfseries{{{Counter(percentages)[0]}}}")
        table.add_row(row)

    table.add_line()
    table.add_header(['Running time (s)'])
    header = ['Method', 's1', 's2', 's3', 's4',
              'a1', 'a2', 'a3', 'unb', 'b1', 'b2', 'dim32']
    table.add_header(header)
    for init_method in init_methods:
        row = []
        row.append(init_method)

        for dataset in header[1:]:
            filepath = results_dir / f"{dataset}-r100" / f"{init_method}.csv"
            df = pd.read_csv(filepath)
            mean = df['elapsed_time'].mean()
            means.append(mean)
            row.append(f"{mean:.1f}")

        table.add_row(row)

    table.add_line()
    result = table.to_str()
    print(result)
    return result


def table6():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    table = LatexTable()

    table.add_caption(
        "CI mínimo atingido em 200k iterações para cada método nos datasets A3 e Unbalanced.")

    for dataset in ["a3", "unb"]:
        table.add_line()
        table.add_header([dataset])
        header = ['Método', 'CI mínimo', 'Iteração']
        table.add_header(header)
        for init_method in init_methods:
            row = []
            row.append(init_method)

            filepath = results_dir / f"{dataset}-200-r100" / f"{init_method}.csv"
            df = pd.read_csv(filepath)
            min_value = df['ci_final'].min()
            row.append(str(min_value))
            min_iter = df['ci_final'].iloc[min_value]
            row.append(str(min_iter))
            table.add_row(row)

    table.add_line()
    result = table.to_str()
    print(result)
    return result


class LatexTable:
    def __init__(self):
        self.table = []
        self.caption = ""

    def add_header(self, header):
        formatted = [f"\\bfseries{{{h}}}" for h in header]
        self.table.extend([
            f"{' & '.join(formatted)} \\\\",
            "\\hline",
        ])

    def add_row(self, row):
        self.table.append(
            f"{' & '.join(row)} \\\\"
        )

    def add_line(self):
        self.table.append(
            "\\hline",
        )

    def add_caption(self, caption):
        self.caption = caption

    def to_str(self):
        self.table.insert(0, "\\begin{table}")
        self.table.insert(1, "\\centering")
        self.table.insert(
            2, "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{cccccccccccccccccccc}")
        self.table.extend([
            "\\end{tabular}}",
            f"\\caption{{{self.caption}}}",
            "\\end{table}",
        ])
        return "\n".join(self.table)


def fig13():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    init_methods_bar = ['Rand-P', 'Rand-C', 'Maxmin', 'KMeansPP',
                        'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    high_overlap_datasets = [
        'g2-2-40',
        'g2-2-50',
        'g2-2-60',
        'g2-2-70',
        'g2-2-80',
        'g2-2-90',
        'g2-2-100',
        'g2-4-50',
        'g2-4-60',
        'g2-4-70',
        'g2-4-80',
        'g2-4-90',
        'g2-4-100',
        'g2-8-70',
        'g2-8-80',
        'g2-8-90',
        'g2-8-100',
        'g2-16-90',
        'g2-16-100',
    ]

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    axs = fig.subplots(ncols=2)

    axs[0].title.set_text(f'Baixa sobreposição')

    initial = []
    final = []
    for init_method in init_methods:
        initial_percentages = []
        final_percentages = []
        for dataset in results_dir.glob('g2*'):
            if dataset.stem in high_overlap_datasets:
                continue

            df = pd.read_csv(dataset / f"{init_method}.csv")
            n_rows = df.shape[0]
            initial_zeros = df['ci_initial'].value_counts().get(0, 0)
            initial_percentage = (initial_zeros / n_rows) * 100
            initial_percentages.append(initial_percentage)
            final_zeros = df['ci_final'].value_counts().get(0, 0)
            final_percentage = (final_zeros / n_rows) * 100
            final_percentages.append(final_percentage)
        initial.append(np.mean(initial_percentages))
        final.append(np.mean(final_percentages)-np.mean(initial_percentages))

    axs[0].bar(init_methods_bar, final, label='Final')
    axs[0].bar(init_methods_bar, initial, label='Inicial')

    axs[0].set_ylabel('Taxa de sucesso (\%)')

    axs[0].tick_params('x', labelrotation=70)

    axs[0].set_yticks([0, 20, 40, 60, 80, 100])
    axs[0].set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])

    axs[0].grid(b=False, axis='x')

    axs[1].title.set_text(f'Alta sobreposição')

    initial = []
    final = []
    for init_method in init_methods:
        initial_percentages = []
        final_percentages = []
        for dataset in high_overlap_datasets:
            df = pd.read_csv(results_dir / dataset / f"{init_method}.csv")
            n_rows = df.shape[0]
            initial_zeros = df['ci_initial'].value_counts().get(0, 0)
            initial_percentage = (initial_zeros / n_rows) * 100
            initial_percentages.append(initial_percentage)
            final_zeros = df['ci_final'].value_counts().get(0, 0)
            final_percentage = (final_zeros / n_rows) * 100
            final_percentages.append(final_percentage)
        initial.append(np.mean(initial_percentages))
        final.append(np.mean(final_percentages)-np.mean(initial_percentages))

    init_methods[3] = 'KMeansPP'
    axs[1].bar(init_methods_bar, final, label='Final')
    axs[1].bar(init_methods_bar, initial, label='Inicial')

    axs[1].tick_params('x', labelrotation=70)

    axs[1].set_yticks([0, 20, 40, 60, 80, 100])
    axs[1].set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])

    axs[1].grid(b=False, axis='x')

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig13.eps')

    return fig, str(filename)


def fig14():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    cluster_sizes = list(range(10, 101, 10))
    for init_method in init_methods:
        final_percentages = []
        for cluster_size in cluster_sizes:
            df = pd.read_csv(
                results_dir / f"b2-sub-{cluster_size}" / f"{init_method}.csv")
            n_rows = df.shape[0]
            final_zeros = df['ci_final'].value_counts().get(0, 0)
            final_percentage = (final_zeros / n_rows) * 100
            final_percentages.append(final_percentage)
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(cluster_sizes, final_percentages, label=label)

    ax.set_xticks(cluster_sizes)
    ax.set_xlabel('Clusters (k)')

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])
    ax.set_ylabel('Taxa de sucesso (\%)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig14.eps')

    return fig, str(filename)


def fig15():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    cluster_sizes = list(range(10, 101, 10))
    for init_method in init_methods:
        relative_cis = []
        for cluster_size in cluster_sizes:
            df = pd.read_csv(
                results_dir / f"b2-sub-{cluster_size}" / f"{init_method}.csv")
            n_rows = df.shape[0]
            ci_mean = df['ci_final'].mean()
            relative_cis.append(ci_mean / cluster_size)
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(cluster_sizes, relative_cis, label=label)

    ax.set_xticks(cluster_sizes)
    ax.set_xlabel('Clusters (k)')

    ax.set_ylabel('CI relativo (CI/k)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig15.eps')

    return fig, str(filename)


def fig16_1():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.title.set_text(f'Dataset DIM (CI inicial)')
    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    dimensions_sizes = ['32', '64', '128', '256', '512', '1024']

    for init_method in init_methods:
        initial_percentages = []
        for dimensions_size in dimensions_sizes:
            df = pd.read_csv(
                results_dir / f"dim{dimensions_size}" / f"{init_method}.csv")
            n_rows = df.shape[0]
            initial_zeros = df['ci_initial'].value_counts().get(0, 0)
            initial_percentage = (initial_zeros / n_rows) * 100
            initial_percentages.append(initial_percentage)
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(dimensions_sizes, initial_percentages, label=label)

    ax.set_xticks(dimensions_sizes)
    ax.set_xlabel('Dimensões')

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])
    ax.set_ylabel('Taxa de sucesso (\%)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig16_1.eps')

    return fig, str(filename)


def fig16_2():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.title.set_text(f'Dataset DIM (CI final)')
    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    dimensions_sizes = ['32', '64', '128', '256', '512', '1024']

    for init_method in init_methods:
        final_percentages = []
        for dimensions_size in dimensions_sizes:
            df = pd.read_csv(
                results_dir / f"dim{dimensions_size}" / f"{init_method}.csv")
            n_rows = df.shape[0]
            final_zeros = df['ci_final'].value_counts().get(0, 0)
            final_percentage = (final_zeros / n_rows) * 100
            final_percentages.append(final_percentage)
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(dimensions_sizes, final_percentages, label=label)

    ax.set_xticks(dimensions_sizes)
    ax.set_xlabel('Dimensões')

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])
    ax.set_ylabel('Taxa de sucesso (\%)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig16_2.eps')

    return fig, str(filename)


def fig16_3():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.title.set_text(f'Dataset G2 (CI inicial)')
    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    dimensions_sizes = ['1', '2', '4', '8', '16',
                        '32', '64', '128', '256', '512', '1024']
    std_sizes = list(range(10, 101, 10))

    for init_method in init_methods:
        initial_percentages_means = []
        for dimensions_size in dimensions_sizes:
            initial_percentages = []
            for std_size in std_sizes:
                df = pd.read_csv(
                    results_dir / f"g2-{dimensions_size}-{std_size}" / f"{init_method}.csv")
                n_rows = df.shape[0]
                initial_zeros = df['ci_initial'].value_counts().get(0, 0)
                initial_percentage = (initial_zeros / n_rows) * 100
                initial_percentages.append(initial_percentage)
            initial_percentages_means.append(np.mean(initial_percentages))
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(dimensions_sizes, initial_percentages_means, label=label)

    ax.set_xticks(dimensions_sizes)
    ax.set_xlabel('Dimensões')

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])
    ax.set_ylabel('Taxa de sucesso (\%)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig16_3.eps')

    return fig, str(filename)


def fig16_4():
    results_dir = get_project_results_dir()
    init_methods = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 7)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot()

    ax.title.set_text(f'Dataset G2 (CI final)')
    ax.set_prop_cycle(color=plt.cm.Set1.colors)

    dimensions_sizes = ['1', '2', '4', '8', '16',
                        '32', '64', '128', '256', '512', '1024']
    std_sizes = list(range(10, 101, 10))

    for init_method in init_methods:
        final_percentages_means = []
        for dimensions_size in dimensions_sizes:
            final_percentages = []
            for std_size in std_sizes:
                df = pd.read_csv(
                    results_dir / f"g2-{dimensions_size}-{std_size}" / f"{init_method}.csv")
                n_rows = df.shape[0]
                final_zeros = df['ci_final'].value_counts().get(0, 0)
                final_percentage = (final_zeros / n_rows) * 100
                final_percentages.append(final_percentage)
            final_percentages_means.append(np.mean(final_percentages))
        label = "KMeansPP" if init_method == "kmeans++" else init_method
        ax.plot(dimensions_sizes, final_percentages_means, label=label)

    ax.set_xticks(dimensions_sizes)
    ax.set_xlabel('Dimensões')

    ax.set_yticks([99, 99.20, 99.40, 99.60, 99.80, 100])
    ax.set_yticklabels(['99\%', '99,2\%', '99,4\%',
                        '99,6\%', '99,8\%', '100\%'])
    ax.set_ylabel('Taxa de sucesso (\%)')

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    filename = get_project_results_dir().joinpath('fig16_4.eps')

    return fig, str(filename)


def show_plots():
    fig13()
    fig14()
    fig15()
    fig16_1()
    fig16_2()
    fig16_3()
    fig16_4()
    plt.show()
