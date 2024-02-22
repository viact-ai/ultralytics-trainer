from ultralytics.utils.metrics import DetMetrics
# from ultralytics.utils
from clearml.task import Task


def log_plot(title, plot_path) -> None:
    """
    Log an image as a plot in the plot section of ClearML.

    Args:
        title (str): The title of the plot.
        plot_path (str): The path to the saved image file.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False,
                      aspect='auto', xticks=[], yticks=[])  # no ticks
    ax.imshow(img)

    Task.current_task().get_logger().report_matplotlib_figure(title=title,
                                                              series='',
                                                              figure=fig,
                                                              report_interactive=False)


def parse_metrics(metrics: DetMetrics) -> dict:
    data = {}
    for i, cls in metrics.names.items():
        met_str = "details/{}"
        key = f"/{i}_{cls}"
        data[met_str.format("precision") + key] = metrics.box.p[i]
        data[met_str.format("recall") + key] = metrics.box.r[i]
        data[met_str.format("ap") + key] = metrics.box.ap[i]
        data[met_str.format("ap50") + key] = metrics.box.ap50[i]
        data[met_str.format("f1") + key] = metrics.box.f1[i]
    data.update(metrics.results_dict)
    return data


def report_metrics(metrics: DetMetrics):
    task: Task = Task.current_task()
    if task:
        files = [*(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(metrics.save_dir / f)
                 for f in files if (metrics.save_dir / f).exists()]  # filter
        for f in files:
            log_plot(title=f.stem, plot_path=f)
        # Log final results, CM matrix + PR plots
        met = parse_metrics(metrics)
        for k, v in met.items():
            task.get_logger().report_single_value(k, v)
        # Log the final model
