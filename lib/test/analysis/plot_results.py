import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import pickle
import json
from lib.test.evaluation.environment import env_settings
from lib.test.analysis.extract_results import extract_results


def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data


def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return  disp_name


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    plt.rcParams['text.usetex']=True
    plt.rcParams["font.family"] = "Times New Roman"
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylabel = "%s"%(ylabel.replace('%','\%'))
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = r"$\bf{%s}$" %(plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    try:
        # add bold to our method
        for i in range(1,2):
            legend_text[-i] = r'\textbf{%s}'%(legend_text[-i])

        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0)
    except:
        pass

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()


def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    # Load data
    settings = env_settings()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path) and not force_evaluation:
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        # print('Pre-computed evaluation data not found. Computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)

    if not check_eval_data_is_valid(eval_data, trackers, dataset):
        # print('Pre-computed evaluation data invalid. Re-computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)
        # pass
    else:
        # Update display names
        tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                         for t in trackers]
        eval_data['trackers'] = tracker_names
    with open(eval_data_path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return eval_data


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score


def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, **kwargs):
    """
    Plot results for the given trackers

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success',
                    'prec' (precision), and 'norm_prec' (normalized precision)
    """
    # Load data
    settings = env_settings()

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    print('\nGenerating plots for: {}'.format(report_name))

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 88), 'title': 'Success'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts)

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       norm_precision_plot_opts)

    plt.show()


def generate_formatted_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text


def print_results(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)


def plot_got_success(trackers, report_name):
    """ Plot success plot for GOT-10k dataset using the json reports.
    Save the json reports from http://got-10k.aitestunion.com/leaderboard in the directory set to
    env_settings.got_reports_path

    The tracker name in the experiment file should be set to the name of the report file for that tracker,
    e.g. DiMP50_report_2019_09_02_15_44_25 if the report is name DiMP50_report_2019_09_02_15_44_25.json

    args:
        trackers - List of trackers to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load data
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    # Load results
    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1:
            raise Exception

        # First field is the tracker name. Index it out
        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id,
                              'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, **kwargs):
    """ Print per-sequence results for the given trackers. Additionally, the sequences to list can be filtered using
    the filter criteria.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        filter_criteria - Filter sequence results which are reported. Following modes are supported
                        None: No filtering. Display results for all sequences in dataset
                        'ao_min': Only display sequences for which the minimum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences where at least one tracker performs poorly.
                        'ao_max': Only display sequences for which the maximum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences all tracker performs poorly.
                        'delta_ao': Only display sequences for which the performance of different trackers vary by at
                                    least filter_criteria['threshold'] in average overlap (AO) score. This mode can
                                    be used to select sequences where the behaviour of the trackers greatly differ
                                    between each other.
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    # Filter sequences
    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist())) if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)

    print(report_text)


def print_results_per_video(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), per_video=False, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    seq_lens = len(eval_data['sequences'])
    eval_datas = [{} for _ in range(seq_lens)]
    if per_video:
        for key, value in eval_data.items():
            if len(value) == seq_lens:
                for i in range(seq_lens):
                    eval_datas[i][key] = [value[i]]
            else:
                for i in range(seq_lens):
                    eval_datas[i][key] = value

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)

    if per_video:
        for i in range(seq_lens):
            eval_data = eval_datas[i]

            print('\n{} sequences'.format(eval_data['sequences'][0]))

            scores = {}
            valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

            # ********************************  Success Plot **************************************
            if 'success' in plot_types:
                threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
                ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

                # Index out valid sequences
                auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
                scores['AUC'] = auc
                scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
                scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

            # ********************************  Precision Plot **************************************
            if 'prec' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

                # Index out valid sequences
                prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
                scores['Precision'] = prec_score

            # ********************************  Norm Precision Plot *********************************
            if 'norm_prec' in plot_types:
                ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

                # Index out valid sequences
                norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
                scores['Norm Precision'] = norm_prec_score

            # Print
            tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
            report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
            print(report_text)

def plot_draw_save_per_seq(y, x, scores, trackers, plot_draw_styles, result_plot_path, seq_name):
    """
    专门为单视频设计的绘图函数，文件名以视频名为准
    """
    # 保持和原代码一致的样式设置
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 20
    
    # 定义绘图参数
    plot_opts = {
        'font_size': 20, 'font_size_axis': 20, 'line_width': 2, 'font_size_legend': 20,
        'legend_loc': 'lower left',
        'xlabel': 'Overlap threshold', 'ylabel': 'Overlap Precision [\%]', 
        'xlim': (0, 1.0), 'ylim': (0, 100)
    }

    matplotlib.rcParams.update({'font.size': plot_opts['font_size']})
    matplotlib.rcParams.update({'axes.titlesize': plot_opts['font_size_axis']})
    matplotlib.rcParams.update({'axes.labelsize': plot_opts['font_size_axis']})

    fig, ax = plt.subplots()

    # 排序：按当前视频的 AUC 排序
    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        # 获取样式（保持跟总图一致的颜色分配）
        style_idx = index_sort.numel() - id - 1
        color = plot_draw_styles[style_idx]['color']
        line_style = plot_draw_styles[style_idx]['line_style']

        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=plot_opts['line_width'],
                       color=color,
                       linestyle=line_style)

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    # 图例加粗第一名
    try:
        legend_text[-1] = r'\textbf{%s}'%(legend_text[-1])
        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=plot_opts['legend_loc'], 
                  fancybox=False, edgecolor='black',
                  fontsize=plot_opts['font_size_legend'], framealpha=1.0)
    except:
        pass

    # 设置标题为视频名称
    ax.set(xlabel=plot_opts['xlabel'],
           ylabel=plot_opts['ylabel'],
           xlim=plot_opts['xlim'], ylim=plot_opts['ylim'],
           title=r"\textbf{%s}" % (seq_name.replace('_', '\_'))) # 转义下划线防止Latex报错

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    # --- 保存文件 ---
    # 文件名直接用视频名
    save_path = os.path.join(result_plot_path, '{}.pdf'.format(seq_name))
    fig.savefig(save_path, dpi=300, format='pdf', transparent=True)
    
    # 释放内存，防止循环画几百张图导致内存爆炸
    plt.close(fig) 


def plot_success_per_sequence(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    """
    主函数：遍历数据集中的每个视频，生成单独的 Success Plot
    """
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    # 1. 加载所有数据
    # 这里我们利用已有的逻辑，它会计算好所有的结果并缓存在 eval_data.pkl 中
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)
    
    # 2. 准备保存目录
    # 在原本的 report 目录下新建一个 per_sequence_plots 文件夹
    base_plot_path = os.path.join(settings.result_plot_path, report_name)
    per_seq_save_path = os.path.join(base_plot_path, 'per_sequence_plots')
    
    if not os.path.exists(per_seq_save_path):
        os.makedirs(per_seq_save_path)
        print(f"Created directory: {per_seq_save_path}")

    # 3. 获取核心数据张量
    # 形状: [Num_Sequences, Num_Trackers, 101]
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
    
    tracker_names = eval_data['trackers']
    sequences = eval_data['sequences'] # 视频名称列表
    
    print(f"Start generating plots for {len(sequences)} sequences...")

    # 4. 遍历每个视频进行绘图
    for i, seq_name in enumerate(sequences):
        # 取出第 i 个视频的数据
        # shape: [Num_Trackers, 101]
        seq_data = ave_success_rate_plot_overlap[i, :, :]
        
        # 计算该视频下的 AUC (Mean over threshold axis)
        # shape: [Num_Trackers]
        seq_auc = seq_data.mean(1) * 100.0
        
        # 数据转换 (0-1 -> 0-100 for y-axis)
        seq_data_plot = seq_data * 100.0
        
        # 绘图并保存
        plot_draw_save_per_seq(seq_data_plot, threshold_set_overlap, seq_auc, 
                               tracker_names, plot_draw_styles, per_seq_save_path, seq_name)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sequences)}: {seq_name}")

    print(f"\nAll per-sequence plots saved to: {per_seq_save_path}")

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from lib.test.evaluation.environment import env_settings

def compute_iou_for_sequence(gt_boxes, pred_boxes):
    """
    输入:
        gt_boxes: [N, 4] numpy array
        pred_boxes: [N, 4] numpy array
    输出:
        iou: [N] numpy array
    """
    # 确保长度对齐
    min_len = min(len(gt_boxes), len(pred_boxes))
    gt_boxes = gt_boxes[:min_len]
    pred_boxes = pred_boxes[:min_len]

    # 批量计算交集
    x1 = np.maximum(gt_boxes[:, 0], pred_boxes[:, 0])
    y1 = np.maximum(gt_boxes[:, 1], pred_boxes[:, 1])
    x2 = np.minimum(gt_boxes[:, 0] + gt_boxes[:, 2], pred_boxes[:, 0] + pred_boxes[:, 2])
    y2 = np.minimum(gt_boxes[:, 1] + gt_boxes[:, 3], pred_boxes[:, 1] + pred_boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 计算并集
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    union_area = gt_area + pred_area - inter_area

    # 计算 IoU，处理除零异常
    iou = np.zeros(min_len)
    mask = union_area > 0
    iou[mask] = inter_area[mask] / union_area[mask]

    return iou


def plot_iou_curve_per_video(trackers, dataset, report_name):
    """
    为数据集中的每个视频生成：帧号 vs. IoU 的曲线图
    (修正版：解决了 Tracker 对象属性访问报错的问题)
    """
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles() 

    # 1. 创建保存目录
    base_plot_path = os.path.join(settings.result_plot_path, report_name)
    iou_plot_path = os.path.join(base_plot_path, 'per_sequence_iou_plots')
    
    if not os.path.exists(iou_plot_path):
        os.makedirs(iou_plot_path)
        print(f"Created directory: {iou_plot_path}")

    print(f"Start plotting IoU curves for {len(dataset)} sequences...")

    # 全局绘图设置
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 18

    for seq_id, seq in enumerate(dataset):
        seq_name = seq.name
        gt_boxes = np.array(seq.ground_truth_rect)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        has_valid_plot = False

        for trk_id, tracker in enumerate(trackers):
            # --- 路径构建 ---
            if tracker.run_id is not None:
                res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, 
                                        '{:03d}'.format(tracker.run_id), seq_name + '.txt')
            else:
                res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, 
                                        seq_name + '.txt')
            
            # 容错：如果找不到，尝试不带 run_id 的路径
            if not os.path.exists(res_file):
                 res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, 
                                        seq_name + '.txt')

            if os.path.exists(res_file):
                try:
                    # --- 读取预测结果 ---
                    try:
                        pred_boxes = np.loadtxt(res_file, delimiter=',')
                    except:
                        pred_boxes = np.loadtxt(res_file)
                    
                    if pred_boxes.ndim == 1: 
                        pred_boxes = pred_boxes[None, :]

                    # --- 计算 IoU ---
                    iou_series = compute_iou_for_sequence(gt_boxes, pred_boxes)
                    
                    # --- 获取显示名称 (修正点：手动构建名称，不调用 get_tracker_display_name) ---
                    # 检查 tracker 是否有 display_name 属性
                    d_name = getattr(tracker, 'display_name', None)
                    if d_name is not None:
                        disp_name = d_name
                    else:
                        # 手动拼接 name_param_runid
                        run_id_str = f"_{tracker.run_id:03d}" if tracker.run_id is not None else ""
                        disp_name = f"{tracker.name}_{tracker.parameter_name}{run_id_str}"

                    # --- 绘图 ---
                    style = plot_draw_styles[trk_id % len(plot_draw_styles)]
                    
                    ax.plot(range(len(iou_series)), iou_series, 
                            color=style['color'], 
                            linestyle=style['line_style'],
                            linewidth=1.5,
                            label=f"{disp_name} ({iou_series.mean():.2f})") # 图例只保留两位小数
                    
                    has_valid_plot = True
                except Exception as e:
                    print(f"Error processing {seq_name} for {tracker.name}: {e}")
            else:
                pass # 没找到结果文件就跳过

        if has_valid_plot:
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.5')
            
            # 标题转义下划线，防止 LaTeX 报错
            safe_title = seq_name.replace('_', '\_')
            ax.set_title(r"\textbf{%s}" % safe_title, fontsize=font_size)
            
            ax.set_xlabel('Frame Number', fontsize=font_size)
            ax.set_ylabel('IoU Score', fontsize=font_size)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, len(gt_boxes))
            
            ax.legend(loc='lower left', fancybox=False, edgecolor='black', framealpha=0.8, fontsize=12)
            ax.grid(True, linestyle='-.', alpha=0.5)
            
            save_path = os.path.join(iou_plot_path, f'{seq_name}.pdf')
            plt.tight_layout()
            fig.savefig(save_path, dpi=300, format='pdf')
        
        plt.close(fig)

        if (seq_id + 1) % 10 == 0:
            print(f"Processed {seq_id + 1}/{len(dataset)} sequences.")

    print(f"\nDone! IoU plots saved to: {iou_plot_path}")

import cv2
import os
import numpy as np
from lib.test.evaluation.environment import env_settings

def visualize_sequence(tracker, dataset, seq_name, save_video=True, show_gt=True):
    """
    可视化指定序列的跟踪结果
    Args:
        tracker: 跟踪器对象
        dataset: 数据集对象
        seq_name: 目标序列名称 (字符串, 如 'Basketball')
        save_video: True则保存为mp4视频, False则保存为每一帧的图片
        show_gt: 是否画出 Ground Truth (绿色)
    """
    settings = env_settings()
    
    # 1. 找到对应的序列对象
    target_seq = None
    for seq in dataset:
        if seq.name == seq_name:
            target_seq = seq
            break
    
    if target_seq is None:
        print(f"Error: Sequence '{seq_name}' not found in dataset.")
        return

    # 2. 读取预测结果 (Pred)
    # 构建路径 (适配 run_id)
    if tracker.run_id is not None:
        res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, 
                                '{:03d}'.format(tracker.run_id), seq_name + '.txt')
    else:
        res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, 
                                seq_name + '.txt')

    if not os.path.exists(res_file):
        # 再次尝试无 run_id 的路径
        res_file = os.path.join(settings.results_path, tracker.name, tracker.parameter_name, seq_name + '.txt')
        if not os.path.exists(res_file):
            print(f"Error: Result file not found: {res_file}")
            return

    # 加载结果
    try:
        pred_boxes = np.loadtxt(res_file, delimiter=',')
    except:
        pred_boxes = np.loadtxt(res_file)
    
    if pred_boxes.ndim == 1:
        pred_boxes = pred_boxes[None, :]

    # 3. 获取 Ground Truth (GT)
    gt_boxes = np.array(target_seq.ground_truth_rect)

    # 4. 准备输出目录
    # 保存到 results_plot/viz/seq_name/ 下
    base_save_path = os.path.join(settings.result_plot_path, 'visualization', seq_name)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    
    print(f"Visualizing sequence: {seq_name}...")
    print(f"Saving output to: {base_save_path}")

    # 5. 视频写入器初始化 (如果需要保存视频)
    video_writer = None
    if save_video:
        # 读取第一帧获取图像尺寸
        first_img = cv2.imread(target_seq.frames[0])
        h, w, _ = first_img.shape
        video_path = os.path.join(base_save_path, f'{seq_name}_{tracker.name}.mp4')
        # mp4v 是比较通用的编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    # 6. 逐帧处理
    # 长度以 seq.frames 为准
    num_frames = len(target_seq.frames)
    
    for i in range(num_frames):
        frame_path = target_seq.frames[i]
        img = cv2.imread(frame_path)
        
        if img is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue

        # --- 画 GT (绿色) ---
        if show_gt and i < len(gt_boxes):
            gt = gt_boxes[i]
            # 只有当 GT 均非 0 且非 NaN 时才画
            if not np.any(np.isnan(gt)) and np.any(gt > 0):
                x, y, w, h = map(int, gt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, 'GT', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 画 Pred (红色) ---
        if i < len(pred_boxes):
            pred = pred_boxes[i]
            # 同样检查有效性
            if not np.any(np.isnan(pred)):
                x, y, w, h = map(int, pred)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # 计算并显示当前帧 IoU
                iou_str = ""
                if i < len(gt_boxes):
                    cur_iou = compute_iou_single(gt_boxes[i], pred)
                    iou_str = f" IoU: {cur_iou:.2f}"
                
                label = f"{tracker.name}{iou_str}"
                cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- 显示帧号 ---
        cv2.putText(img, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- 保存 ---
        if save_video:
            video_writer.write(img)
        else:
            # 保存为图片
            img_save_path = os.path.join(base_save_path, f'{i:04d}.jpg')
            cv2.imwrite(img_save_path, img)
            
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{num_frames} frames")

    # 释放资源
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {video_path}")
    else:
        print(f"Images saved in: {base_save_path}")

def compute_iou_single(box1, box2):
    """辅助函数：计算单帧 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0: return 0
    return inter_area / union_area


def print_attribute_results(trackers, dataset, report_name, merge_results=False, **kwargs):
    """
    Print the results broken down by sequence attributes (e.g., IV, SV, OCC, etc.)
    Usually used for LaSOT, OTB, UAV123 datasets.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate (must contain 'attributes' property)
        report_name - Name of the folder where results are saved
        merge_results - If True, multiple random runs are averaged
    """
    # 1. Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs if needed
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    # Data shape: [Num_Sequences, Num_Trackers, 101]
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    
    # Map sequence names to their index in the eval_data
    # We need this because dataset list order might conceptually differ or we need fast lookup
    seq_name_to_idx = {name: i for i, name in enumerate(eval_data['sequences'])}
    
    # 2. Group sequences by attributes
    # Structure: {'IV': [idx1, idx2...], 'SV': [idx3, idx4...]}
    attr_indices = {}
    
    # Standard LaSOT attributes for sorting order (optional, makes table look standard)
    lasot_order = ['IV', 'SV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR', 'OV', 'BC', 'LR', 'VC', 'CM', 'ROT', 'POC']
    
    print("Aggregating results by attributes...")
    
    for seq in dataset:
        if seq.name in seq_name_to_idx:
            idx = seq_name_to_idx[seq.name]
            
            # Check if sequence has attributes
            if not hasattr(seq, 'attributes'):
                continue
                
            attrs = seq.attributes
            # Handle case where attributes might be a single string or empty
            if not attrs: 
                continue
            
            for attr in attrs:
                if attr not in attr_indices:
                    attr_indices[attr] = []
                attr_indices[attr].append(idx)

    # 3. Compute AUC for each attribute
    scores = {}
    
    # Sort attributes: specific order first, then others alphabetically
    sorted_attrs = sorted(attr_indices.keys(), key=lambda x: (lasot_order.index(x) if x in lasot_order else 999, x))

    for attr in sorted_attrs:
        indices = torch.tensor(attr_indices[attr], dtype=torch.long)
        
        if len(indices) == 0:
            continue
            
        # Select data for this attribute: [Num_Attr_Seqs, Num_Trackers, 101]
        attr_data = ave_success_rate_plot_overlap[indices, :, :]
        
        # Calculate AUC: Mean over sequences (dim 0) -> Mean over thresholds (dim 1)
        # Result shape: [Num_Trackers]
        attr_auc = attr_data.mean(0).mean(1) * 100.0
        
        # Add sequence count to column name for clarity, e.g., "IV (15)"
        col_name = "{} ({})".format(attr, len(indices))
        scores[col_name] = attr_auc

    # 4. Print Table
    if not scores:
        print("No attributes found in the dataset sequences.")
        return

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    
    print('\nAttribute-based Performance (AUC):')
    # Use the existing report generator
    # Note: If there are too many attributes, the table might wrap in the console.
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name='Attributes')
    print(report_text)