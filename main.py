import csv
import math

import numpy as np
from matplotlib import pyplot as plt

import logging as log
log.basicConfig(filename='out.md', filemode='w', format='%(message)s', level=log.INFO)

TRAIN_PARAMS = [
    [1500, 0.005],
    [500, 0.001], [500, 0.0005], [500, 0.0005], [1000, 0.0005],  [1000, 0.0005],
    [1000, 0.0005], [1000, 0.0005], [2000, 0.0005],  [500, 0.0005], [500, 0.0001], [500, 0.0005],
    [500, 0.0005], [500, 0.0005], [1000, 0.0005],  [1000, 0.0005], [1000, 0.0001], [1000, 0.0001],
    [1000, 0.0001], [1000, 0.0001], [1000, 0.0001], [1000, 0.0001], [1000, 0.00005], [1000, 0.00001],
    [1000, 0.00005]
]

DATA_FILE = 'data.csv'

TRAINING_MSE_FILE = 'output/Training_MSE_plot.jpg'
TEST_MSE_FILE = 'output/Test_MSE_plot.jpg'
APPROX_KGF_FILE = 'output/Kgf_Approx_accuracy_plot.jpg'
APPROX_GT_FILE = 'output/Gt_Approx_accuracy_plot.jpg'
KGF_FILE = 'output/Kgf_plot.jpg'
GT_FILE = 'output/Gt_plot.jpg'


def save_weights(file_name: str, layers: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    layer_1, layer_2, layer_3 = layers
    np.savez(file_name, l1=layer_1, l2=layer_2, l3=layer_3)


def normalize(array, array_max, array_min, out_max, out_min):
    return (array - array_min) / (array_max - array_min) * (out_max - out_min) + out_min


def load_data(
        data_file_name: str = DATA_FILE) -> (tuple[np.ndarray, np.ndarray],
                                             tuple[np.ndarray, np.ndarray],
                                             tuple[np.ndarray, np.ndarray]):
    with open(data_file_name, encoding='UTF-8') as data_file:
        reader = csv.reader(data_file, delimiter=',')
        rows = list(reader)

        data = np.asanyarray(rows, dtype=float)
        dataset: np.ndarray = data[:, :-2]
        targets: np.ndarray = data[:, -2:]

        kgf_max = np.nanmax(targets[:, -1:])
        kgf_min = np.nanmin(targets[:, -1:])

        gtotal_max = np.nanmax(targets[:, :-1])
        gtotal_min = np.nanmin(targets[:, :-1])

        out_max = 1
        out_min = 0

        targets[:, 0] = normalize(targets[:, 0], gtotal_max, gtotal_min, out_max, out_min)
        targets[:, 1] = normalize(targets[:, 1], kgf_max, kgf_min, out_max, out_min)

        data_indices = list(range(dataset.shape[0]))

        train_part = 0.72

        train_size = round(len(data_indices) * train_part)
        train_indices = data_indices[:train_size]
        test_indices = data_indices[train_size:]

    dataset = dataset[data_indices[:]]
    targets = targets[data_indices[:]]

    train_data = dataset[train_indices].copy()
    train_target = targets[train_indices].copy()

    test_data = dataset[test_indices].copy()
    test_target = targets[test_indices].copy()

    return (dataset, targets), (train_data, train_target), (test_data, test_target)


def zero_nans(array: np.ndarray) -> None:
    array[np.isnan(array)] = 0.0


def to_one_row_mat(array: np.ndarray) -> np.ndarray:
    return array.reshape(1, array.shape[0])


def make_layer(inputs_count: int, outputs_count: int, weight_index: int) -> np.ndarray:
    return np.random.uniform(
        0.01 * math.pow(0.2, weight_index),
        0.02 * math.pow(0.2, weight_index),
        size=(inputs_count, outputs_count))


def make_output_buffer(layers: tuple[np.ndarray, ...], data: np.ndarray) -> tuple[np.ndarray, ...]:
    layers_count = len(layers)
    outputs = (np.empty(shape=data.shape),) + tuple(
        np.empty(shape=(data.shape[0], layers[i].shape[1])) for i in range(layers_count))
    return outputs


def make_neurons(*neurons_count) -> (tuple[np.ndarray, ...], tuple[np.ndarray, ...]):
    layers_count: int = len(neurons_count) - 1
    layers: list[np.ndarray] = []
    weight_corrections: list[np.ndarray] = []

    weight_index = layers_count - 1

    for index in range(layers_count):
        inputs_count = neurons_count[index] + 1
        outputs_count = neurons_count[index + 1]

        layers.append(make_layer(inputs_count, outputs_count, weight_index))
        weight_corrections.append(np.empty(shape=(inputs_count, outputs_count)))

        weight_index -= 1

    return tuple(layers), tuple(weight_corrections[::-1])


def weight_correction(
        current_layer_input: np.ndarray,
        current_layer_output: np.ndarray,
        next_layer: np.ndarray,
        next_layer_delta: np.ndarray) -> (np.ndarray, np.ndarray):
    w_times_delta_sum = next_layer_delta @ next_layer.T
    delta = w_times_delta_sum[:, 1:] * current_layer_output * (1.0 - current_layer_output)

    return delta, current_layer_input.T @ delta


def copy_to(dst: np.ndarray, src: np.ndarray) -> None:
    n = len(dst)
    for i in range(n):
        dst[i] = src[i]


def append_ones_column(array: np.ndarray) -> np.ndarray:
    return np.concatenate(((np.ones(shape=(array.shape[0], 1))), array), axis=1)


def weight_correction_last(
        output_layer_input: np.ndarray,
        output_layer_output: np.ndarray,
        target: np.ndarray) -> (np.ndarray, np.ndarray):
    delta = output_layer_output * (1.0 - output_layer_output) * (output_layer_output - target)
    zero_nans(delta)

    return delta, output_layer_input.T @ delta


def back_propagate(
        neurons: tuple[np.ndarray, ...],
        target: np.ndarray,
        outputs_buffer: tuple[np.ndarray, ...],
        weight_corrections_buffer: tuple[np.ndarray]) -> None:
    layers_count = len(neurons)

    output_layer_input = append_ones_column(outputs_buffer[-2])
    output_layer_output = outputs_buffer[-1]

    delta, correction = weight_correction_last(output_layer_input, output_layer_output, target)

    copy_to(weight_corrections_buffer[0], correction)
    i: int = 1

    for layer_index in range(2, layers_count + 1):
        current_layer_input = append_ones_column(outputs_buffer[-layer_index - 1])
        current_layer_output = outputs_buffer[-layer_index]

        delta, correction = weight_correction(
            current_layer_input, current_layer_output, neurons[-layer_index + 1], delta)

        copy_to(weight_corrections_buffer[i], correction)
        i += 1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_layer(layer: np.ndarray, data: np.ndarray) -> np.ndarray:
    return sigmoid(data @ layer)


def feed_forward(
        layers: tuple[np.ndarray, ...],
        data: np.ndarray,
        outputs: tuple[np.ndarray, ...]) -> None:
    copy_to(outputs[0], data)
    for i, layer in enumerate(layers):
        copy_to(outputs[i + 1], apply_layer(layer, append_ones_column(outputs[i])))


def learn(
        neurons: tuple[np.ndarray, ...],
        train_data: np.ndarray,
        train_target: np.ndarray,
        learning_rate: float,
        outputs_buffer: tuple[np.ndarray, ...],
        weight_corr_buffer: tuple[np.ndarray]) -> None:
    layers_count = len(neurons)

    for i in range(len(train_data)):
        feed_forward(neurons, to_one_row_mat(train_data[i]), outputs_buffer)
        back_propagate(neurons, to_one_row_mat(train_target[i]), outputs_buffer, weight_corr_buffer)

        idx = len(weight_corr_buffer) - 1
        for j in range(layers_count):
            copy_to(neurons[j], neurons[j] - weight_corr_buffer[idx] * learning_rate)
            idx -= 1


def predict(neurons: tuple[np.ndarray, ...], data: np.ndarray, outputs_buffer: tuple[np.ndarray]) -> np.ndarray:
    feed_forward(neurons, data, outputs_buffer)
    return outputs_buffer[-1]


def mse_loss(output: np.ndarray, target: np.ndarray) -> float:
    diff = output - target
    zero_nans(diff)
    return np.mean(diff ** 2)


def train(
        train_neurons: tuple[np.ndarray, ...],
        train_dataset: np.ndarray,
        train_targets: np.ndarray,
        test_dataset: np.ndarray,
        test_targets: np.ndarray,
        train_errors: list[float],
        test_errors: list[float],
        epoch_count: int,
        learning_rate: float,
        single_row_outputs_buffer: tuple[np.ndarray, ...],
        train_output_buffer: tuple[np.ndarray, ...],
        test_output_buffer: tuple[np.ndarray, ...],
        weight_corrections_buffer: tuple[np.ndarray, ...]):
    for ind in range(epoch_count):
        learn(train_neurons, train_dataset, train_targets, learning_rate, single_row_outputs_buffer,
              weight_corrections_buffer)

        train_predict = predict(train_neurons, train_dataset, train_output_buffer)
        test_predict = predict(train_neurons, test_dataset, test_output_buffer)

        train_error = mse_loss(train_predict, train_targets)
        test_error = mse_loss(test_predict, test_targets)

        train_errors.append(train_error)
        test_errors.append(test_error)


def rmse_loss(output: np.ndarray, target: np.ndarray) -> float:
    return mse_loss(output, target) ** 0.5


def plot_results(
        neurons: tuple[np.ndarray, ...],
        all_data: np.ndarray,
        all_target: np.ndarray,
        test_data: np.ndarray,
        test_target: np.ndarray,
        test_errors: list[float],
        train_errors: list[float]):

    log.info("### Results")

    plt.plot(train_errors)
    plt.title('Training sample MS-Error')
    plt.savefig(TRAINING_MSE_FILE)
    plt.clf()

    plt.plot(test_errors)
    plt.title('Test sample MS-Error')
    plt.savefig(TEST_MSE_FILE)
    plt.clf()

    output = predict(neurons, test_data, make_output_buffer(neurons, test_data))
    data_points = list(range(len(test_data)))
    log.info("- Test sample MS-Error: ***" + f"{mse_loss(output, test_target):.5f}" + "***")

    g_total_rmse_error = rmse_loss(output[:, 0], test_target[:, 0])
    kgf_rmse_error = rmse_loss(output[:, 1], test_target[:, 1])

    log.info("- G_total RMS-Error: ***" + f"{g_total_rmse_error:.5f}" + "***")
    log.info("- Kgf RMS-Error: ***" + f"{kgf_rmse_error:.5f}" + "***")

    plt.plot(data_points, output[:, 0], test_target[:, 0])
    plt.title('G_total Test Approximation accuracy')
    plt.savefig(APPROX_GT_FILE)
    plt.clf()


    plt.plot(data_points, output[:, 1], test_target[:, 1])
    plt.title('KGF Test Approximation accuracy')
    plt.tight_layout()
    plt.savefig(APPROX_KGF_FILE)
    plt.clf()

    output = predict(neurons, all_data, make_output_buffer(neurons, all_data))
    data_points = list(range(len(all_data)))
    plt.plot(data_points, output[:, 0], all_target[:, 0])
    plt.title('G_total Approximation accuracy')
    plt.legend(('Predicted', 'Expected'))
    plt.tight_layout()
    plt.savefig(GT_FILE)
    plt.clf()

    plt.plot(data_points, output[:, 1], all_target[:, 1])
    plt.title('KGF Approximation accuracy')
    plt.legend(('Predicted', 'Expected'))
    plt.tight_layout()
    plt.savefig(KGF_FILE)


def main() -> None:
    (data, target), (train_data, train_target), (test_data, test_target) = load_data(DATA_FILE)
    zero_nans(data)
    zero_nans(train_data)
    zero_nans(test_data)

    log.info("### Dataset sizes")
    log.info("- Training sample: ***" + str(len(train_data)) + "*** rows")
    log.info("- Test sample: ***" + str(len(test_data)) + "*** rows")

    input_layer_count = train_data.shape[1]
    first_layer_count = 48
    second_layer_count = 59
    output_layer = 2

    log.info("### Layers sizes")
    log.info("- First layer: ***" + str(first_layer_count) + "*** neurons")
    log.info("- Second layer: ***" + str(second_layer_count) + "*** neurons")

    neurons, weight_corrections_buffer = make_neurons(input_layer_count, first_layer_count, second_layer_count,
                                                      output_layer)

    single_row_outputs_buffer = make_output_buffer(neurons, to_one_row_mat(train_data[0]))
    train_output_buffer = make_output_buffer(neurons, train_data)
    test_output_buffer = make_output_buffer(neurons, test_data)

    train_errors = []
    test_errors = []

    total_epoch_count = 0

    for i, (epoch_count, learning_rate) in enumerate(TRAIN_PARAMS):
        train(
            neurons, train_data, train_target, test_data, test_target,
            train_errors, test_errors, epoch_count, learning_rate,
            single_row_outputs_buffer, train_output_buffer, test_output_buffer,
            weight_corrections_buffer)
        total_epoch_count += epoch_count

    log.info("### Epoch info")
    log.info("- Total count: ***" + str(total_epoch_count) + "*** epochs")

    plot_results(neurons, data, target, test_data, test_target, train_errors, test_errors)


if __name__ == '__main__':
    main()
