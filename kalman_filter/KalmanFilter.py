import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


class KalmanFilter():
    def __init__(self, dynamParams, measureParams, measurementMatrix, transitionMatrix, processNoiseCov, measurementNoiseCov):
        self.dynamParams = dynamParams
        self.measureParams = measureParams

        self.measurementMatrix = measurementMatrix
        self.transitionMatrix = transitionMatrix
        self.processNoiseCov = processNoiseCov
        self.measurementNoiseCov = measurementNoiseCov

        self.objDict = dict()

    def update(self, measurement_list, id_list):
        out_dict = dict()

        for i in range(len(id_list)):
            index = id_list[i]
            measurement = measurement_list[i]
            measurement = np.array(measurement, dtype=np.float32)

            if index not in self.objDict:
                currentKfObj = cv2.KalmanFilter(
                    self.dynamParams, self.measureParams)
                currentKfObj.measurementMatrix = self.measurementMatrix
                currentKfObj.transitionMatrix = self.transitionMatrix
                currentKfObj.processNoiseCov = self.processNoiseCov
                currentKfObj.measurementNoiseCov = self.measurementNoiseCov

                init_state = measurement.copy()
                init_state.resize(self.dynamParams, 1, refcheck=False)
                currentKfObj.statePre = init_state

                self.objDict[index] = currentKfObj
            else:
                currentKfObj = self.objDict[index]

            currentKfObj.correct(measurement)
            predict = currentKfObj.predict()
            out_dict[index] = list(float(x)
                                   for x in predict[:self.measureParams])

        return out_dict


if __name__ == "__main__":
    with open("temp.json") as fp:
        temp_dict = json.load(fp)
        index = "20"
        time_stamps = range(235, 247)
        time_stamps = map(str, time_stamps)
        temp_list = []
        for time in time_stamps:
            temp_list.append(temp_dict[index][time])

    measurement_matrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32)
    transition_matrix = np.array(
        [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
         [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
    process_noise_cov = np.eye(6, dtype=np.float32) * 1e-3
    measurement_noise_cov = np.eye(3, dtype=np.float32) * 1e-1
    kalman_filter = KalmanFilter(6, 3, measurement_matrix,
                                 transition_matrix, process_noise_cov, measurement_noise_cov)

    y_org = []
    y_filt = []

    for coord in temp_list:
        filtered = kalman_filter.update([coord], ["test"])
        print(coord, filtered["test"])
        y_org.append(coord[1])
        y_filt.append(filtered["test"][1])

    plt.plot(y_org)
    plt.plot(y_filt)
    plt.savefig('temp.png')
